import math
import numpy as np
import os
import shutil
import json
import re
import glob  
import sys
import itertools
import pandas as pd 

from openfold.utils.script_utils import load_model_w_intrinsic_param, parse_fasta, run_model_w_intrinsic_dim, prep_output, \
    update_timings, relax_protein

import subprocess 
import pickle

import random
import torch
from torch import nn

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'

def write_timings(timing_dict, output_dir_base, timing_fname):
    output_dir = '%s/timings' % output_dir_base
    os.makedirs(output_dir, exist_ok=True)
    output_file = '%s/%s.json' % (output_dir, timing_fname) 
    with open(output_file, "w") as f:
        json.dump(timing_dict, f)

def remove_files_in_dir(path):
    file_list = glob.glob('%s/*' % path)
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)

def remove_files(file_list):
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)

def remove_trailing_zeros(number):
    number_str = str(number)
    number_str = number_str.rstrip('0').rstrip('.') if '.' in number_str else number_str
    return(number_str)

def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]

def get_latest_version_num(fine_tuning_save_dir):
    latest_version_num = -1
    if os.path.isdir(fine_tuning_save_dir):
        items = os.listdir(fine_tuning_save_dir)
        sub_dir = [item for item in items if os.path.isdir(os.path.join(fine_tuning_save_dir, item))]
        version_num_list = [] 
        for d in sub_dir:
            if 'version' in d:
                version_num = int(d.split('_')[1])
                version_num_list.append(version_num)
        latest_version_num = max(version_num_list)
    return latest_version_num


def calc_disordered_percentage(pdb_path):
 
    dssp_tuple = dssp_dict_from_pdb_file(pdb_path)
    dssp_dict = dssp_tuple[0]
    ss_all = [] 
    for key in dssp_dict:
        ss = dssp_dict[key][1]
        ss_all.append(ss)

    print(ss_all)

    disordered_num = 0 
    for s in ss_all:
        if s in ['-','T','S']:
            disordered_num += 1
    disordered_percentage = (disordered_num/len(ss_all))*100
 
    return disordered_percentage

def accept_criteria(mean_plddt, disordered_percentage, mean_plddt_threshold, disordered_percentage_threshold):

    if (mean_plddt > mean_plddt_threshold) and (disordered_percentage < disordered_percentage_threshold):
        return 1
    else:
        return 0


def get_intrinsic_param_matrix(model_train_out_dir):

    out_fname = '%s/intrinsic_param_data.csv' % model_train_out_dir
    if os.path.isfile(out_fname):
        out = pd.read_csv(out_fname)
        return out 
        
    metrics_df = pd.read_csv('%s/metrics.csv' % model_train_out_dir)
    drmsd_ca_all = list(metrics_df['train/drmsd_ca'].dropna())
    lddt_ca_all = list(metrics_df['train/lddt_ca'].dropna())
    step_all = list(set(list(metrics_df['step'].dropna())))

    dim_size = 0 
    all_intrinsic_param = [] 
    all_filepaths = ['%s/step_%d.pt' % (model_train_out_dir,s) for s in step_all]
    for i,f in enumerate(all_filepaths):
        d = torch.load(f)
        intrinsic_param = d['intrinsic_parameter'].tolist()
        dim_size = len(intrinsic_param)
        drmsd_ca = drmsd_ca_all[i]
        lddt_ca = lddt_ca_all[i]
        step = step_all[i]
        intrinsic_param.append(drmsd_ca)
        intrinsic_param.append(step)
        all_intrinsic_param.append(intrinsic_param)

    cols = [('v%d' % i) for i in range(0,dim_size)]
    cols.append('drmsd_ca')
    cols.append('step_num')

    out = pd.DataFrame(all_intrinsic_param, columns = cols)
    out.to_csv(out_fname, index=False)
    return out  

def sample_from_normal(data, num_samples):
    mean = np.mean(data)
    std = np.std(data)
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    return samples 

#https://stats.stackexchange.com/questions/26185/estimation-of-white-noise-parameters-in-gaussian-random-walk-model
def get_sigma(intrinsic_dim, model_train_out_dir):
    intrinsic_param_matrix = get_intrinsic_param_matrix(model_train_out_dir)
    intrinsic_param_matrix = intrinsic_param_matrix.iloc[:,0:intrinsic_dim].to_numpy()
    intrinsic_param_diff = np.diff(intrinsic_param_matrix, axis=0)
    sigma = np.std(intrinsic_param_diff, axis=0)
    return sigma 

#https://stats.stackexchange.com/questions/62850/obtaining-covariance-matrix-from-correlation-matrix
def get_random_cov(sigma, random_corr):
    random_cov = (sigma[:, np.newaxis]*random_corr)*sigma
    return random_cov

def autopopulate_state_history_dict(state_history_dict, grid_search_combinations, optimal_combination, num_total_steps):
    for i,items in enumerate(grid_search_combinations):
        if items != optimal_combination:
            state_history_dict[items] = [-1]*num_total_steps 
    return state_history_dict
         

def populate_rw_hp_dict(rw_type, items, is_multimer):

    rw_hp_dict = {} 

    if is_multimer:
        rw_hp_dict['epsilon_scaling_factor'] = list(items)
    else:
        if rw_type == 'vanila':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
        elif rw_type == 'discrete_ou':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
            rw_hp_dict['alpha'] = items[1]
        elif rw_type == 'rw_w_momentum':
            rw_hp_dict['epsilon_scaling_factor'] = items[0]
            rw_hp_dict['gamma'] = items[1]

    return rw_hp_dict



def run_grid_search(grid_search_combinations, state_history_dict, source_str, target_str, pdb_path_initial, intrinsic_dim, rw_type, num_total_steps, L, cov_type, model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_config_data, output_dir, phase, args, save_intrinsic_param, is_multimer=False):

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

    #to speed things up when searching for scaling factors, we evaluate just the min(grid_search_combinations) and max(grid_search_combinations) when beginning a new search with parameters that have not been previously evaluated
    ###based on the results of the min(grid_search_combinations) and max(grid_search_combinations) we may terminate the search early
    ###if only a single combination exists in grid_search_combinations or running for combinations that have been previously evaluated, run for all combinations 

    if len(grid_search_combinations) == 1 or len(state_history_dict) > 0:

        for i,items in enumerate(grid_search_combinations): 
            rw_hp_dict = populate_rw_hp_dict(rw_type, items, is_multimer)
            print(asterisk_line)
            print('EVALUATING RW HYPERPARAMETERS:')
            print(rw_hp_dict)
            print(asterisk_line)

            if is_multimer:
                rw_hp_output_dir = '%s/combo_num=%d/%s/%s' % (output_dir,i,source_str,target_str)
            else:
                rw_hp_output_dir = '%s/combo_num=%d/%s' % (output_dir,i,target_str)

            pdb_files = glob.glob('%s/**/*.pdb' % rw_hp_output_dir)
            if len(pdb_files) > 0: #restart
                print('removing pdb files in %s' % rw_hp_output_dir)
                remove_files(pdb_files)

            print('BEGINNING RW FOR: %s' % rw_hp_output_dir)

            state_history, conformation_info = run_rw(pdb_path_initial, intrinsic_dim, rw_type, rw_hp_dict, num_total_steps, L, 'full', model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_output_dir, 'rw', save_intrinsic_param, args, early_stop=True)

            if items not in state_history_dict:
                state_history_dict[items] = state_history
            else:
                state_history_dict[items].extend(state_history)

            acceptance_rate = sum(state_history_dict[items])/len(state_history_dict[items])
            if args.early_stop_rw_hp_tuning:
                if acceptance_rate <= upper_bound_acceptance_threshold and acceptance_rate >= lower_bound_acceptance_threshold:
                    state_history_dict = autopopulate_state_history_dict(state_history_dict, grid_search_combinations, items, num_total_steps)
                    return state_history_dict

    else: #new set of grid_searching_combinations being tested 

        #precondition: grid_searching_combinations is sorted in ascending order by sum of all elements in each tuple
        min_max_combination = [grid_search_combinations[0], grid_search_combinations[-1]]
        grid_search_combinations_excluding_min_max = grid_search_combinations[1:-1]

        extrapolated_state_history = []

        #if the acceptance rate of max(grid_search_combinations) >= ub_threshold, we set the acceptance rate of all other combinations to 1 (because decreasing scaling factor only can increase acceptance rate)
        #if the acceptance rate of min(grid_search_combinations) <= lb_threshold, we set the acceptance rate of all other combinations to 0 (because increasing scaling factor only can decrease acceptance rate)
        
        for i in reversed(range(2)): #order doesn't technically matter, but we tend to undershoot scaling_factor, so we start evaluating the max_combination as opposed to the min_combination 
            items = min_max_combination[i]
            rw_hp_dict = populate_rw_hp_dict(rw_type, items)
            print(asterisk_line)
            print('EVALUATING RW HYPERPARAMETERS:')
            print(rw_hp_dict)
            print(asterisk_line)

            if i == 0:      
                combo_num = i 
            else:
                combo_num = len(grid_search_combinations)-1
            
            if is_multimer:
                rw_hp_output_dir = '%s/combo_num=%d/%s/%s' % (output_dir,combo_num,source_str,target_str)
            else:
                rw_hp_output_dir = '%s/combo_num=%d/%s' % (output_dir,combo_num,target_str)

            pdb_files = glob.glob('%s/**/*.pdb' % rw_hp_output_dir)
            if len(pdb_files) > 0: #restart
                print('removing pdb files in %s' % rw_hp_output_dir)
                remove_files(pdb_files)

            print('BEGINNING RW FOR: %s' % rw_hp_output_dir)

            state_history, conformation_info = run_rw(pdb_path_initial, intrinsic_dim, rw_type, rw_hp_dict, num_total_steps, L, 'full', model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_output_dir, 'rw', save_intrinsic_param, args, early_stop=True)

            if items not in state_history_dict:
                state_history_dict[items] = state_history
            else:
                state_history_dict[items].extend(state_history)

            acceptance_rate = sum(state_history_dict[items])/len(state_history_dict[items])
            if args.early_stop_rw_hp_tuning:
                if acceptance_rate <= upper_bound_acceptance_threshold and acceptance_rate >= lower_bound_acceptance_threshold:
                    state_history_dict = autopopulate_state_history_dict(state_history_dict, grid_search_combinations, items, num_total_steps)
                    return state_history_dict
                
            if i == 0: #min_combination
                if acceptance_rate <= lower_bound_acceptance_threshold:
                    state_history_dict = autopopulate_state_history_dict(state_history_dict, grid_search_combinations, None, num_total_steps) #extrapolate all combinations with -1 
                    return state_history_dict
            else: #max_combination
                if acceptance_rate >= upper_bound_acceptance_threshold:
                    state_history_dict = autopopulate_state_history_dict(state_history_dict, grid_search_combinations, None, num_total_steps) #extrapolate all combinations with -1
                    return state_history_dict
                
        #calculate for all other combinations if did not extrapolate 
        for i,items in enumerate(grid_search_combinations_excluding_min_max): 
            rw_hp_dict = populate_rw_hp_dict(rw_type, items, is_multimer)
            print(asterisk_line)
            print('EVALUATING RW HYPERPARAMETERS:')
            print(rw_hp_dict)
            print(asterisk_line)

            if is_multimer:
                rw_hp_output_dir = '%s/combo_num=%d/%s/%s' % (output_dir,i,source_str,target_str)
            else:
                rw_hp_output_dir = '%s/combo_num=%d/%s' % (output_dir,i,target_str)

            pdb_files = glob.glob('%s/**/*.pdb' % rw_hp_output_dir)
            if len(pdb_files) > 0: #restart
                print('removing pdb files in %s' % rw_hp_output_dir)
                remove_files(pdb_files)

            print('BEGINNING RW FOR: %s' % rw_hp_output_dir)

            state_history, conformation_info = run_rw(pdb_path_initial, intrinsic_dim, rw_type, rw_hp_dict, num_total_steps, L, 'full', model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_output_dir, 'rw', save_intrinsic_param, args, early_stop=True)

            if items not in state_history_dict:
                state_history_dict[items] = state_history
            else:
                state_history_dict[items].extend(state_history)

            acceptance_rate = sum(state_history_dict[items])/len(state_history_dict[items])
            if args.early_stop_rw_hp_tuning:
                if acceptance_rate <= upper_bound_acceptance_threshold and acceptance_rate >= lower_bound_acceptance_threshold:
                    state_history_dict = autopopulate_state_history_dict(state_history_dict, grid_search_combinations, items, num_total_steps)
                    return state_history_dict 

    return state_history_dict



def get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data, is_multimer=False):

    if is_multimer:
        f = lambda x: sum(x)
    else:
        f = lambda x: x[0] #sort by eps scaling factor (eps scaling factor is always first element in zipped list) 

    logger.info('GETTING OPTIMAL HP GIVEN CURRENT:')
    logger.info(hp_acceptance_rate_dict)

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)
    
    acceptance_rate_delta_dict = {} 
    for key in sorted(hp_acceptance_rate_dict.keys(),key=f): 
            acceptance_rate = hp_acceptance_rate_dict[key]
            acceptance_rate_delta_dict[key] = abs(acceptance_rate - rw_hp_config_data['rw_tuning_acceptance_threshold'])

    #get hyperparameters whose acceptance rate is closest to acceptance_threshold (and at least as large as the acceptance threshold) 
    optimal_hyperparameters = None
    min_delta = 100
    for key in sorted(acceptance_rate_delta_dict.keys(),key=f):  
        acceptance_rate_delta = acceptance_rate_delta_dict[key]
        acceptance_rate = hp_acceptance_rate_dict[key]
        if acceptance_rate > upper_bound_acceptance_threshold or acceptance_rate < lower_bound_acceptance_threshold:
            continue 
        elif acceptance_rate_delta < min_delta:
            min_delta = acceptance_rate_delta
            optimal_hyperparameters = key

    if optimal_hyperparameters is not None:
        rw_hp_dict = populate_rw_hp_dict(rw_hp_config_data['rw_type'], optimal_hyperparameters, is_multimer)
    else:
        rw_hp_dict = {} 

    return rw_hp_dict


def get_rw_hp_tuning_info(state_history_dict, hp_acceptance_rate_dict, grid_search_combinations):

    for key in state_history_dict: 
        logger.info('FOR RW HYPERPARAMETER COMBINATION:')
        logger.info(key)
        logger.info('STATE HISTORY:')
        logger.info(state_history_dict[key])
        cumm_acceptance_rate = sum(state_history_dict[key])/len(state_history_dict[key])
        logger.info('ACCEPTANCE RATE= %.3f' % cumm_acceptance_rate)
        hp_acceptance_rate_dict[key] = cumm_acceptance_rate

        if cumm_acceptance_rate > upper_bound_acceptance_threshold or cumm_acceptance_rate < lower_bound_acceptance_threshold:
            if key in grid_search_combinations:
                logger.info('REMOVING RW HYPERPARAMETER COMBINATION WITH ACCEPTANCE RATE %.2f' % cumm_acceptance_rate)
                logger.info(key)
                grid_search_combinations.remove(key)

    logger.info('CURRENT GRID SEARCH PARAMETERS:')
    logger.info(grid_search_combinations)

    exit_status = 0 

    if len(grid_search_combinations) == 0:
        logger.info('ALL CURRENT HYPERPARAMETER COMBINATIONS HAVE BEEN ELIMINATED')
        exit_status = 1
    elif len(grid_search_combinations) == 1:
        logger.info('ONLY SINGLE HYPERPARAMETER COMBINATION  EXISTS:')
        exit_status = 1
    elif len(grid_search_combinations) > 1:
        completed = False
        if (i+1) >= args.num_rw_hp_tuning_rounds_total:
            completed = True
        if completed:
            logger.info('ALL HYPERPARAMETER COMBINATIONS HAVE BEEN RUN THE SPECIFIED NUMBER OF ROUNDS (%d)' % args.num_rw_hp_tuning_rounds_total)
            exit_status = 1 

    return hp_acceptance_rate_dict, grid_search_combinations, exit_status


