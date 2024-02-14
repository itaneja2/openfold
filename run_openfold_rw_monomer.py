import argparse
import logging
import math
import numpy as np
import os
import shutil
import json
from collections import Counter
import re
import glob  
import sys
from datetime import date
import itertools
import time 

from openfold.utils.script_utils import load_model_w_intrinsic_param, parse_fasta, run_model_w_intrinsic_dim, prep_output, \
    update_timings, relax_protein

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

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

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args
import pandas as pd 

from random_corr_utils.random_corr_sap import gen_randcorr_sap
from pdb_utils.pdb_utils import align_and_get_rmsd
import rw_helper_functions

finetune_openfold_path = './finetune_openfold.py'

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'

def eval_model(model, config, intrinsic_parameter, feature_processor, feature_dict, processed_feature_dict, tag, output_dir, phase, args):

    model.intrinsic_parameter = nn.Parameter(torch.tensor(intrinsic_parameter, dtype=torch.float).to(args.model_device))

    print('Tag: %s' % tag)
    os.makedirs(output_dir, exist_ok=True)

    out, inference_time = run_model_w_intrinsic_dim(model, processed_feature_dict, tag, output_dir, return_inference_time=True)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    mean_plddt = np.mean(out["plddt"])

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        args.config_preset,
        args.multimer_ri_gap,
        args.subtract_plddt
    )

    output_name = 'temp'
    model_output_dir_temp = '%s/temp' % output_dir
    os.makedirs(model_output_dir_temp, exist_ok=True)

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir_temp, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    disordered_percentage = rw_helper_functions.calc_disordered_percentage(unrelaxed_output_path)
    accept_conformation = rw_helper_functions.accept_criteria(mean_plddt, disordered_percentage, args.mean_plddt_threshold, args.disordered_percentage_threshold)
    shutil.rmtree(model_output_dir_temp)

    if phase == 'initial': #i.e the initial AF prediction from the original MSA
        output_name = tag
        model_output_dir = output_dir
    else:
        if accept_conformation:
            output_name = '%s-A' % tag  
            model_output_dir = '%s/ACCEPTED' % (output_dir)
            os.makedirs(model_output_dir, exist_ok=True)
        else:
            output_name = '%s-R' % tag 
            model_output_dir = '%s/REJECTED' % (output_dir) 
            os.makedirs(model_output_dir, exist_ok=True) 

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))


    logger.info(f"Output written to {unrelaxed_output_path}...")

    return mean_plddt, disordered_percentage, inference_time, accept_conformation, unrelaxed_output_path 


def propose_new_state_vanila_rw(intrinsic_param_curr, intrinsic_dim, epsilon_scaling_factor, cov_type, L=None, sigma=None):
    if cov_type == 'spherical':
        epsilon = np.random.standard_normal(intrinsic_dim)
        epsilon = epsilon/np.linalg.norm(epsilon)
        intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif cov_type == 'diagonal':
        if sigma is None:
            raise ValueError("Sigma is missing. It must be provided if cov_type=diagonal")
        epsilon = np.squeeze(np.random.normal(np.zeros(intrinsic_dim), sigma, (1,intrinsic_dim)))
        intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif 'full' in cov_type:
        if L is None:
            raise ValueError("Cholesky decomposition is missing. It must be provided if cov_type=full")
        x = np.random.standard_normal((intrinsic_dim, 1))
        epsilon = np.dot(L, x).squeeze()
        intrinsic_param_proposed = intrinsic_param_curr + epsilon*epsilon_scaling_factor 
    
    return intrinsic_param_proposed


def propose_new_state_discrete_ou(intrinsic_param_curr, intrinsic_dim, epsilon_scaling_factor, alpha, cov_type, L=None, sigma=None):
    if cov_type == 'spherical':
        epsilon = np.random.standard_normal(intrinsic_dim)
        epsilon = epsilon/np.linalg.norm(epsilon)
        intrinsic_param_proposed = (1-alpha)*intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif cov_type == 'diagonal':
        if sigma is None:
            raise ValueError("Sigma is missing. It must be provided if cov_type=diagonal")
        epsilon = np.squeeze(np.random.normal(np.zeros(intrinsic_dim), sigma, (1,intrinsic_dim)))
        intrinsic_param_proposed = (1-alpha)*intrinsic_param_curr + epsilon*epsilon_scaling_factor
    elif 'full' in cov_type:
        if L is None:
            raise ValueError("Cholesky decomposition is missing. It must be provided if cov_type=full")
        x = np.random.standard_normal((intrinsic_dim, 1))
        epsilon = np.dot(L, x).squeeze()
        intrinsic_param_proposed = (1-alpha)*intrinsic_param_curr + epsilon*epsilon_scaling_factor 
    
    return intrinsic_param_proposed


def propose_new_state_rw_w_momentum(intrinsic_param_curr, velocity_param_curr, intrinsic_dim, epsilon_scaling_factor, gamma, cov_type, L=None,sigma=None):
    if cov_type == 'spherical':
        epsilon = np.random.standard_normal(intrinsic_dim)
        epsilon = epsilon/np.linalg.norm(epsilon)
        v = gamma*velocity_param_curr + epsilon*epsilon_scaling_factor
        intrinsic_param_proposed = intrinsic_param_curr + v
    elif cov_type == 'diagonal':
        if sigma is None:
            raise ValueError("Sigma is missing. It must be provided if cov_type=diagonal")
        epsilon = np.squeeze(np.random.normal(np.zeros(intrinsic_dim), sigma, (1,intrinsic_dim)))
        v = gamma*velocity_param_curr + epsilon*epsilon_scaling_factor
        intrinsic_param_proposed = intrinsic_param_curr + v
    elif 'full' in cov_type:
        if L is None:
            raise ValueError("Cholesky decomposition is missing. It must be provided if cov_type=full")
        x = np.random.standard_normal((intrinsic_dim, 1))
        epsilon = np.dot(L, x).squeeze()
        v = gamma*velocity_param_curr + epsilon*epsilon_scaling_factor
        intrinsic_param_proposed = intrinsic_param_curr + v 
    
    return intrinsic_param_proposed


def run_rw(
    pdb_path_initial, 
    intrinsic_dim, 
    rw_type, 
    rw_hp_dict, 
    num_total_steps, 
    L, 
    cov_type, 
    model, 
    config, 
    feature_processor, 
    feature_dict, 
    processed_feature_dict, 
    output_dir, 
    phase, 
    save_intrinsic_param, 
    args,
    early_stop=False,
):

    base_tag = '%s_train-%s_rw-%s' % (args.module_config, args.train_hp_config, args.rw_hp_config)
    base_tag = base_tag.replace('=','-')

    intrinsic_param_prev = np.zeros(intrinsic_dim)
    if rw_type == 'rw_w_momentum':
        velocity_param_prev = np.zeros(intrinsic_dim)
    num_rejected_steps = 0
    num_accepted_steps = 0  
    iter_n = 0 
    curr_step_iter_n = 0
    curr_step_aggregate = 0 
    state_history = [] 
    conformation_info = [] 

    while (num_rejected_steps+num_accepted_steps) < num_total_steps:

        if rw_type == 'vanila':
            intrinsic_param_proposed = propose_new_state_vanila_rw(intrinsic_param_prev, intrinsic_dim, rw_hp_dict['epsilon_scaling_factor'], cov_type, L)
        elif rw_type == 'discrete_ou':
            intrinsic_param_proposed = propose_new_state_discrete_ou(intrinsic_param_prev, intrinsic_dim, rw_hp_dict['epsilon_scaling_factor'], rw_hp_dict['alpha'], cov_type, L)
        elif rw_type == 'rw_w_momentum':
            intrinsic_param_proposed = propose_new_state_rw_w_momentum(intrinsic_param_prev, velocity_param_prev, intrinsic_dim, rw_hp_dict['epsilon_scaling_factor'], rw_hp_dict['gamma'], cov_type, L)

        curr_tag = '%s_iter%d_step-iter%d_step-agg%d' % (base_tag, iter_n, curr_step_iter_n, curr_step_aggregate) 
        mean_plddt, disordered_percentage, inference_time, accept_conformation, pdb_path_rw = eval_model(model, config, intrinsic_param_proposed, feature_processor, feature_dict, processed_feature_dict, curr_tag, output_dir, phase, args)    
        state_history.append(accept_conformation)
        print('pLDDT: %.3f, disordered percentage: %.3f, step: %d' % (mean_plddt, disordered_percentage, curr_step_aggregate)) 

        if accept_conformation:
            print('STEP %d: ACCEPTED' % curr_step_aggregate)
            intrinsic_param_prev = intrinsic_param_proposed
            curr_step_iter_n += 1
            num_accepted_steps += 1
            rmsd = align_and_get_rmsd(pdb_path_initial, pdb_path_rw) #important that we align prior to training
            if save_intrinsic_param:
                conformation_info.append((pdb_path_rw, rmsd, mean_plddt, disordered_percentage, inference_time, intrinsic_param_proposed, rw_hp_dict['epsilon_scaling_factor']))
            else:
                conformation_info.append((pdb_path_rw, rmsd, mean_plddt, disordered_percentage, inference_time, None, None))
        else:
            print('STEP %d: REJECTED' % curr_step_aggregate)
            if early_stop:
                return state_history, conformation_info
            else:
                intrinsic_param_prev = np.zeros(intrinsic_dim)
                if rw_type == 'rw_w_momentum':
                    velocity_param_prev = np.zeros(intrinsic_dim)
                iter_n += 1
                curr_step_iter_n = 0 
                num_rejected_steps += 1 

        curr_step_aggregate += 1

    return state_history, conformation_info


def get_new_scaling_factor_candidates(hp_acceptance_rate_dict, rw_hp_config_data):

    print('GETTING NEW SCALING FACTOR CANDIDATES GIVEN CURRENT:')
    print(hp_acceptance_rate_dict)

    upper_bound_scaling_factor = None
    lower_bound_scaling_factor = None 

    #find largest eps_scaling factor greater than target threshold 
    for key in sorted(hp_acceptance_rate_dict, key=lambda x: x[0], reverse=True): #sort by eps scaling factor (eps scaling factor is always first element in zipped list) 
        acceptance_rate = hp_acceptance_rate_dict[key]
        if acceptance_rate > rw_hp_config_data['rw_tuning_acceptance_threshold']:
            lower_bound_scaling_factor = key[0]
            break

    #find smallest eps_scaling factor less than target threshold 
    for key in sorted(hp_acceptance_rate_dict, key=lambda x: x[0]): #sort by eps scaling factor in reverse order (eps scaling factor is always first element in zipped list) 
        acceptance_rate = hp_acceptance_rate_dict[key]
        if acceptance_rate < rw_hp_config_data['rw_tuning_acceptance_threshold']:
            upper_bound_scaling_factor = key[0]
            break

    curr_scaling_factor_candidates = [key[0] for key in hp_acceptance_rate_dict.keys()] #extract eps scaling factor from hp_acceptance_rate_dict

    if lower_bound_scaling_factor is None:
        lower_bound_scaling_factor = min(curr_scaling_factor_candidates)/2    
    if upper_bound_scaling_factor is None:
        upper_bound_scaling_factor = max(curr_scaling_factor_candidates)*2

    new_scaling_factor_candidates = list(np.linspace(lower_bound_scaling_factor, upper_bound_scaling_factor, 5))
    new_scaling_factor_candidates = new_scaling_factor_candidates[1:-1] #exclude first and last element
    return new_scaling_factor_candidates


def get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data):

    print('GETTING OPTIMAL HP GIVEN CURRENT:')
    print(hp_acceptance_rate_dict)

    upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
    lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)
    
    acceptance_rate_delta_dict = {} 
    for key in sorted(hp_acceptance_rate_dict.keys(),key=lambda x: x[0]): #sort by eps scaling factor (eps scaling factor is always first element in zipped list) 
            acceptance_rate = hp_acceptance_rate_dict[key]
            acceptance_rate_delta_dict[key] = abs(acceptance_rate - rw_hp_config_data['rw_tuning_acceptance_threshold'])

    #get hyperparameters whose acceptance rate is closest to acceptance_threshold (and at least as large as the acceptance threshold) 
    optimal_hyperparameters = None
    min_delta = 100
    for key in sorted(acceptance_rate_delta_dict.keys(),key=lambda x: x[0]):
        acceptance_rate_delta = acceptance_rate_delta_dict[key]
        acceptance_rate = hp_acceptance_rate_dict[key]
        if acceptance_rate > upper_bound_acceptance_threshold or acceptance_rate < lower_bound_acceptance_threshold:
            continue 
        elif acceptance_rate_delta < min_delta:
            min_delta = acceptance_rate_delta
            optimal_hyperparameters = key

    if optimal_hyperparameters is not None:
        rw_hp_dict = populate_rw_hp_dict(rw_hp_config_data['rw_type'], optimal_hyperparameters)
    else:
        rw_hp_dict = {} 

    return rw_hp_dict

             
def construct_grid_search_combinations(rw_type, scaling_factor_candidates, alpha_candidates, gamma_candidates):

    if rw_type == 'vanila':
        grid_search_combinations = list(itertools.product(scaling_factor_candidates))
    elif rw_type == 'discrete_ou':
        grid_search_combinations = list(itertools.product(scaling_factor_candidates,alpha_candidates)) 
    elif rw_type == 'rw_w_momentum':
        grid_search_combinations = list(itertools.product(scaling_factor_candidates,gamma_candidates)) 

    grid_search_combinations = sorted(grid_search_combinations, key=lambda x: x[0])

    return grid_search_combinations

def populate_rw_hp_dict(rw_type, items):

    rw_hp_dict = {} 

    if rw_type == 'vanila':
        rw_hp_dict['epsilon_scaling_factor'] = items[0]
    elif rw_type == 'discrete_ou':
        rw_hp_dict['epsilon_scaling_factor'] = items[0]
        rw_hp_dict['alpha'] = items[1]
    elif rw_type == 'rw_w_momentum':
        rw_hp_dict['epsilon_scaling_factor'] = items[0]
        rw_hp_dict['gamma'] = items[1]

    return rw_hp_dict



def main(args):
    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)

    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )
 
    output_dir = '%s/%s/%s/train-%s/rw-%s' % (args.output_dir_base, 'rw_v5', args.module_config, args.train_hp_config, args.rw_hp_config)
    l1_output_dir = '%s/%s/%s' % (args.output_dir_base, 'rw_v5', args.module_config)
    l2_output_dir = '%s/%s/%s/train-%s' % (args.output_dir_base, 'rw_v5', args.module_config, args.train_hp_config)
    
    output_dir = os.path.abspath(output_dir)
    l1_output_dir = os.path.abspath(l1_output_dir)
    l2_output_dir = os.path.abspath(l2_output_dir) 

    print('Output Directory: %s' % output_dir)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    os.makedirs(output_dir, exist_ok=True)
    alignment_dir = args.alignment_dir
    file_id = os.listdir(alignment_dir)
    if len(file_id) > 1:
        raise ValueError("should only be a single directory under %s" % alignment_dir)
    else:
        file_id = file_id[0] #e.g 1xyz_A
        file_id_wo_chain = file_id.split('_')[0]
    alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
    print("alignment directory with file_id: %s" % alignment_dir_w_file_id)

    if args.fasta_file is None:
        pattern = "%s/*.fasta" % alignment_dir_w_file_id
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            fasta_file = files[0]
        else: 
            raise FileNotFoundError("Multiple .fasta files found in alignment_dir -- should only be one")
    else:
        fasta_file = args.fasta_file

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seq = parse_fasta(fasta_data)

    with open('./rw_monomer_config.json') as f:
        rw_config_data = json.load(f)

    module_config_data = rw_config_data['SAID'][args.module_config]
    rw_hp_config_data = rw_config_data['hyperparameter']['rw'][args.rw_hp_config]
    train_hp_config_data = rw_config_data['hyperparameter']['train'][args.train_hp_config]
    intrinsic_dim = module_config_data['intrinsic_dim']
 
    pattern = "%s/features.pkl" % alignment_dir_w_file_id
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        features_output_path = files[0]
        print('features.pkl path: %s' % features_output_path)
    else:
        features_output_path = ''

    if os.path.isfile(features_output_path):
        feature_dict = np.load(features_output_path, allow_pickle=True) #this is used for all predictions, so this assumes you are predicting a single sequence 
    else:
        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=4,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        feature_dict = data_processor.process_fasta(
            fasta_path=fasta_file, alignment_dir=alignment_dir_w_file_id
        )
        features_output_path = os.path.join(alignment_dir_w_file_id, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)
        print('SAVED %s' % features_output_path)

    initial_pred_output_dir = '%s/initial_pred' %  l1_output_dir
        
    intrinsic_param_zero = np.zeros(intrinsic_dim)
    model = load_model_w_intrinsic_param(config, module_config_data, args.model_device, args.openfold_checkpoint_path, args.jax_param_path, intrinsic_param_zero)

    #for m_name, module in dict(model.named_modules()).items():
    #    print(m_name)
    #    print('****')
    #    for c_name, layer in dict(module.named_children()).items():
    #        print(c_name)
    

    bootstrap_output_dir = '%s/bootstrap' % l1_output_dir
    bootstrap_training_conformations_dir = '%s/bootstrap_training_conformations' % l2_output_dir

    if args.skip_bootstrap_phase:
        conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
        pdb_path_initial = '%s/initial_pred_unrelaxed.pdb' % initial_pred_output_dir   
        seed_fname = '%s/seed.txt' % bootstrap_output_dir
        if os.path.exists(conformation_info_fname) and os.path.exists(pdb_path_initial) and os.path.exists(seed_fname):
            with open(conformation_info_fname, 'rb') as f:
                conformation_info = pickle.load(f)
            random_seed = int(np.loadtxt(seed_fname))
            np.random.seed(random_seed)
            torch.manual_seed(random_seed + 1)
            print('SKIPPING BOOTSTRAP PHASE')
        else:
            args.skip_bootstrap_phase = False

    #process features after updating seed 
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )
    processed_feature_dict = {
        k:torch.as_tensor(v, device=args.model_device)
        for k,v in processed_feature_dict.items()
    } 


    ############################################

    if not(args.skip_bootstrap_phase):       
   
        t0 = time.perf_counter()
   
        mean_plddt_initial, disordered_percentage_initial, _, _, pdb_path_initial = eval_model(model, config, intrinsic_param_zero, feature_processor, feature_dict, processed_feature_dict, 'initial_pred', initial_pred_output_dir, 'initial', args)    
        print('pLDDT: %.3f, disordered percentage: %.3f, INITIAL' % (mean_plddt_initial, disordered_percentage_initial)) 
 
        scaling_factor_candidate = 10.  
        all_scaling_factor_candidates = [scaling_factor_candidate] #keeps track of scaling_factor_candidates used 
        scaling_factor_bootstrap = None       

        upper_bound_acceptance_threshold = round(rw_hp_config_data['bootstrap_tuning_acceptance_threshold']+rw_hp_config_data['bootstrap_tuning_threshold_ub_tolerance'],2)
        lower_bound_acceptance_threshold = round(rw_hp_config_data['bootstrap_tuning_acceptance_threshold']-rw_hp_config_data['bootstrap_tuning_threshold_lb_tolerance'],2)

        print(asterisk_line) 
        print('BOOTSTRAP TUNING PHASE:')

        base_tag = '%s_train-%s_rw-%s' % (args.module_config, args.train_hp_config, args.rw_hp_config)
        base_tag = base_tag.replace('=','-')

        ##run a pseudo binary search to determine scaling_factor_bootstrap
        while scaling_factor_bootstrap is None:
            print(asterisk_line)
            print('TESTING SCALING FACTOR: %.3f' % scaling_factor_candidate)
            state_history = []  
            for i in range(0,args.num_bootstrap_hp_tuning_steps):  
                warmup_output_dir = '%s/warmup/scaling_factor=%s' % (l1_output_dir, rw_helper_functions.remove_trailing_zeros(scaling_factor_candidate))
                intrinsic_param_proposed = propose_new_state_vanila_rw(intrinsic_param_zero, intrinsic_dim, scaling_factor_candidate, 'spherical')

                curr_tag = '%s_step%d' % (base_tag,i)         
                mean_plddt, disordered_percentage, _, accept_conformation, _ = eval_model(model, config, intrinsic_param_proposed, feature_processor, feature_dict, processed_feature_dict, curr_tag, warmup_output_dir, 'bootstrap_warmup', args)    
                state_history.append(accept_conformation)

                print('pLDDT: %.3f, disordered percentage: %.3f, step: %d' % (mean_plddt, disordered_percentage, i)) 

                if accept_conformation:
                    print('STEP %d: ACCEPTED' % i)
                else:
                    print('STEP %d: REJECTED' % i)

                if i == 0 and not(accept_conformation):
                    print('EXITING EARLY FOR %d' % scaling_factor_candidate)
                    break 

            acceptance_rate = sum(state_history)/len(state_history)
            if acceptance_rate >= lower_bound_acceptance_threshold and acceptance_rate <= upper_bound_acceptance_threshold:
                scaling_factor_bootstrap = scaling_factor_candidate 
            else:
                if acceptance_rate > upper_bound_acceptance_threshold:
                    proposed_scaling_factor_candidate = scaling_factor_candidate*2
                elif acceptance_rate < lower_bound_acceptance_threshold:
                    proposed_scaling_factor_candidate = scaling_factor_candidate/2
                
                if proposed_scaling_factor_candidate not in all_scaling_factor_candidates:
                    all_scaling_factor_candidates.append(proposed_scaling_factor_candidate)
                    scaling_factor_candidate = proposed_scaling_factor_candidate
                else:
                    #we are proposing to use a previously encountered scaling_factor_candidate (i.e 10 --> 20 --> 10, or 10 --> 5 --> 10)
                    if acceptance_rate > upper_bound_acceptance_threshold:
                        scaling_factor_bootstrap = scaling_factor_candidate #to ensure scaling_factor_bootstrap remains higher than lower_bound_acceptance_threshold 
                    elif acceptance_rate < lower_bound_acceptance_threshold:
                        scaling_factor_bootstrap = scaling_factor_candidate/2 #to ensure scaling_factor_bootstrap remains higher than lower_bound_acceptance_threshold
                    

        print(asterisk_line)
        print('BOOTSTRAP PHASE:')
        print('scaling factor to be used for bootstrapping: %s' % rw_helper_functions.remove_trailing_zeros(scaling_factor_bootstrap))
        print(asterisk_line)

        rw_hp_dict = {}
        rw_hp_dict['epsilon_scaling_factor'] = scaling_factor_bootstrap 
        state_history, conformation_info = run_rw(pdb_path_initial, intrinsic_dim, 'vanila', rw_hp_dict, args.num_bootstrap_steps, None, 'spherical', model, config, feature_processor, feature_dict, processed_feature_dict, bootstrap_output_dir, 'bootstrap', args, False)
 
        bootstrap_acceptance_rate = sum(state_history)/len(state_history)
        print('bootstrap acceptance rate: %.3f' % bootstrap_acceptance_rate)

        conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
        conformation_info = sorted(conformation_info, key=lambda x: x[0], reverse=True)
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info, f)
        print(conformation_info)

        run_time = time.perf_counter() - t0
        timing_dict = {'bootstrap': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'bootstrap')

        seed_fname = '%s/seed.txt' % bootstrap_output_dir
        np.savetxt(seed_fname, [random_seed], fmt='%d')

    bootstrap_conformations = {}
    iter_num_list = [] 
    for i in range(0,len(conformation_info)):
        f = conformation_info[i][0]
        rmsd = conformation_info[i][1]
        match = re.search(r'_iter(\d+)', f)
        iter_num = int(match.group(1))
        match = re.search(r'step-iter(\d+)', f) 
        step_num = int(match.group(1))
        if iter_num not in bootstrap_conformations:
            bootstrap_conformations[iter_num] = [(step_num,rmsd,f)]
        else:
            bootstrap_conformations[iter_num].append((step_num,rmsd,f)) 
        iter_num_list.append(iter_num)

    del bootstrap_conformations[max(iter_num_list)] #delete key corresponding to max(iter_num_list) because this iteration did not 'finish' 

    for key in bootstrap_conformations:
        bootstrap_conformations[key] = sorted(bootstrap_conformations[key], key=lambda x:x[0], reverse=True) #sort by step_num in reverse order 

    bootstrap_candidate_conformations = []
    for key in bootstrap_conformations:
        bootstrap_candidate_conformations.append(bootstrap_conformations[key][0]) #get last conformation in each iteration (i.e last conformation prior to rejection)
        
    bootstrap_candidate_conformations = sorted(bootstrap_candidate_conformations, key=lambda x:x[1], reverse=True) #sort by rmsd in reverse order 

    print('BOOTSTRAP CONFORMATIONS ALL:')
    print(bootstrap_conformations)    

    print('BOOTSTRAP CANDIDATE CONFORMATIONS:')
    print(bootstrap_candidate_conformations)
    
    num_conformations_to_optimize = len(bootstrap_candidate_conformations)

    #optimize initial to bootstrap conformations#

    #####################################################
    print(asterisk_line)

    if not(args.skip_gd_phase):

        t0 = time.perf_counter()

        for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

            if iter_num < args.num_training_conformations:

                print('ON CONFORMATION %d/%d:' % (iter_num+1,min(args.num_training_conformations,num_conformations_to_optimize)))
                print(conformation_info_i)
                bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)
                os.makedirs(bootstrap_training_conformations_dir_conformation_i, exist_ok=True)
                rw_helper_functions.rw_helper_functions.remove_files_in_dir(bootstrap_training_conformations_dir_conformation_i) #this directory should contain a single conformation so we can run train_openfold on it 
                curr_pdb_path = conformation_info_i[2]
                curr_pdb_fname = '%s.pdb' % file_id_wo_chain #during training, expected filename is without chain_id (i.e 1xyz)
                dst_path = '%s/%s' % (bootstrap_training_conformations_dir_conformation_i,curr_pdb_fname)
                shutil.copy(curr_pdb_path,dst_path)

                target_str = 'target=conformation%d' % iter_num
                fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)
 
                arg1 = '--train_data_dir=%s' % bootstrap_training_conformations_dir_conformation_i
                arg2 = '--train_alignment_dir=%s' % args.alignment_dir
                arg3 = '--fine_tuning_save_dir=%s' % fine_tuning_save_dir
                arg4 = '--template_mmcif_dir=%s' % args.template_mmcif_dir
                arg5 = '--max_template_date=%s' % date.today().strftime("%Y-%m-%d")
                arg6 = '--precision=bf16'
                arg7 = '--gpus=1'
                arg8 = '--openfold_checkpoint_path=%s' % args.openfold_checkpoint_path 
                arg9 = '--resume_model_weights_only=True'
                arg10 = '--config_preset=custom_finetuning-SAID-all'
                arg11 = '--module_config=%s' % args.module_config
                arg12 = '--hp_config=%s' % args.train_hp_config
                arg13 = '--save_structure_output'
         
                if args.save_training_conformations:       
                    script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13]
                else:
                    script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12]
 
                cmd_to_run = ["python", finetune_openfold_path] + script_arguments
                cmd_to_run_str = s = ' '.join(cmd_to_run)
                print(asterisk_line)
                print("RUNNING GRADIENT DESCENT WRT TO: %s" % curr_pdb_fname)
                print(cmd_to_run_str)
                subprocess.run(cmd_to_run)

        run_time = time.perf_counter() - t0
        timing_dict = {'gradient_descent': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'gradient_descent')
        

    #################################################
 

    print('generating random_corr')
    if rw_hp_config_data['cov_type'] == 'full': 
        random_corr = gen_randcorr_sap.randcorr(intrinsic_dim)
    
    rw_hp_dict = {} #aim is to populate this 

    print(asterisk_line)
    rw_hp_parent_dir = '%s/rw_hp_tuning' % output_dir
    print('RW HYPERPARAMETER TUNING PHASE')
    hp_acceptance_rate_dict = {}  
    hp_acceptance_rate_fname = '%s/hp_acceptance_rate_info.pkl' % rw_hp_parent_dir

    skip_auto_calc = False
    if os.path.exists(hp_acceptance_rate_fname):
        print('LOADING %s' % hp_acceptance_rate_fname)
        with open(hp_acceptance_rate_fname, 'rb') as f:
            hp_acceptance_rate_dict = pickle.load(f)
        print(hp_acceptance_rate_dict)
        rw_hp_dict = get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data)
        if rw_hp_dict != {}:
            skip_auto_calc = True

    if not(skip_auto_calc):

        print('TUNING HYPERPARAMETERS VIA GRID SEARCH')

        t0 = time.perf_counter()

        upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
        lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

        scaling_factor_candidates = train_hp_config_data['initial_scaling_factor_candidate']
        scaling_factor_candidates = list(map(float, scaling_factor_candidates))  
        alpha_candidates = [10e-4,10e-3,10e-2]
        gamma_candidates = [.9,.99,.999]
        
        while rw_hp_dict == {}:

            grid_search_combinations = construct_grid_search_combinations(rw_hp_config_data['rw_type'], scaling_factor_candidates, alpha_candidates, gamma_candidates)
            print('INITIAL GRID SEARCH PARAMETERS:')
            print(grid_search_combinations)

            state_history_dict = {} 

            for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

                print('ON CONFORMATION %d/%d:' % (iter_num+1,min(args.num_training_conformations,num_conformations_to_optimize)))
                print(conformation_info_i)
                bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)

                pdb_files = glob.glob('%s/*.pdb' % bootstrap_training_conformations_dir_conformation_i)
                if len(pdb_files) == 0: #files only exist in this directory if we trained with respect to this target structure
                    continue 

                target_str = 'target=conformation%d' % iter_num
                fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)
                latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)

                model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
                sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir)
        
                if 'full' in rw_hp_config_data['cov_type']:
                    random_cov = rw_helper_functions.get_random_cov(sigma, random_corr)
                    print('calculating cholesky')
                    L = np.linalg.cholesky(random_cov)
                else:
                    L = None 

                state_history_dict = rw_helper_functions.run_grid_search(grid_search_combinations, state_history_dict, target_str, pdb_path_initial, intrinsic_dim, rw_hp_config_data['rw_type'], args.num_rw_hp_tuning_steps_per_round, L, rw_hp_config_data['cov_type'], model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_config_data, rw_hp_parent_dir, 'rw', args, False)
  
                for key in state_history_dict: 
                    print('FOR RW HYPERPARAMETER COMBINATION:')
                    print(key)
                    print('STATE HISTORY:')
                    print(state_history_dict[key])
                    cumm_acceptance_rate = sum(state_history_dict[key])/len(state_history_dict[key])
                    print('ACCEPTANCE RATE= %.3f' % cumm_acceptance_rate)
                    hp_acceptance_rate_dict[key] = cumm_acceptance_rate

                    if cumm_acceptance_rate > upper_bound_acceptance_threshold or cumm_acceptance_rate < lower_bound_acceptance_threshold:
                        if key in grid_search_combinations:
                            print('REMOVING RW HYPERPARAMETER COMBINATION WITH ACCEPTANCE RATE %.2f' % cumm_acceptance_rate)
                            print(key)
                            grid_search_combinations.remove(key)

                print('CURRENT GRID SEARCH PARAMETERS:')
                print(grid_search_combinations)

                if len(grid_search_combinations) == 0:
                    print('ALL CURRENT HYPERPARAMETER COMBINATIONS HAVE BEEN ELIMINATED')
                    break 
                elif len(grid_search_combinations) == 1:
                    print('ONLY SINGLE HYPERPARAMETER COMBINATION  EXISTS:')
                    print(grid_search_combinations)
                    break
                elif len(grid_search_combinations) > 1:
                    completed = False
                    if (i+1) >= args.num_rw_hp_tuning_rounds_total:
                        completed = True
                    if completed:
                        print('ALL HYPERPARAMETER COMBINATIONS HAVE BEEN RUN THE SPECIFIED NUMBER OF ROUNDS (%d)' % args.num_rw_hp_tuning_rounds_total)
                        print(grid_search_combinations)
                        break 
                    
            rw_hp_dict = get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data)
            if rw_hp_dict == {}:
                print('NO SCALING FACTOR CANDIDATES FOUND THAT MATCHED ACCEPTANCE CRITERIA')
                scaling_factor_candidates = get_new_scaling_factor_candidates(hp_acceptance_rate_dict, rw_hp_config_data)
                print('HYPERPARAMETER TUNING WITH NEW SCALING FACTOR CANDIDATES')
                print(scaling_factor_candidates)
            else:
                hp_acceptance_rate_fname = '%s/hp_acceptance_rate_info.pkl' % rw_hp_parent_dir
                with open(hp_acceptance_rate_fname, 'wb') as f:
                    pickle.dump(hp_acceptance_rate_dict, f)
                print(hp_acceptance_rate_dict)

        run_time = time.perf_counter() - t0
        timing_dict = {'hp_tuning': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'hp_tuning')
     
                

    #################################################

    print(asterisk_line)
    print('RW PHASE')
    print('HYPERPARAMETERS BEING USED:')
    print(rw_hp_dict)

    for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

        t0 = time.perf_counter()

        conformation_info_dict = {} #maps bootstrap_key to (pdb_path,plddt,disordered_percentage,rmsd)

        print('ON CONFORMATION %d/%d:' % (iter_num+1,min(args.num_training_conformations,num_conformations_to_optimize)))
        print(conformation_info_i)
        bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)

        pdb_files = glob.glob('%s/*.pdb' % bootstrap_training_conformations_dir_conformation_i)
        if len(pdb_files) == 0: #files only exist in this directory if we trained with respect to this target structure
            continue 

        target_str = 'target=conformation%d' % iter_num
        fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)
        latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)

        model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
        sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir)

        if 'full' in rw_hp_config_data['cov_type']:
            random_cov = rw_helper_functions.get_random_cov(sigma, random_corr)
            print('calculating cholesky')
            L = np.linalg.cholesky(random_cov)
        else:
            L = None 
 
        rw_output_dir = '%s/rw/%s' % (output_dir,target_str)

        pdb_files = glob.glob('%s/**/*.pdb' % rw_output_dir)
        if len(pdb_files) >= args.num_rw_steps:
            if args.overwrite_pred:
                print('removing pdb files in %s' % rw_output_dir)
                rw_helper_functions.remove_files(pdb_files)
            else:
                print('SKIPPING RW FOR: %s --%d files already exist--' % (rw_output_dir, len(pdb_files)))
                continue 
        elif len(pdb_files) > 0: #incomplete job
            print('removing pdb files in %s' % rw_output_dir)
            rw_helper_functions.remove_files(pdb_files)

        print('BEGINNING RW FOR: %s' % rw_output_dir)

        state_history, conformation_info = run_rw(pdb_path_initial, intrinsic_dim, rw_hp_config_data['rw_type'], rw_hp_dict, args.num_rw_steps, L, rw_hp_config_data['cov_type'], model, config, feature_processor, feature_dict, processed_feature_dict, rw_output_dir, 'rw', args, early_stop=True)

        conformation_info_dict[iter_num] = conformation_info

        acceptance_rate = sum(state_history)/len(state_history)
        print('ACCEPTANCE RATE: %.3f' % acceptance_rate)

        conformation_info_output_dir = rw_output_dir
        conformation_info_fname = '%s/conformation_info.pkl' % conformation_info_output_dir
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info_dict, f)

        print(asterisk_line)
        print(conformation_info_dict[iter_num])
        print(asterisk_line)
        rmsd_all = np.array([conformation_info[i][1] for i in range(0,len(conformation_info))])
        max_rmsd = np.max(rmsd_all)
        print('MAX RMSD: %.3f' % max_rmsd)

        inference_key = 'inference_%d' % i
        run_time = time.perf_counter() - t0
        timing_dict = {inference_key: run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, inference_key)



           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--alignment_dir", type=str, required=True,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--conformation_dir", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )

    parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--num_bootstrap_steps", type=int, default=50
    )
    parser.add_argument(
        "--num_bootstrap_hp_tuning_steps", type=int, default=10
    )

    parser.add_argument(
        "--num_rw_steps", type=int, default=100
    )
    parser.add_argument(
        "--num_rw_hp_tuning_steps_per_round", type=int, default=10
    )
    parser.add_argument(
        "--num_rw_hp_tuning_rounds_total", type=int, default=2
    )
    parser.add_argument(
        "--early_stop_rw_hp_tuning", action="store_true", default=False,
    )
    parser.add_argument(
        "--num_training_conformations", type=int, default=5
    )
    parser.add_argument(
        "--save_training_conformations", action="store_true", default=False
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=1,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument(
        "--module_config", type=str, default='model_config_0',
        help=(
            "module_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--rw_hp_config", type=str, default='hp_config_0',
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--train_hp_config", type=str, default='hp_config_2',
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--skip_bootstrap_phase", action="store_true", default=False
    )
    parser.add_argument(
        "--skip_gd_phase", action="store_true", default=False
    )
    parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )
    parser.add_argument(
        "--mean_plddt_threshold", type=int, default=70
    )
    parser.add_argument(
        "--disordered_percentage_threshold", type=int, default=80
    )






    add_data_args(parser)
    args = parser.parse_args()

    if(args.jax_param_path is None and args.openfold_checkpoint_path is None):
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)

