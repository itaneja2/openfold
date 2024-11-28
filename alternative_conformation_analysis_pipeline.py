from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd 
from pathlib import Path
import pickle 
import io 
import os
import sys
import argparse
import subprocess
import glob 
import shutil 

from eval_pred_conformations import get_clustered_conformations_metrics, compare_clustered_conformations_pca, compare_clustered_conformations_pca_w_ref 

cluster_conformations_path = './cluster_conformations.py' 

def get_uniprot_id(method_str, conformation_info_path):
    if method_str == 'rw':
        start_index = conformation_info_path.find('rw_predictions')+len('rw_predictions')+1
        end_index = conformation_info_path.find('alternative_conformations-verbose')-1
    elif method_str == 'benchmark':
        start_index = conformation_info_path.find('benchmark_predictions')+len('benchmark_predictions')+1
        end_index = conformation_info_path.find('msa_mask')-1
 
    return conformation_info_path[start_index:end_index]


def cluster_conformations_wrapper(script_arguments):
    cmd_to_run = ["python", cluster_conformations_path] + script_arguments
    cmd_to_run_str = s = ' '.join(cmd_to_run)
    subprocess.run(cmd_to_run)


def run_gt_comparison_pipeline(method_str, template_str, testset_dir, conformational_states_df):

    monomer_or_multimer = 'monomer'

    if method_str == 'rw':
        conformation_info_pattern = '%s/rw_predictions/*/*/%s/**/conformation_info.pkl' % (testset_dir, template_str)
    elif method_str == 'benchmark':
        conformation_info_pattern = '%s/benchmark_predictions/*/*/%s/conformation_info.pkl' % (testset_dir, template_str)

    conformation_info_fpaths = glob.glob(conformation_info_pattern, recursive=True)

    if method_str == 'rw':
        conformation_info_files_all_uniprot_id = [] 
        for f in conformation_info_fpaths:
            if 'target=conformation0' in f:
                conformation_info_files_all_uniprot_id.append(f)
    else:
        conformation_info_files_all_uniprot_id = conformation_info_fpaths 

    conformation_info_dir_all_uniprot_id = [] 
    for f in conformation_info_files_all_uniprot_id:
        conformation_info_dir = f[0:f.rindex('/')]
        if method_str == 'rw':
            conformation_info_dir = conformation_info_dir[0:conformation_info_dir.rindex('/')] #grandparent dir for rw 
        conformation_info_dir_all_uniprot_id.append(conformation_info_dir)

    print('**********************************')
    print(conformation_info_dir_all_uniprot_id)
    print(len(conformation_info_dir_all_uniprot_id))
    print('**********************************')

    for f in conformation_info_dir_all_uniprot_id:

        root_dir = f[0:f.rindex(template_str)-1]
        uniprot_id = get_uniprot_id(method_str, f) 

        if method_str == 'rw':
            initial_pred_pattern = '%s/**/initial_pred_unrelaxed.pdb' % root_dir
        else:
            initial_pred_pattern = '%s/**/initial_pred_msa_mask_fraction-0_unrelaxed.pdb' % root_dir

        initial_pred_paths = glob.glob(initial_pred_pattern, recursive=True)

        if len(initial_pred_paths) > 1:
            print("multiple files at:")
            print(initial_pred_paths)
            sys.exit()

        initial_pred_path = initial_pred_paths[0]

        if uniprot_id not in list(conformational_states_df['uniprot_id']):
            print('SKIPPING %s' % uniprot_id)
            continue 
        else:   
            print('RUNNING for %s' % uniprot_id)

        md_starting_structure_fpaths = glob.glob('%s/**/md_starting_structure_info.pkl' % root_dir, recursive=True)

        if len(md_starting_structure_fpaths) == 0:
            continue 

        cluster_representative_conformation_info_fname = md_starting_structure_fpaths[0]
        cluster_representative_conformation_info_dir = cluster_representative_conformation_info_fname[0:cluster_representative_conformation_info_fname.rindex('/')]

        print(cluster_representative_conformation_info_dir)
       
        conformational_states_df_subset = conformational_states_df[conformational_states_df['uniprot_id'] == uniprot_id]
        pdb_id_ref = str(conformational_states_df_subset.iloc[0]['pdb_id_ref'])
        pdb_id_state_i = str(conformational_states_df_subset.iloc[0]['pdb_id_state_i'])

        if 'pdb_id_outside_training_set' in conformational_states_df.columns:
            pdb_id_outside_training_set = str(conformational_states_df_subset.iloc[0]['pdb_id_outside_training_set'])
            if pdb_id_outside_training_set == pdb_id_ref or pdb_id_outside_training_set == 'both':
                pdb_id_outside_training_set_bool = True
        else:
            pdb_id_outside_training_set_bool = False 

        get_clustered_conformations_metrics(uniprot_id, pdb_id_ref, pdb_id_outside_training_set_bool, cluster_representative_conformation_info_dir, initial_pred_path, monomer_or_multimer, method_str, save=True)

        if 'pdb_id_outside_training_set' in conformational_states_df.columns:
            pdb_id_outside_training_set = str(conformational_states_df_subset.iloc[0]['pdb_id_outside_training_set'])
            if pdb_id_outside_training_set == pdb_id_state_i or pdb_id_outside_training_set == 'both':
                pdb_id_outside_training_set_bool = True
        else:
            pdb_id_outside_training_set_bool = False 

        get_clustered_conformations_metrics(uniprot_id, pdb_id_state_i, pdb_id_outside_training_set_bool, cluster_representative_conformation_info_dir, initial_pred_path, monomer_or_multimer, method_str, save=True)


def run_pca_pipeline(template_str, testset_dir, conformational_states_df):

    monomer_or_multimer = 'monomer'
    conformation_info_dir_dict = {}

    for method_str in ['rw', 'benchmark']:

        if method_str == 'rw':
            conformation_info_pattern = '%s/rw_predictions/*/*/%s/**/conformation_info.pkl' % (testset_dir, template_str)
        elif method_str == 'benchmark':
            conformation_info_pattern = '%s/benchmark_predictions/*/*/%s/conformation_info.pkl' % (testset_dir, template_str)

        conformation_info_fpaths = glob.glob(conformation_info_pattern, recursive=True)

        if method_str == 'rw':
            conformation_info_files_all_uniprot_id = [] 
            for f in conformation_info_fpaths:
                if 'target=conformation0' in f:
                    conformation_info_files_all_uniprot_id.append(f)
        else:
            conformation_info_files_all_uniprot_id = conformation_info_fpaths 

        for f in conformation_info_files_all_uniprot_id:
            conformation_info_dir = f[0:f.rindex('/')]
            if method_str == 'rw':
                conformation_info_dir = conformation_info_dir[0:conformation_info_dir.rindex('/')] #grandparent dir for rw 
            uniprot_id = get_uniprot_id(method_str, conformation_info_dir)
            if uniprot_id not in conformation_info_dir_dict:
                conformation_info_dir_dict[uniprot_id] = {}
            conformation_info_dir_dict[uniprot_id][method_str] = conformation_info_dir

    print(conformation_info_dir_dict)    

    for uniprot_id in conformation_info_dir_dict:

        rw_dir = conformation_info_dir_dict[uniprot_id]['rw']
        benchmark_dir = conformation_info_dir_dict[uniprot_id]['benchmark']

        root_dir = rw_dir[0:rw_dir.rindex(template_str)-1]

        rw_initial_pred_pattern = '%s/**/initial_pred_unrelaxed.pdb' % root_dir
        benchmark_initial_pred_pattern = '%s/**/initial_pred_msa_mask_fraction-0_unrelaxed.pdb' % root_dir

        rw_initial_pred_paths = glob.glob(rw_initial_pred_pattern, recursive=True)
        benchmark_initial_pred_paths = glob.glob(benchmark_initial_pred_pattern, recursive=True)

        if len(rw_initial_pred_paths) > 1:
            print("multiple files at:")
            print(rw_initial_pred_paths)
            sys.exit()

        if len(benchmark_initial_pred_paths) > 1:
            print("multiple files at:")
            print(benchmark_initial_pred_paths)
            sys.exit()

        rw_initial_pred_path = rw_initial_pred_paths[0]
        benchmark_initial_pred_path = rw_initial_pred_paths[0]

        if uniprot_id not in list(conformational_states_df['uniprot_id']):
            print('SKIPPING %s' % uniprot_id)
            continue 
        else:   
            print('RUNNING for %s' % uniprot_id)
 
        md_starting_structure_fpaths = glob.glob('%s/**/md_starting_structure_info.pkl' % rw_dir, recursive=True)

        if len(md_starting_structure_fpaths) == 0:
            continue 

        rw_cluster_representative_conformation_info_fname = md_starting_structure_fpaths[0]
        rw_cluster_representative_conformation_info_dir = rw_cluster_representative_conformation_info_fname[0:rw_cluster_representative_conformation_info_fname.rindex('/')]

        benchmark_cluster_representative_conformation_info_fname = glob.glob('%s/**/md_starting_structure_info.pkl' % benchmark_dir, recursive=True)[0]
        benchmark_cluster_representative_conformation_info_dir = benchmark_cluster_representative_conformation_info_fname[0:benchmark_cluster_representative_conformation_info_fname.rindex('/')]

        print(rw_cluster_representative_conformation_info_dir)
        print(benchmark_cluster_representative_conformation_info_dir)
       
        conformational_states_df_subset = conformational_states_df[conformational_states_df['uniprot_id'] == uniprot_id]
        pdb_id_ref = str(conformational_states_df_subset.iloc[0]['pdb_id_ref'])
        pdb_id_state_i = str(conformational_states_df_subset.iloc[0]['pdb_id_state_i'])

        if 'pdb_id_outside_training_set' in conformational_states_df.columns:
            pdb_id_outside_training_set = str(conformational_states_df_subset.iloc[0]['pdb_id_outside_training_set'])
            pdb_id_outside_training_set_enum = pdb_id_outside_training_set
            if pdb_id_outside_training_set_enum not in ['both','none']:
                pdb_id_outside_training_set_enum = 'single'
        else:
            pdb_id_outside_training_set_enum = 'none' 
        
        pca_df = compare_clustered_conformations_pca(uniprot_id, pdb_id_outside_training_set_enum, rw_initial_pred_path, benchmark_initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testset_results/pca_rw_benchmark_comparison/%s/noref' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)

        '''pca_df = compare_clustered_conformations_pca_w_ref(uniprot_id, pdb_id_ref, initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testset_results/pca_rw_benchmark_comparison/%s/ref1' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)

        pca_df = compare_clustered_conformations_pca_w_ref(uniprot_id, pdb_id_state_i, initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testset_results/pca_rw_benchmark_comparison/%s/ref2' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)''' 


testset_dir = './metadynamics_testset_results' #./conformational_states_testset_results
conformational_states_df_path = './metadynamics_dataset/metadynamics_dataset_processed.csv' #'./afsample2_dataset/afsample2_dataset_processed_adjudicated.csv'

conformational_states_df = pd.read_csv(conformational_states_df_path)
conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 
print(conformational_states_df)

template_str = 'template=none'
run_pca_pipeline(template_str, testset_dir, conformational_states_df) 
run_gt_comparison_pipeline('benchmark', template_str, testset_dir, conformational_states_df)
run_gt_comparison_pipeline('rw', template_str, testset_dir, conformational_states_df)
