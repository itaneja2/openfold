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



def run_gt_comparison_pipeline(method_str, template_str, conformational_states_df):

    monomer_or_multimer = 'monomer'

    if method_str == 'rw':
        conformation_info_pattern = './conformational_states_testing_data_af_weights/rw_predictions/*/*/%s/**/conformation_info.pkl' % template_str
    elif method_str == 'benchmark':
        conformation_info_pattern = './conformational_states_testing_data_af_weights/benchmark_predictions/*/*/%s/conformation_info.pkl' % template_str

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
        initial_pred_pattern = '%s/**/initial_pred*.pdb' % root_dir
        initial_pred_path = glob.glob(initial_pred_pattern, recursive=True)[0]

        if uniprot_id not in list(conformational_states_df['uniprot_id']):
            print('SKIPPING %s' % uniprot_id)
            continue 
        else:   
            print('RUNNING for %s' % uniprot_id)

        #arg1 = '--conformation_info_dir=%s' % f
        #arg2 = '--skip_relaxation'
        #script_arguments = [arg1]
        #cluster_conformations_wrapper(script_arguments)


        md_starting_structure_fpaths = glob.glob('%s/**/md_starting_structure_info.pkl' % rw_dir, recursive=True)

        if len(md_starting_structure_fpaths) == 0:
            continue 

        cluster_representative_conformation_info_fname = md_starting_structure_fpaths[0]
        cluster_representative_conformation_info_dir = cluster_representative_conformation_info_fname[0:cluster_representative_conformation_info_fname.rindex('/')]

        print(cluster_representative_conformation_info_dir)
       
        conformational_states_df_subset = conformational_states_df[conformational_states_df['uniprot_id'] == uniprot_id]
        pdb_id_ref = str(conformational_states_df_subset.iloc[0]['pdb_id_ref'])
        pdb_id_state_i = str(conformational_states_df_subset.iloc[0]['pdb_id_state_i'])

        get_clustered_conformations_metrics(uniprot_id, pdb_id_ref, cluster_representative_conformation_info_dir, initial_pred_path, monomer_or_multimer, method_str, save=True)
        get_clustered_conformations_metrics(uniprot_id, pdb_id_state_i, cluster_representative_conformation_info_dir, initial_pred_path, monomer_or_multimer, method_str, save=True)


def run_pca_pipeline(template_str, conformational_states_df):

    monomer_or_multimer = 'monomer'
    conformation_info_dir_dict = {}

    for method_str in ['rw', 'benchmark']:

        if method_str == 'rw':
            conformation_info_pattern = './conformational_states_testing_data_af_weights/rw_predictions/*/*/%s/**/conformation_info.pkl' % template_str
        elif method_str == 'benchmark':
            conformation_info_pattern = './conformational_states_testing_data_af_weights/benchmark_predictions/*/*/%s/conformation_info.pkl' % template_str

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
        initial_pred_pattern = '%s/**/initial_pred*.pdb' % root_dir
        rw_initial_pred_path = glob.glob(initial_pred_pattern, recursive=True)[0]

        root_dir = benchmark_dir[0:benchmark_dir.rindex(template_str)-1]
        initial_pred_pattern = '%s/**/initial_pred*.pdb' % root_dir
        benchmark_initial_pred_path = glob.glob(initial_pred_pattern, recursive=True)[0]

        if uniprot_id not in list(conformational_states_df['uniprot_id']):
            print('SKIPPING %s' % uniprot_id)
            continue 
        else:   
            print('RUNNING for %s' % uniprot_id)

        #arg1 = '--conformation_info_dir=%s' % f
        #arg2 = '--skip_relaxation'
        #script_arguments = [arg1]
        #cluster_conformations_wrapper(script_arguments)
        
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
        
        pca_df = compare_clustered_conformations_pca(uniprot_id, pdb_id_ref, rw_initial_pred_path, benchmark_initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testing_data_af_weights/pca_rw_benchmark_comparison/%s/noref' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)

        '''pca_df = compare_clustered_conformations_pca_w_ref(uniprot_id, pdb_id_ref, initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testing_data_af_weights/pca_rw_benchmark_comparison/%s/ref1' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)

        pca_df = compare_clustered_conformations_pca_w_ref(uniprot_id, pdb_id_state_i, initial_pred_path, rw_cluster_representative_conformation_info_dir, benchmark_cluster_representative_conformation_info_dir)
        pca_save_dir = './conformational_states_testing_data_af_weights/pca_rw_benchmark_comparison/%s/ref2' % (uniprot_id)
        Path(pca_save_dir).mkdir(parents=True, exist_ok=True)
        pca_df.to_csv('%s/pca_comparison_metrics.csv' % pca_save_dir, index=False)''' 




conformational_states_df = pd.read_csv('./conformational_states_dataset/dataset/conformational_states_filtered_adjudicated_post_AF_training_final.csv')
conformational_states_df = conformational_states_df[conformational_states_df['any_structures_present_in_training_set'] == 'n']
print(conformational_states_df)

template_str = 'template=none' 
run_pca_pipeline(template_str, conformational_states_df)
#run_gt_comparison_pipeline('benchmark', template_str, conformational_states_df)
#run_gt_comparison_pipeline('rw', template_str, conformational_states_df)


conformational_states_df = pd.read_csv('./afsample2_dataset/conformational_states_testing_data_processed_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['any_structures_present_in_training_set'] == 'n']
print(conformational_states_df)

template_str = 'template=none'
run_pca_pipeline(template_str, conformational_states_df) 
#run_gt_comparison_pipeline('benchmark', template_str, conformational_states_df)
#run_gt_comparison_pipeline('rw', template_str, conformational_states_df)
