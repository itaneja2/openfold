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

from eval_pred_conformations import get_clustered_conformations_metrics 


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



def get_out_df(method_str, template_str, conformational_states_df):

    monomer_or_multimer = 'monomer'

    if method_str == 'rw':
        conformation_info_pattern = './conformational_states_testing_data/rw_predictions/*/*/%s/**/conformation_info.pkl' % template_str
    elif method_str == 'benchmark':
        conformation_info_pattern = './conformational_states_testing_data/benchmark_predictions/*/*/%s/conformation_info.pkl' % template_str 

    files = glob.glob(conformation_info_pattern, recursive=True)

    if method_str == 'rw':
        conformation_info_files_all_uniprot_id = [] 
        for f in files:
            if 'target=conformation0' in f:
                conformation_info_files_all_uniprot_id.append(f)
    else:
        conformation_info_files_all_uniprot_id = files 

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

        arg1 = '--conformation_info_dir=%s' % f
        #arg2 = '--skip_relaxation'
        script_arguments = [arg1]
        cluster_conformations_wrapper(script_arguments)

        cluster_representative_conformation_info_fname = glob.glob('%s/**/cluster_representative_conformation_info.pkl' % f, recursive=True)[0]
        cluster_representative_conformation_info_dir = cluster_representative_conformation_info_fname[0:cluster_representative_conformation_info_fname.rindex('/')]
       
        uniprot_id = get_uniprot_id(method_str, cluster_representative_conformation_info_fname)
        conformational_states_df_subset = conformational_states_df[conformational_states_df['uniprot_id'] == uniprot_id]
        pdb_id_ref = str(conformational_states_df_subset.iloc[0]['pdb_id_ref'])
        pdb_id_state_i = str(conformational_states_df_subset.iloc[0]['pdb_id_state_i'])

        out_df_pdb_id_ref = get_clustered_conformations_metrics(uniprot_id, pdb_id_ref, cluster_representative_conformation_info_dir, monomer_or_multimer, method_str, save=True)
        out_df_pdb_id_state_i = get_clustered_conformations_metrics(uniprot_id, pdb_id_state_i, cluster_representative_conformation_info_dir, monomer_or_multimer, method_str, save=True)


conformational_states_df = pd.read_csv('./conformational_states_testing_data/dataset/conformational_states_testing_data_processed_adjudicated.csv')

template_str = 'template=none' 
get_out_df('rw', template_str, conformational_states_df)
get_out_df('benchmark', template_str, conformational_states_df)
