import argparse
from pathlib import Path
import os
import subprocess 
import shutil 
import pandas as pd
import itertools
import pickle 
import glob
import sys  
import boto3
from datetime import date
import io 
import requests
import urllib.request
from pymol import cmd
import json 
from joblib import Parallel, delayed
import numpy as np

sys.path.insert(0, '../')
from pdb_utils.pdb_utils import (
    get_pdb_path_seq, 
    align_and_get_rmsd,
    get_residues_idx_in_seq2_not_seq1,
    get_af_disordered_residues, 
    get_pdb_disordered_residues_idx,
    delete_residues
)

asterisk_line = '*********************************'

############################

 
conformational_states_df = pd.read_csv('../conformational_states_dataset/data/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

uniprot_pdb_dict = {} 

for index,row in conformational_states_df.iterrows():
 
    uniprot_id = str(row['uniprot_id'])
    pdb_id_ref = str(row['pdb_id_ref'])
    pdb_id_state_i = str(row['pdb_id_state_i'])

    if uniprot_id not in uniprot_pdb_dict:
        uniprot_pdb_dict[uniprot_id] = [pdb_id_ref, pdb_id_state_i]
    else:
        if pdb_id_ref not in uniprot_pdb_dict[uniprot_id]:
           uniprot_pdb_dict[uniprot_id].append(pdb_id_ref)
        if pdb_id_state_i not in uniprot_pdb_dict[uniprot_id]:
            uniprot_pdb_dict[uniprot_id].append(pdb_id_state_i)


for index,row in conformational_states_df.iterrows():

    print(asterisk_line)
    print('On uniprot_id %d of %d' % (index, len(uniprot_pdb_dict.keys())))
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_id_ref = str(row['pdb_id_ref'])
    pdb_id_state_i = str(row['pdb_id_state_i'])
    seg_len = int(row['seg_len'])

    alignment_dir = '../conformational_states_dataset/alignment_data/%s' % uniprot_id
    output_dir_base = '../conformational_states_dataset/predictions/%s' % uniprot_id 

    module_config = 'module_config_0'
    rw_hp_config = 'hp_config_0-0' 
    output_dir = '%s/%s/%s/rw-%s' % (output_dir_base, 'rw', module_config, rw_hp_config)
    l1_output_dir = '%s/%s/%s' % (output_dir_base, 'rw', module_config)
    initial_pred_output_dir = '%s/initial_pred' %  l1_output_dir
    bootstrap_output_dir = '%s/bootstrap' % output_dir

    conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
    pdb_pred_path = '%s/initial_pred_unrelaxed.pdb' % initial_pred_output_dir 

    if not(os.path.exists(pdb_pred_path)):
        print('%s does not exist' % pdb_pred_path)
        continue
    else:
        print('reading %s' % pdb_pred_path) 

    #these pdbs have already been cleaned 
    pdb_ref_path = '../conformational_states_dataset/pdb_structures/pdb_ref_structure/%s.pdb' % pdb_id_ref
    pdb_state_i_path = '../conformational_states_dataset/pdb_structures/pdb_superimposed_structures/%s.pdb' % pdb_id_state_i

    rmsd_pdb_ref_wrt_pred = round(align_and_get_rmsd(pdb_pred_path, pdb_ref_path, 'A', 'A'),3)
    rmsd_pdb_state_i_wrt_pred = round(align_and_get_rmsd(pdb_pred_path, pdb_state_i_path, 'A', 'A'),3)

    print('RMSD w.r.t state 1: %.3f' % rmsd_pdb_ref_wrt_pred)
    print('RMSD w.r.t state 2: %.3f' % rmsd_pdb_state_i_wrt_pred)

    for i in range(0,2):

        if i == 0:
            rel_pdb_path = pdb_ref_path
            rel_pdb_id = pdb_id_ref
        else:
            rel_pdb_path = pdb_state_i_path
            rel_pdb_id = pdb_id_state_i

        print('generating conformation for %s' % rel_pdb_path)

        pred_seq = get_pdb_path_seq(pdb_pred_path, None)
        rel_pdb_seq = get_pdb_path_seq(rel_pdb_path, None)

        try: 
            af_disordered_domains_idx, af_disordered_domains_seq = get_af_disordered_residues(pdb_pred_path)      
        except ValueError as e:
            print('TROUBLE PARSING AF PREDICTION, SKIPPING') 
            print(e)
            continue

        pdb_disordered_idx, pdb_disordered_seq = get_pdb_disordered_residues_idx(pred_seq, rel_pdb_seq, af_disordered_domains_idx) 
        exclusive_pdb_residues_idx = get_residues_idx_in_seq2_not_seq1(pred_seq, rel_pdb_seq) 

        print('AF disordered:')
        print(af_disordered_domains_idx)
        print(af_disordered_domains_seq)

        print('PDB disordered:')
        print(pdb_disordered_idx)
        print(pdb_disordered_seq)

        print('Residues in PDB, but not AF:')
        print(exclusive_pdb_residues_idx)

        residues_delete_idx = pdb_disordered_idx + exclusive_pdb_residues_idx
        print('PDB RESIDUES TO REMOVE IDX:')
        print(residues_delete_idx)

        state_num = uniprot_pdb_dict[uniprot_id].index(rel_pdb_id)

        if len(exclusive_pdb_residues_idx) <= 5:
            pdb_model_name = rel_pdb_path.split('/')[-1]
            pdb_id = pdb_model_name.split('_')[0]
            pdb_output_dir = './conformation_data/state_%d/%s' % (state_num,uniprot_id)
            if not(os.path.exists(pdb_output_dir)):
                os.makedirs(pdb_output_dir, exist_ok=True)
                pdb_output_path = '%s/%s.pdb' % (pdb_output_dir, pdb_id)
                print('pdb output path: %s' % pdb_output_path)
                delete_residues(rel_pdb_path, pdb_output_path, residues_delete_idx)
        else:
            print("SKIPPING this PDB because more than 5 residues exist in PDB but not in AF")

