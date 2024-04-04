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
    fetch_mmcif,
    select_rel_chain_and_delete_residues
)

asterisk_line = '*********************************'

############################

 
conformational_states_df = pd.read_csv('../conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
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

    alignment_dir = './alignment_data/%s' % uniprot_id
    rw_dir = './rw_predictions/%s' % uniprot_id 
    pdb_preds = [f for f in glob.glob('%s/*/*/*/initial_pred/*' % rw_dir) if os.path.isfile(f)]

    if len(pdb_preds) == 0:
        print('no pdb predictions exist in %s' % rw_dir)
        continue
    else:
        print(pdb_preds)
        pdb_pred_path = pdb_preds[0]
        print('reading %s' % pdb_pred_path) 

    #these pdbs have already been cleaned via openMM, so all correspond to chain A in the file  
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

        residues_ignore_idx = pdb_disordered_idx + exclusive_pdb_residues_idx
        print('PDB RESIDUES TO REMOVE IDX:')
        print(residues_ignore_idx)

        state_num = uniprot_pdb_dict[uniprot_id].index(rel_pdb_id)

        if len(exclusive_pdb_residues_idx) <= 5:
            pdb_model_name = rel_pdb_path.split('/')[-1].split('.')[0]
            pdb_id, chain_id = pdb_model_name.split('_')
            cif_output_dir = './ground_truth_conformation_data/%s' % uniprot_id
            os.makedirs(cif_output_dir, exist_ok=True)
            print('fetching %s' % pdb_id)
            fetch_mmcif(pdb_id, cif_output_dir)
            cif_input_path = '%s/%s.cif' % (cif_output_dir, pdb_id)
            cif_output_path = '%s/%s.cif' % (cif_output_dir, pdb_model_name)
            os.rename(cif_input_path, cif_output_path)
            residues_ignore_idx_fname = '%s/%s-residues_ignore_idx.pkl' % (cif_output_dir, pdb_model_name)
            with open(residues_ignore_idx_fname, 'wb') as f:
                pickle.dump(residues_ignore_idx, f)
            #print('cif output path: %s' % cif_output_path)
            #select_rel_chain_and_delete_residues(cif_input_path, cif_output_path, chain_id, residues_ignore_idx) 
            #os.remove(cif_input_path) 
        else:
            print("SKIPPING this PDB because more than 5 residues exist in PDB but not in AF")

