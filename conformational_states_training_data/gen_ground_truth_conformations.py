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
import json 
import numpy as np

sys.path.insert(0, '../')
from pdb_utils.pdb_utils import (
    get_pdb_path_seq, 
    get_rmsd,
    fetch_mmcif,
    get_residues_ignore_idx_between_pdb_conformations
)

asterisk_line = '*********************************'

############################

def get_pdb_pred_path(pdb_pred_dir):

    files_in_pdb_pred_dir = [f for f in glob.glob(pdb_pred_dir) if os.path.isfile(f)]

    if len(files_in_pdb_pred_dir) == 0:
        print('no pdb predictions exist in %s' % pdb_pred_dir)
        return [] 
    else:
        print(files_in_pdb_pred_dir)
        pdb_pred_path = files_in_pdb_pred_dir[0]
        print('reading %s' % pdb_pred_path)
    
    return pdb_pred_path 

def save_ground_truth_conformation(pdb_model_name, uniprot_id):

    #for training, we need the original mmcif 
    pdb_id, chain_id = pdb_model_name.split('_')
    cif_output_dir = './ground_truth_conformation_data/%s' % uniprot_id
    os.makedirs(cif_output_dir, exist_ok=True)
    print('fetching %s' % pdb_id)
    fetch_mmcif(pdb_id, cif_output_dir)
    cif_input_path = '%s/%s.cif' % (cif_output_dir, pdb_id)
    cif_output_path = '%s/%s.cif' % (cif_output_dir, pdb_model_name)
    os.rename(cif_input_path, cif_output_path)
    
    return cif_output_path 


def save_metadata(residues_ignore_idx_dict):

    metadata_output_dir = './ground_truth_conformation_data/metadata' 
    os.makedirs(metadata_output_dir, exist_ok=True)
    
    residues_ignore_idx_fname = '%s/residues_ignore_idx.pkl' % (metadata_output_dir, pdb_model_name)
    print('saving %s' % residues_ignore_idx_fname)
    with open(residues_ignore_idx_fname, 'wb') as f:
        pickle.dump(residues_ignore_idx_dict, f)


#def dump_residues_ignore_idx(pdb_model_name, uniprot_id, residues_ignore_idx):
#    cif_output_dir = './ground_truth_conformation_data/%s' % uniprot_id
#    residues_ignore_idx_fname = '%s/%s-residues_ignore_idx.pkl' % (cif_output_dir, pdb_model_name)
#    print('saving %s' % residues_ignore_idx_fname)
#    with open(residues_ignore_idx_fname, 'wb') as f:
#        pickle.dump(residues_ignore_idx, f)



 
conformational_states_df = pd.read_csv('../conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)


conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'P69441'].reset_index(drop=True)
#conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'A3XHF9'].reset_index(drop=True)

uniprot_pdb_dict = {} 
for index,row in conformational_states_df.iterrows(): 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_model_name_ref'])
    pdb_model_name_state_i = str(row['pdb_model_name_state_i'])
    if uniprot_id not in uniprot_pdb_dict:
        uniprot_pdb_dict[uniprot_id] = [pdb_model_name_ref, pdb_model_name_state_i]
    else:
        if pdb_model_name_ref not in uniprot_pdb_dict[uniprot_id]:
           uniprot_pdb_dict[uniprot_id].append(pdb_model_name_ref)
        if pdb_model_name_state_i not in uniprot_pdb_dict[uniprot_id]:
            uniprot_pdb_dict[uniprot_id].append(pdb_model_name_state_i)


residues_ignore_idx_dict = {}
rmsd_dict = {} 
spherical_coords_dict = {} 


for index,row in conformational_states_df.iterrows():

    #print(asterisk_line)
    #print('On uniprot_id %d of %d' % (index, len(uniprot_pdb_dict.keys())))
    #print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_model_name_ref'])
    pdb_model_name_state_i = str(row['pdb_model_name_state_i'])
    seg_len = int(row['seg_len'])

    ref_original_cif_path = save_ground_truth_conformation(pdb_model_name_ref, uniprot_id)
    state_i_original_cif_path = save_ground_truth_conformation(pdb_model_name_state_i, uniprot_id)

    alignment_dir = './alignment_data/%s' % uniprot_id
    rw_dir = './rw_predictions/%s' % uniprot_id 

    #these pdbs have already been cleaned and relevant chain extracted via openMM, so all correspond to chain A in the file  
    pdb_ref_path = '../conformational_states_dataset/pdb_structures/pdb_ref_structure/%s.pdb' % pdb_model_name_ref
    pdb_state_i_path = '../conformational_states_dataset/pdb_structures/pdb_superimposed_structures/%s.pdb' % pdb_model_name_state_i

    pdb_ref_pred_dir = '%s/template=%s/*/*/initial_pred/*' % (rw_dir, pdb_model_name_ref)
    pdb_state_i_pred_dir = '%s/template=%s/*/*/initial_pred/*' % (rw_dir, pdb_model_name_state_i)

    pdb_ref_pred_path = get_pdb_pred_path(pdb_ref_pred_dir)
    pdb_state_i_pred_path = get_pdb_pred_path(pdb_state_i_pred_dir)

    pdb_ref_ignore_residues_idx, pdb_state_i_ignore_residues_idx = get_residues_ignore_idx_between_pdb_conformations(pdb_ref_path, pdb_state_i_path, pdb_ref_pred_path, pdb_state_i_pred_path)

    residues_ignore_idx_dict[pdb_model_name_ref] = pdb_ref_ignore_residues_idx
    residues_ignore_idx_dict[pdb_state_i_ignore_residues_idx] = pdb_state_i_ignore_residues_idx 

    rw_conformations = glob.glob('%s/*/*/*/*/bootstrap/ACCEPTED/*.pdb' % rw_data_dir_curr_uniprot_id) 

    for rw_conformation_path in rw_conformations:
        rmsd_dict[rw_conformation_path] = {}
        for ground_truth_conformation_path in [pdb_ref_path, pdb_state_i_path]:
            rmsd = get_rmsd(rw_conformation_path, ground_truth_conformation_path)
            rmsd_dict[rw_conformation_path][ground_truth_conformation_path] = rmsd

    for rw_conformation_path in rw_conformations:
        nearest_gtc_path = min(rmsd_dict[rw_conformation_path], key=rmsd_dict.get)
        if nearest_gtc_path == pdb_ref_path:
            residues_ignore_idx = pdb_ref_ignore_residues_idx
        elif nearest_gtc_path == pdb_state_i_path:
            residues_ignore_idx = pdb_state_i_ignore_residues_idx

        spherical_coords_vector_diff = get_spherical_coordinate_vector_diff(rw_conformation_path, nearest_gtc_path, residues_ignore_idx) 
        spherical_coords_dict[rw_conformation_path] = spherical_coords_vector_diff
    
    
    print(rmsd_dict)


save_metadata(residues_ignore_idx_dict)
