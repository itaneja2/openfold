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
import re 
from typing import Any, List, Sequence, Optional, Tuple

from custom_openfold_utils.pdb_utils import get_pdb_path_seq, align_and_get_rmsd, get_cif_string_from_pdb, fetch_pdb
from custom_openfold_utils.fetch_utils import fetch_mmcif
from custom_openfold_utils.conformation_utils import get_residues_ignore_idx_between_pdb_conformations, get_residues_ignore_idx_between_pdb_af_conformation, get_conformation_vectorfield_spherical_coordinates

asterisk_line = '*********************************'

############################

def get_pdb_pred_path(pdb_pred_dir):
    files_in_pdb_pred_dir = [f for f in glob.glob(pdb_pred_dir) if os.path.isfile(f)]
    if len(files_in_pdb_pred_dir) == 0:
        print('no pdb predictions exist in %s' % pdb_pred_dir)
        return [] 
    else:
        pdb_pred_path = files_in_pdb_pred_dir[0]
        print('reading %s' % pdb_pred_path)
    
    return pdb_pred_path 

def save_conformation_vectorfield_training_data(data, output_fname):
    output_dir = './conformation_vectorfield_data' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def save_gtc_metadata(data, output_fname):
    output_dir = './ground_truth_conformation_data/metadata' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)




conformational_states_df = pd.read_csv('./dataset/conformational_states_testing_data_processed_adjudicated.csv')
conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 

af_conformations_residues_mask_dict = {}
pdb_conformations_residues_ignore_idx_dict = {} 
rmsd_dict = {} 
conformation_vectorfield_dict = {} 
uniprot_id_dict = {}
template_id_rw_conformation_path_dict = {} #maps uniprot_id-template_pdb_id to rw_conformation_path
num_rows_included = 0 

only_gen_for_clustered_conformations = True 
template_str = 'template=none' 

for index,row in conformational_states_df.iterrows():

    print(asterisk_line)
    print('On uniprot_id %d of %d' % (index, len(conformational_states_df)))
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref'])
    pdb_model_name_state_i = str(row['pdb_id_state_i'])

    alignment_dir = './alignment_data/%s' % uniprot_id
    rw_dir = './rw_predictions/%s' % uniprot_id 

    initial_pred_dir = '%s/alternative_conformations-verbose/%s/initial_pred/*' % (rw_dir, template_str)
    af_initial_pred_path = get_pdb_pred_path(initial_pred_dir)

    if len(af_initial_pred_path) == 0:
        continue 

    if only_gen_for_clustered_conformations:
        conformation_info_pattern = '%s/**/cluster_representative_conformation_info.pkl' % rw_dir
    else: 
        conformation_info_pattern = '%s/**/conformation_info.pkl' % rw_dir

    rw_conformation_info_files = glob.glob(conformation_info_pattern, recursive=True)

    if len(rw_conformation_info_files) == 0:
        continue 

    pdb_ref_struc_folder = './pdb_structures/pdb_ref_structure' 
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_state_i_folder = './pdb_structures/pdb_state_i'
    Path(pdb_state_i_folder).mkdir(parents=True, exist_ok=True)

    pdb_ref_path = fetch_pdb(pdb_model_name_ref, pdb_ref_struc_folder, clean=True)
    pdb_state_i_path = fetch_pdb(pdb_model_name_state_i, pdb_state_i_folder, clean=True)

    #combine into single dictionary
    for i in range(0,len(rw_conformation_info_files)):
        with open(rw_conformation_info_files[i], 'rb') as f:
            curr_conformation_info = pickle.load(f)
        if i == 0:
            rw_conformation_info = curr_conformation_info.copy()
        else:
            rw_conformation_info.update(curr_conformation_info)

    if only_gen_for_clustered_conformations:
        rw_conformation_paths_all = [rw_conformation_info[key][0] for key in rw_conformation_info]
    else:
        rw_conformation_paths_all = []
        for key in rw_conformation_info:
            for i,val in enumerate(rw_conformation_info[key]):
                pdb_path = val[0]
                rw_conformation_paths_all.append(pdb_path)

    print('%d total rw conformations for uniprot_id: %s' % (len(rw_conformation_paths_all),uniprot_id))

    for rw_conformation_path in rw_conformation_paths_all:

        uniprot_id_dict[rw_conformation_path] = uniprot_id

        rw_conformation_path = os.path.abspath(rw_conformation_path) 
        rw_conformation_fname = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_dir = rw_conformation_path[0:rw_conformation_path.rindex('/rw_output')]

        rmsd_dict[rw_conformation_path] = []

        for i,gtc_pdb_path in enumerate([pdb_ref_path, pdb_state_i_path]):

            if i == 0:
                pdb_model_name = pdb_model_name_ref
            else:
                pdb_model_name = pdb_model_name_state_i

            aligned_gtc_dir = '%s/cvf_eval_info/aligned_gtc=%s' % (rw_conformation_dir, pdb_model_name)
            os.makedirs(aligned_gtc_dir, exist_ok=True)
            shutil.copy(gtc_pdb_path, aligned_gtc_dir)
            aligned_gtc_pdb_path = '%s/%s.pdb' % (aligned_gtc_dir, pdb_model_name)
            aligned_gtc_pdb_path_renamed = '%s/%s_%s.pdb' % (aligned_gtc_dir, pdb_model_name, rw_conformation_fname) 
            shutil.move(aligned_gtc_pdb_path, aligned_gtc_pdb_path_renamed)
            aligned_gtc_pdb_path = os.path.abspath(aligned_gtc_pdb_path_renamed)

            rmsd = align_and_get_rmsd(rw_conformation_path, aligned_gtc_pdb_path) #align gtc to rw conformation  
            rmsd_dict[rw_conformation_path].append((rmsd, pdb_model_name, aligned_gtc_pdb_path))

        nearest_pdb_model_name, nearest_aligned_gtc_pdb_path = min(rmsd_dict[rw_conformation_path], key = lambda x: x[0])[1:]

        af_residues_ignore_idx = get_residues_ignore_idx_between_pdb_af_conformation(nearest_aligned_gtc_pdb_path, rw_conformation_path, af_initial_pred_path)
        if af_residues_ignore_idx is None:
            continue 

        conformation_vectorfield_spherical_coords, rw_residues_mask, rw_conformation_ca_pos, aligned_gtc_ca_pos = get_conformation_vectorfield_spherical_coordinates(rw_conformation_path, nearest_aligned_gtc_pdb_path, af_residues_ignore_idx)
        #rw_conformation_cif_string = get_cif_string_from_pdb(rw_conformation_path) 
        conformation_vectorfield_dict[rw_conformation_path] = (conformation_vectorfield_spherical_coords, nearest_aligned_gtc_pdb_path, nearest_pdb_model_name)

        #mask needs to be tied to rw_conformation, not pdb
        af_conformations_residues_mask_dict[rw_conformation_path] = rw_residues_mask
    

    num_rows_included += 1 






print('SAVING ALL TRAINING DATA') 

print('%d unique rows' % num_rows_included)

#print('saving rmsd_dict')   
#save_gtc_metadata(rmsd_dict, 'rmsd_dict')

print('saving uniprot_id_dict')
save_conformation_vectorfield_training_data(uniprot_id_dict, 'uniprot_id_dict')

print('saving af_conformations_residues_mask_dict')
save_conformation_vectorfield_training_data(af_conformations_residues_mask_dict, 'af_conformations_residues_mask_dict')

print('saving conformation_vectorfield_dict')
save_conformation_vectorfield_training_data(conformation_vectorfield_dict, 'conformation_vectorfield_dict')

