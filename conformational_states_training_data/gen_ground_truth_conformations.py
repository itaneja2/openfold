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
    align_and_get_rmsd,
    fetch_mmcif,
    get_residues_ignore_idx_between_pdb_conformations,
    get_vector_diff_spherical_coordinates
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


def save_vector_diff_training_data(data, output_fname):

    output_dir = './vector_diff_data' 
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

 
conformational_states_df = pd.read_csv('../conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)


conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'P69441'].reset_index(drop=True)
#conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'A3XHF9'].reset_index(drop=True)

uniprot_pdb_dict = {} 
for index,row in conformational_states_df.iterrows(): 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref'])
    pdb_model_name_state_i = str(row['pdb_id_state_i'])
    if uniprot_id not in uniprot_pdb_dict:
        uniprot_pdb_dict[uniprot_id] = [pdb_model_name_ref, pdb_model_name_state_i]
    else:
        if pdb_model_name_ref not in uniprot_pdb_dict[uniprot_id]:
           uniprot_pdb_dict[uniprot_id].append(pdb_model_name_ref)
        if pdb_model_name_state_i not in uniprot_pdb_dict[uniprot_id]:
            uniprot_pdb_dict[uniprot_id].append(pdb_model_name_state_i)


residues_ignore_idx_dict = {}
rmsd_dict = {} 
vector_diff_dict = {} 


for index,row in conformational_states_df.iterrows():

    #print(asterisk_line)
    #print('On uniprot_id %d of %d' % (index, len(uniprot_pdb_dict.keys())))
    #print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref'])
    pdb_model_name_state_i = str(row['pdb_id_state_i'])
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
    residues_ignore_idx_dict[pdb_model_name_state_i] = pdb_state_i_ignore_residues_idx 

    #print(pdb_ref_path)
    #vector_diff_spherical_coords = get_spherical_coordinate_vector_diff(pdb_ref_path, pdb_state_i_path, pdb_ref_ignore_residues_idx) 
    #print(vector_diff_spherical_coords) 
    #sdf

    rw_conformations = sorted(glob.glob('%s/*/*/*/*/bootstrap/ACCEPTED/*.pdb' % rw_dir))

    for conformation_num,rw_conformation_path in enumerate(rw_conformations):

        print("ON CONFORMATION %d/%d" % (conformation_num,len(rw_conformations)))

        print(rw_conformation_path)
        rw_conformation_fname = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]

        rmsd_dict[rw_conformation_path] = []

        for i,gtc_path in enumerate([pdb_ref_path, pdb_state_i_path]):

            if i == 0:
                pdb_model_name = pdb_model_name_ref
            else:
                pdb_model_name = pdb_model_name_state_i

            aligned_gtc_dir = '%s/aligned_gtc=%s' % (rw_conformation_dir, pdb_model_name)
            os.makedirs(aligned_gtc_dir, exist_ok=True)
            shutil.copy(gtc_path, aligned_gtc_dir)
            aligned_gtc_path = '%s/%s.pdb' % (aligned_gtc_dir, pdb_model_name)
            aligned_gtc_path_renamed = '%s/%s_%s.pdb' % (aligned_gtc_dir, pdb_model_name, rw_conformation_fname) 
            shutil.move(aligned_gtc_path, aligned_gtc_path_renamed)
            aligned_gtc_path = aligned_gtc_path_renamed

            rmsd = align_and_get_rmsd(rw_conformation_path, aligned_gtc_path) #align gtc to rw conformation  
            rmsd_dict[rw_conformation_path].append((rmsd, pdb_model_name, aligned_gtc_path))
        
        print(rmsd_dict[rw_conformation_path])

        nearest_pdb_model_name, nearest_aligned_gtc_path = min(rmsd_dict[rw_conformation_path], key = lambda x: x[0])[1:3]

        if pdb_model_name_ref == nearest_pdb_model_name:
            residues_ignore_idx = pdb_ref_ignore_residues_idx
        elif pdb_model_name_state_i == nearest_pdb_model_name:
            residues_ignore_idx = pdb_state_i_ignore_residues_idx

        residues_ignore_idx_dict[rw_conformation_path] = residues_ignore_idx

        vector_diff_spherical_coords, rw_conformation_ca_pos, aligned_gtc_ca_pos = get_vector_diff_spherical_coordinates(rw_conformation_path, nearest_aligned_gtc_path, residues_ignore_idx) 
        vector_diff_dict[rw_conformation_path] = (vector_diff_spherical_coords, rw_conformation_path, nearest_aligned_gtc_path)

        phi = vector_diff_spherical_coords[1]
        theta = vector_diff_spherical_coords[2]
        r = vector_diff_spherical_coords[3]
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        vector_diff_cartesian = np.transpose(np.array([x,y,z]))
        print(vector_diff_cartesian.shape)
        print(vector_diff_cartesian + rw_conformation_ca_pos)

        #print(rw_conformation_path)
        #print(vector_diff_spherical_coords)

        #print(rw_conformation_ca_pos)

        


   
    #print(vector_diff_dict) 
    print('********')
    print(rmsd_dict)

save_vector_diff_training_data(rmsd_dict, 'rmsd_dict')
save_vector_diff_training_data(residues_ignore_idx_dict, 'residues_ignore_idx_dict')
save_vector_diff_training_data(vector_diff_dict, 'vector_diff_dict')
