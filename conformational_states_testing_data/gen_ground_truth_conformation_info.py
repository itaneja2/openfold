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

from custom_openfold_utils.pdb_utils import get_pdb_path_seq, align_and_get_rmsd, get_cif_string_from_pdb
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
        print(files_in_pdb_pred_dir)
        pdb_pred_path = files_in_pdb_pred_dir[0]
        print('reading %s' % pdb_pred_path)
    
    return pdb_pred_path 

def get_bootstrap_candidate_conformations(
    conformation_info: List[Tuple[Any,...]],
    num_training_conformations=30,
    return_all_conformations=False
):
    """Generates a set of candidate conformations to use for the gradient descent phase
       of the pipeline. 
       
       The candidate conformations are derived from the bootstrap phase. More specifically,
       the candidate conformations correspond to those that were outputted one step prior 
       to a rejected conformation.   
    """ 
  
    all_bootstrap_conformations = []  
 
    for i in range(0,len(conformation_info)):
        f = conformation_info[i][0]
        rmsd = conformation_info[i][1]
        match = re.search(r'_iter(\d+)', f)
        iter_num = int(match.group(1)) #this corresponds to a given iteration (i.e a sequence of conformations that terminates in a rejection
        match = re.search(r'step-iter(\d+)', f) 
        step_num = int(match.group(1)) #this corresponds to the step_num for a given iteration 
        all_bootstrap_conformations.append(f)

    if return_all_conformations:
        return all_bootstrap_conformations

    all_bootstrap_conformations = sorted(all_bootstrap_conformations, key=lambda x:x[1], reverse=True) #sort by rmsd in reverse order 
    
    num_training_conformations = min(num_training_conformations, len(all_bootstrap_conformations))
    print('num candidate conformations: %d' % len(all_bootstrap_conformations))
    bootstrap_candidate_conformations = all_bootstrap_conformations[0:num_training_conformations]   
 
    return bootstrap_candidate_conformations



def save_ground_truth_conformation(pdb_model_name, uniprot_id):
    #for training, we need the original mmcif 
    pdb_id, chain_id = pdb_model_name.split('_')
    cif_output_dir = './ground_truth_conformation_data/%s' % uniprot_id
    os.makedirs(cif_output_dir, exist_ok=True)
    print('fetching %s' % pdb_id)
    fetch_mmcif(pdb_id, cif_output_dir)
    cif_input_path = '%s/%s.cif' % (cif_output_dir, pdb_id)
    cif_output_path = '%s/%s.cif' % (cif_output_dir, pdb_model_name)
    while not(os.path.exists(cif_input_path)):
        print('retrying to fetch %s' % pdb_id)
        fetch_mmcif(pdb_id, cif_output_dir)
    os.rename(cif_input_path, cif_output_path)
    
    return os.path.abspath(cif_output_path)


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




 
conformational_states_df = pd.read_csv('./dataset/conformational_states_testing_data.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

af_conformations_residues_mask_dict = {}
pdb_conformations_residues_ignore_idx_dict = {} 
rmsd_dict = {} 
conformation_vectorfield_dict = {} 
uniprot_id_dict = {}
template_id_rw_conformation_path_dict = {} #maps uniprot_id-template_pdb_id to rw_conformation_path
num_rows_included = 0 

for index,row in conformational_states_df.iterrows():

    print(asterisk_line)
    print('On uniprot_id %d of %d' % (index, len(conformational_states_df)))
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref'])
    pdb_model_name_state_i = str(row['pdb_id_state_i'])

    ref_original_cif_path = save_ground_truth_conformation(pdb_model_name_ref, uniprot_id)
    state_i_original_cif_path = save_ground_truth_conformation(pdb_model_name_state_i, uniprot_id)

    alignment_dir = './alignment_data/%s' % uniprot_id
    rw_dir = './rw_predictions/%s' % uniprot_id 

    #these pdbs have already been cleaned and relevant chain extracted via openMM, so all correspond to chain A in the file  
    pdb_ref_path = '../conformational_states_dataset/pdb_structures/pdb_ref_structure/%s.pdb' % pdb_model_name_ref
    pdb_state_i_path = '../conformational_states_dataset/pdb_structures/pdb_superimposed_structures/%s.pdb' % pdb_model_name_state_i

    pdb_ref_pred_dir = '%s/template=%s/*/initial_pred/*' % (rw_dir, pdb_model_name_ref)
    pdb_state_i_pred_dir = '%s/template=%s/*/initial_pred/*' % (rw_dir, pdb_model_name_state_i)

    pdb_ref_af_pred_path = get_pdb_pred_path(pdb_ref_pred_dir)
    pdb_state_i_af_pred_path = get_pdb_pred_path(pdb_state_i_pred_dir)

    pdb_ref_ignore_residues_idx, pdb_state_i_ignore_residues_idx = get_residues_ignore_idx_between_pdb_conformations(pdb_ref_path, pdb_state_i_path, pdb_ref_af_pred_path, pdb_state_i_af_pred_path)
 
    #this is not used for training, just for later downstream analysis 
    pdb_conformations_residues_ignore_idx_dict[pdb_model_name_ref] = pdb_ref_ignore_residues_idx
    pdb_conformations_residues_ignore_idx_dict[pdb_model_name_state_i] = pdb_state_i_ignore_residues_idx
 
    rw_conformation_info = sorted(glob.glob('%s/*/*/*/*/training_conformations/conformation_info.pkl' % rw_dir))

    boostrap_candidate_conformations_all = [] 
    for i in range(0,len(rw_conformation_info)):
        with open(rw_conformation_info[i], 'rb') as f:
            conformation_info = pickle.load(f)
        if pdb_model_name_state_i in rw_conformation_info[i]:
            print('pdb_state_i')
            boostrap_candidate_conformations = get_bootstrap_candidate_conformations(conformation_info)
            num_candidate_conformations_state_i = len(boostrap_candidate_conformations)
            key = '%s-%s' % (uniprot_id, pdb_model_name_state_i)
            template_id_rw_conformation_path_dict[key] = boostrap_candidate_conformations
        else:
            print('pdb_ref')
            boostrap_candidate_conformations = get_bootstrap_candidate_conformations(conformation_info)
            num_candidate_conformations_ref = len(boostrap_candidate_conformations)
            key = '%s-%s' % (uniprot_id, pdb_model_name_ref)
            template_id_rw_conformation_path_dict[key] = boostrap_candidate_conformations

        boostrap_candidate_conformations_all.extend(boostrap_candidate_conformations)

    print('%d total training conformations for uniprot_id: %s' % (len(boostrap_candidate_conformations_all),uniprot_id))
    print('**** %d from pdb_ref' % (num_candidate_conformations_ref))
    print('**** %d from pdb_state_i' % (num_candidate_conformations_state_i))

    for conformation_num in range(0,len(boostrap_candidate_conformations_all)):

        rw_conformation_path = boostrap_candidate_conformations_all[conformation_num]

        uniprot_id_dict[rw_conformation_path] = uniprot_id

        print("ON CONFORMATION %d/%d" % (conformation_num,len(boostrap_candidate_conformations_all)))

        rw_conformation_path = os.path.abspath(rw_conformation_path) 
        rw_conformation_fname = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]

        rmsd_dict[rw_conformation_path] = []

        for i,gtc_pdb_path in enumerate([pdb_ref_path, pdb_state_i_path]):

            if i == 0:
                pdb_model_name = pdb_model_name_ref
                gtc_cif_path = ref_original_cif_path 
            else:
                pdb_model_name = pdb_model_name_state_i
                gtc_cif_path = state_i_original_cif_path

            aligned_gtc_dir = '%s/aligned_gtc=%s' % (rw_conformation_dir, pdb_model_name)
            os.makedirs(aligned_gtc_dir, exist_ok=True)
            shutil.copy(gtc_pdb_path, aligned_gtc_dir)
            aligned_gtc_pdb_path = '%s/%s.pdb' % (aligned_gtc_dir, pdb_model_name)
            aligned_gtc_pdb_path_renamed = '%s/%s_%s.pdb' % (aligned_gtc_dir, pdb_model_name, rw_conformation_fname) 
            shutil.move(aligned_gtc_pdb_path, aligned_gtc_pdb_path_renamed)
            aligned_gtc_pdb_path = os.path.abspath(aligned_gtc_pdb_path_renamed)

            rmsd = align_and_get_rmsd(rw_conformation_path, aligned_gtc_pdb_path) #align gtc to rw conformation  
            rmsd_dict[rw_conformation_path].append((rmsd, gtc_cif_path, pdb_model_name, aligned_gtc_pdb_path))

        nearest_pdb_model_name, nearest_aligned_gtc_pdb_path = min(rmsd_dict[rw_conformation_path], key = lambda x: x[0])[2:]

        if pdb_model_name_ref == nearest_pdb_model_name:
            af_initial_pred_path = pdb_ref_af_pred_path
        elif pdb_model_name_state_i == nearest_pdb_model_name:
            af_initial_pred_path = pdb_state_i_af_pred_path 

        af_residues_ignore_idx = get_residues_ignore_idx_between_pdb_af_conformation(nearest_aligned_gtc_pdb_path, rw_conformation_path, af_initial_pred_path)
        if af_residues_ignore_idx is None:
            continue 

        conformation_vectorfield_spherical_coords, rw_residues_mask, rw_conformation_ca_pos, aligned_gtc_ca_pos = get_conformation_vectorfield_spherical_coordinates(rw_conformation_path, nearest_aligned_gtc_pdb_path, af_residues_ignore_idx)
        #rw_conformation_cif_string = get_cif_string_from_pdb(rw_conformation_path) 
        conformation_vectorfield_dict[rw_conformation_path] = (conformation_vectorfield_spherical_coords, nearest_aligned_gtc_pdb_path, nearest_pdb_model_name)

        #mask needs to be tied to rw_conformation, not pdb
        af_conformations_residues_mask_dict[rw_conformation_path] = rw_residues_mask
    
        '''phi = conformation_vectorfield_spherical_coords[1]
        theta = conformation_vectorfield_spherical_coords[2]
        r = conformation_vectorfield_spherical_coords[3]
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        conformation_vectorfield_cartesian = np.transpose(np.array([x,y,z]))
        print('conformation_vectorfield_cartesian + rw_conformation_ca_pos')
        print(conformation_vectorfield_cartesian + rw_conformation_ca_pos)
        print('aligned_gtc_ca_pos')
        print(aligned_gtc_ca_pos)
        print('diff')
        print(aligned_gtc_ca_pos - (conformation_vectorfield_cartesian + rw_conformation_ca_pos))'''

    num_rows_included += 1 

    if index > 0 and index % 100 == 0: 
        print('SAVING CHECKPOINT TRAINING DATA')
        index_str = '_%d' % index  
        print('saving rmsd_dict')   
        save_gtc_metadata(rmsd_dict, 'rmsd_dict%s' % index_str)
        print('saving pdb_conformations_residues_ignore_idx_dict')
        save_gtc_metadata(pdb_conformations_residues_ignore_idx_dict, 'pdb_conformations_residues_ignore_idx_dict%s' % index_str)
        print('saving uniprot_id_dict')
        save_conformation_vectorfield_training_data(uniprot_id_dict, 'uniprot_id_dict%s' % index_str)
        print('saving af_conformations_residues_mask_dict')
        save_conformation_vectorfield_training_data(af_conformations_residues_mask_dict, 'af_conformations_residues_mask_dict%s' % index_str)
        print('saving conformation_vectorfield_dict')
        save_conformation_vectorfield_training_data(conformation_vectorfield_dict, 'conformation_vectorfield_dict%s' % index_str)
        print('saving template_id_rw_conformation_path_dict')
        save_conformation_vectorfield_training_data(template_id_rw_conformation_path_dict, 'template_id_rw_conformation_path_dict%s' % index_str) 





print('SAVING ALL TRAINING DATA') 

print('%d unique rows' % num_rows_included)

print('saving rmsd_dict')   
save_gtc_metadata(rmsd_dict, 'rmsd_dict')

print('saving pdb_conformations_residues_ignore_idx_dict')
save_gtc_metadata(pdb_conformations_residues_ignore_idx_dict, 'pdb_conformations_residues_ignore_idx_dict')

print('saving uniprot_id_dict')
save_conformation_vectorfield_training_data(uniprot_id_dict, 'uniprot_id_dict')

print('saving af_conformations_residues_mask_dict')
save_conformation_vectorfield_training_data(af_conformations_residues_mask_dict, 'af_conformations_residues_mask_dict')

print('saving conformation_vectorfield_dict')
save_conformation_vectorfield_training_data(conformation_vectorfield_dict, 'conformation_vectorfield_dict')

print('saving template_id_rw_conformation_path_dict')
save_conformation_vectorfield_training_data(template_id_rw_conformation_path_dict, 'template_id_rw_conformation_path_dict')
