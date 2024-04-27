import logging
import os 
import io 
import subprocess 
import numpy as np
import pandas as pd 
import string
from io import StringIO
from pathlib import Path
import shutil 
import sys

from typing import List

from custom_openfold_utils.seq_utils import align_seqs, get_residues_idx_in_seq2_not_seq1, get_residues_idx_in_seq1_and_seq2 
from custom_openfold_utils.pdb_utils import get_ca_coords_dict, get_ca_coords_matrix, get_af_disordered_domains, get_pdb_disordered_domains 

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

 
def get_residue_idx_below_rmsf_threshold(pdb1_seq, pdb2_seq, pdb1_path, pdb2_path, pdb1_include_idx, pdb2_include_idx, pdb1_exclude_idx, pdb2_exclude_idx, rmsf_threshold = 2.0):

    pdb1_seq_aligned, pdb2_seq_aligned, pdb1_seq_aligned_to_original_idx_mapping, pdb2_seq_aligned_to_original_idx_mapping = align_seqs(pdb1_seq, pdb2_seq)

    pdb1_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb1_path)
    pdb2_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb2_path)
 
    pdb1_ca_pos_aligned = []
    pdb2_ca_pos_aligned = [] 

    pdb1_residue_idx_below_threshold = []
    pdb2_residue_idx_below_threshold = [] 

    for i in range(0,len(pdb1_seq_aligned)):
        if (pdb1_seq_aligned[i] == pdb2_seq_aligned[i]) and (pdb1_seq_aligned[i] != '-'):
            pdb1_seq_original_idx = pdb1_seq_aligned_to_original_idx_mapping[i]
            pdb2_seq_original_idx = pdb2_seq_aligned_to_original_idx_mapping[i]
            valid_idx = (pdb1_seq_original_idx in pdb1_include_idx) and (pdb2_seq_original_idx in pdb2_include_idx) and (pdb1_seq_original_idx not in pdb1_exclude_idx) and (pdb2_seq_original_idx not in pdb2_exclude_idx)
            if valid_idx:
                logger.info("aligned idx for pdb1: %d, seq idx: %d:" % (i, pdb1_seq_original_idx))
                logger.info("aligned idx for pdb2: %d, seq idx: %d:" % (i, pdb2_seq_original_idx))
                pdb1_ca_pos = list(pdb1_residue_idx_ca_coords_dict[pdb1_seq_original_idx][0:3])
                pdb2_ca_pos = list(pdb2_residue_idx_ca_coords_dict[pdb2_seq_original_idx][0:3])
                curr_ca_rmsf = np.linalg.norm(np.array(pdb1_ca_pos) - np.array(pdb2_ca_pos))
                logger.info(curr_ca_rmsf)
                if curr_ca_rmsf < rmsf_threshold:
                    pdb1_residue_idx_below_threshold.append(pdb1_seq_original_idx)
                    pdb2_residue_idx_below_threshold.append(pdb2_seq_original_idx)
                pdb1_ca_pos_aligned.append(pdb1_ca_pos)
                pdb2_ca_pos_aligned.append(pdb2_ca_pos)
                
    pdb1_ca_pos_aligned = np.array(pdb1_ca_pos_aligned)
    pdb2_ca_pos_aligned = np.array(pdb2_ca_pos_aligned)

    ca_rmsf = np.linalg.norm(pdb1_ca_pos_aligned-pdb2_ca_pos_aligned, axis=1)

    logger.info(ca_rmsf)
    logger.info(pdb1_residue_idx_below_threshold)
    logger.info(pdb2_residue_idx_below_threshold)

    return pdb1_residue_idx_below_threshold, pdb2_residue_idx_below_threshold

def get_residues_ignore_idx_between_pdb_conformations(state1_pdb_path, state2_pdb_path, state1_af_path, state2_af_path):

    """Ignore residues based on:
        1. disorder
        2. common residues
        3. rmsf threshold
    """  

    state1_pdb_seq = get_pdb_path_seq(state1_pdb_path, None)
    state2_pdb_seq = get_pdb_path_seq(state2_pdb_path, None)
    state1_af_seq = get_pdb_path_seq(state1_af_path, None)
    state2_af_seq = get_pdb_path_seq(state2_af_path, None)

    try: 
        state1_af_disordered_domains_idx, _ = get_af_disordered_domains(state1_af_path)      
    except ValueError as e:
        logger.info('TROUBLE PARSING AF PREDICTION') 
        logger.info(e)
        state1_af_disordered_domains_idx = [] 

    try: 
        state2_af_disordered_domains_idx, _ = get_af_disordered_domains(state2_af_path)      
    except ValueError as e:
        logger.info('TROUBLE PARSING AF PREDICTION') 
        logger.info(e)
        state2_af_disordered_domains_idx = [] 

    state1_pdb_disordered_domains_idx, _ = get_pdb_disordered_domains(state1_af_seq, state1_pdb_seq, state1_af_disordered_domains_idx) 
    state2_pdb_disordered_domains_idx, _ = get_pdb_disordered_domains(state2_af_seq, state2_pdb_seq, state2_af_disordered_domains_idx) 
    
    state1_pdb_exclusive_residues_idx = get_residues_idx_in_seq2_not_seq1(state1_af_seq, state1_pdb_seq)
    if len(state1_pdb_exclusive_residues_idx) > 0:
        logger.info('Residues in PDB, but not AF:')
        logger.info(pdb_exclusive_residues_idx)
        logger.info('This should be empty')
        logger.info('exiting...')
        sys.exit()
           
    state1_pdb_common_residues_idx, state2_pdb_common_residues_idx = get_residues_idx_in_seq1_and_seq2(state1_pdb_seq, state2_pdb_seq)
    state1_pdb_rmsf_below_threshold_idx, state2_pdb_rmsf_below_threshold_idx = get_residue_idx_below_rmsf_threshold(
                                                                                                state1_pdb_seq,
                                                                                                state2_pdb_seq,
                                                                                                state1_pdb_path,
                                                                                                state2_pdb_path,
                                                                                                state1_pdb_common_residues_idx,
                                                                                                state2_pdb_common_residues_idx,
                                                                                                state1_pdb_disordered_domains_idx,
                                                                                                state2_pdb_disordered_domains_idx)

    state1_pdb_common_residues_idx_complement = sorted(list(set(range(len(state1_pdb_seq))) - set(state1_pdb_common_residues_idx)))
    state2_pdb_common_residues_idx_complement = sorted(list(set(range(len(state2_pdb_seq))) - set(state2_pdb_common_residues_idx)))

    logger.info('STATE 1 PDB disordered:')
    logger.info(state1_pdb_disordered_domains_idx)
    logger.info('STATE 1 PDB intersection complement:')
    logger.info(state1_pdb_common_residues_idx_complement)
    logger.info('STATE 1 PDB below RMSF:')
    logger.info(state1_pdb_rmsf_below_threshold_idx)

    logger.info('STATE 2 PDB disordered:')
    logger.info(state2_pdb_disordered_domains_idx)
    logger.info('STATE 2 PDB intersection complement:')
    logger.info(state2_pdb_common_residues_idx_complement)
    logger.info('STATE 2 PDB below RMSF:')
    logger.info(state2_pdb_rmsf_below_threshold_idx)
    
    state1_pdb_residues_ignore_idx = state1_pdb_disordered_domains_idx+state1_pdb_common_residues_idx_complement+state1_pdb_rmsf_below_threshold_idx
    state2_pdb_residues_ignore_idx = state2_pdb_disordered_domains_idx+state2_pdb_common_residues_idx_complement+state2_pdb_rmsf_below_threshold_idx

    state1_pdb_residues_ignore_idx = tuple(sorted(set(state1_pdb_residues_ignore_idx)))
    state2_pdb_residues_ignore_idx = tuple(sorted(set(state2_pdb_residues_ignore_idx)))


    return state1_pdb_residues_ignore_idx, state2_pdb_residues_ignore_idx


def get_residues_ignore_idx_between_af_conformations(state1_af_path, state2_af_path, af_initial_pred_path, rmsf_threshold=1.0):

    """Ignore residues based on:
        1. disorder
        2. rmsf threshold
        
        Assumption: same sequence for all structures
        we use stricted rmsf threshold here 

    """  

    try: 
        af_disordered_domains_idx, _ = get_af_disordered_domains(af_initial_pred_path)      
    except ValueError as e:
        logger.info('TROUBLE PARSING AF PREDICTION') 
        logger.info(e)
        af_disordered_domains_idx = [] 

    state1_af_ca_pos = get_ca_coords_matrix(state1_af_path)
    state2_af_ca_pos = get_ca_coords_matrix(state2_af_path)

    ca_rmsf = np.linalg.norm(state2_af_ca_pos-state1_af_ca_pos, axis=1)
    af_rmsf_below_threshold_idx = list(np.where(ca_rmsf < rmsf_threshold)[0])
    
    af_residues_ignore_idx = af_disordered_domains_idx+af_rmsf_below_threshold_idx
    af_residues_ignore_idx = tuple(sorted(set(af_residues_ignore_idx)))

    logger.info(af_disordered_domains_idx)
    logger.info(af_rmsf_below_threshold_idx)


    return af_residues_ignore_idx

 

def cartesian_to_spherical(ca_pos_diff):
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""

    #utilizing this convention: https://dynref.engr.illinois.edu/rvs.html
    radius = np.linalg.norm(ca_pos_diff, axis=1)
    xy_norm = np.linalg.norm(ca_pos_diff[:,0:2], axis=1)
    phi = np.arctan2(xy_norm, ca_pos_diff[:,2])
    theta = np.arctan2(ca_pos_diff[:,1], ca_pos_diff[:,0])

    return phi, theta, radius


def get_conformation_vectorfield_spherical_coordinates(af_pred_path, pdb_path, pdb_residues_ignore_idx):

    #transformation taking af_pred_path --> pdb_path

    af_pred_seq = get_pdb_path_seq(af_pred_path, None)
    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_pred_seq_aligned, pdb_seq_aligned, af_pred_seq_aligned_to_original_idx_mapping, pdb_seq_aligned_to_original_idx_mapping = align_seqs(af_pred_seq, pdb_seq)

    af_pred_residue_idx_ca_coords_dict = get_ca_coords_dict(af_pred_path)
    pdb_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb_path)
    
    af_residues_include_idx = [] 

    pdb_ca_pos_aligned = []
    af_pred_ca_pos_aligned = [] 

    for i in range(0,len(af_pred_seq_aligned)):
        if af_pred_seq_aligned[i] != '-' and pdb_seq_aligned[i] != '-':
            pdb_seq_original_idx = pdb_seq_aligned_to_original_idx_mapping[i]
            af_pred_seq_original_idx = af_pred_seq_aligned_to_original_idx_mapping[i]
            if pdb_seq_original_idx not in pdb_residues_ignore_idx:
                af_residues_include_idx.append(af_pred_seq_original_idx)
                pdb_ca_pos = list(pdb_residue_idx_ca_coords_dict[pdb_seq_original_idx][0:3])
                af_pred_ca_pos = list(af_pred_residue_idx_ca_coords_dict[af_pred_seq_original_idx][0:3])
                pdb_ca_pos_aligned.append(pdb_ca_pos)
                af_pred_ca_pos_aligned.append(af_pred_ca_pos)
            else:
                pdb_ca_pos_aligned.append([0,0,0])
                af_pred_ca_pos_aligned.append([0,0,0])
        elif af_pred_seq_aligned[i] != '-' and pdb_seq_aligned[i] == '-':
            pdb_ca_pos_aligned.append([0,0,0])
            af_pred_ca_pos_aligned.append([0,0,0])
            
    pdb_ca_pos_aligned = np.array(pdb_ca_pos_aligned)
    af_pred_ca_pos_aligned = np.array(af_pred_ca_pos_aligned)

    ca_pos_diff = pdb_ca_pos_aligned - af_pred_ca_pos_aligned
    phi, theta, radius = cartesian_to_spherical(ca_pos_diff)

    af_residues_mask = [] 
    for i in range(0, len(af_pred_seq)):
        if i in af_residues_include_idx:
            af_residues_mask.append(1)
        else:
            af_residues_mask.append(0)

    return np.array([af_residues_include_idx, phi, theta, radius]), af_residues_mask, af_pred_ca_pos_aligned, pdb_ca_pos_aligned

