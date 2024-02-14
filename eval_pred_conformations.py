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

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO

import collections
import functools

from DockQ.DockQ import calc_DockQ

from pdb_utils.pdb_utils import superimpose_wrapper_monomer, superimpose_wrapper_multimer, renumber_chain_wrt_reference

def compare_conformations_to_structure_multimer(pdb_ref_id, pdb_ref_path, conformation_info_path, superimposed_structures_save_dir):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    mean_plddt_list = []  
    ptm_iptm_list = [] 

    fname = '%s/cluster_representative_conformation_info.pkl' % conformation_info_path
    with open(fname, 'rb') as f:
         conformation_info = pickle.load(f)
    for key in conformation_info:
        pdb_pred_path =  conformation_info[key][0] #this points to the original file 
        mean_plddt = conformation_info[key][2]
        ptm_iptm = conformation_info[key][3]
        pdb_pred_path_named_by_cluster = conformation_info[key][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
        pdb_pred_path_list.append(pdb_pred_path)
        pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
        mean_plddt_list.append(mean_plddt)
        ptm_iptm_list.append(ptm_iptm)

    print('PDB FILES:')
    print(pdb_pred_path_list)
        
    rmsd_list = [] 
    dockq_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (pdb_ref_id, pdb_pred_path))
        rmsd, pdb_ref_path_output, _ = superimpose_wrapper_multimer(pdb_ref_id, None, 'pdb', 'pred_rw', pdb_ref_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd_list.append(rmsd)
        num_chains = renumber_chain_wrt_reference(pdb_ref_path_output, pdb_pred_path) #arg1 is the pdb to renumber, arg2 is the pdb to renumber w.r.t to 
        if num_chains == 2:
            dockq_info = calc_DockQ(pdb_pred_path, pdb_ref_path_output)
            dockq_val = dockq_info['DockQ']
        else:
            dockq_val = np.nan 
        dockq_list.append(dockq_val)

    out_df = pd.DataFrame({'pdb_id_reference_structure': pdb_ref_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list,
                          'mean_plddt': mean_plddt_list, 'ptm_iptm': mean_plddt_list, 'rmsd': rmsd_list, 'dockq': dockq_list})
 
    return out_df

def compare_conformations_to_structure_monomer(pdb_ref_id, pdb_ref_path, conformation_info_path, superimposed_structures_save_dir):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    mean_plddt_list = []  

    fname = '%s/cluster_representative_conformation_info.pkl' % conformation_info_path
    with open(fname, 'rb') as f:
         conformation_info = pickle.load(f)
    for key in conformation_info:
        pdb_pred_path =  conformation_info[key][0] #this points to the original file
        mean_plddt = conformation_info[key][2]
        pdb_pred_path_named_by_cluster = conformation_info[key][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
        pdb_pred_path_list.append(pdb_pred_path)
        pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
        mean_plddt_list.append(mean_plddt)

    print('PDB FILES:')
    print(pdb_pred_path_list)
        
    rmsd_list = [] 
    dockq_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (pdb_ref_id, pdb_pred_path))
        rmsd, _, _ = superimpose_wrapper_multimer(pdb_ref_id, None, 'pdb', 'pred_rw', pdb_ref_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd_list.append(rmsd)

    out_df = pd.DataFrame({'pdb_id_reference_structure': pdb_ref_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list,
                          'mean_plddt': mean_plddt_list, 'rmsd': rmsd_list})
 
    return out_df



def save_metrics(pdb_ref_id, pdb_ref_path, conformation_info_path_list, pred_type):
    
    if pred_type == 'monomer':
        if len(pdb_ref_id.split('_')) != 2:
            raise ValueError("specify chain in pdb_ref_id for predicting monomer")
    if pred_type == 'multimer':
        if len(pdb_ref_id.split('_')) != 1:
            raise ValueError("do not specify chain in pdb_ref_id for predicting monomer")

    out_df_list = []

    for i in range(0,len(conformation_info_path_list)):
        conformation_info_path = conformation_info_path_list[i]
        superimposed_structures_save_dir = '%s/superimpose-%s' % (conformation_info_path, pdb_ref_id)
        if pred_type == 'monomer':
            out_df = compare_conformations_to_structure_monomer(pdb_ref_id, pdb_ref_path, conformation_info_path, superimposed_structures_save_dir)
        elif pred_type == 'multimer':
            out_df = compare_conformations_to_structure_multimer(pdb_ref_id, pdb_ref_path, conformation_info_path, superimposed_structures_save_dir)
        out_df_list.append(out_df)
        
    out_df = pd.concat(out_df_list, ignore_index=True)
    print(out_df)

    Path(superimposed_structures_save_dir).mkdir(parents=True, exist_ok=True)
    out_df.to_csv('%s/metrics.csv' % superimposed_structures_save_dir)

'''
pred_type = 'multimer'
home_dir = '/gpfs/home/itaneja/af_conformational_states_multimer/openfold_multimer_experimental'
module_config_str = 'module_config_0'
train_hp_config_str = 'train-hp_config_1'
rw_hp_config_str = 'rw-hp_config_1'
num_clusters = 10 

conformation_info_path_list = [] 
conformation_info_path = '%s/P69905-P69905-P69905-P69905/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=None' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_path_list.append(conformation_info_path)
conformation_info_path = '%s/P69905-P69905-P69905-P69905/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=0.7' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_path_list.append(conformation_info_path)



pdb_ref_id = '1qxe'
pdb_ref_path = None
save_metrics(pdb_ref_id, pdb_ref_path, conformation_info_path_list, pred_type)

pdb_ref_id = '2dn2'
pdb_ref_path = None
save_metrics(pdb_ref_id, pdb_ref_path, conformation_info_path_list, pred_type)'''


pred_type = 'multimer'
home_dir = '/gpfs/home/itaneja/af_conformational_states_multimer/openfold_multimer_experimental'
module_config_str = 'module_config_1'
train_hp_config_str = 'train-hp_config_1'
rw_hp_config_str = 'rw-hp_config_0'
num_clusters = 5


conformation_info_path_list = [] 
conformation_info_path = '%s/CNPase-Nb8d/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=None' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_path_list.append(conformation_info_path)


pdb_ref_id = 'H1140'
pdb_ref_path = './casp15/H1140/H1140_wallner.pdb'
save_metrics(pdb_ref_id, pdb_ref_path, conformation_info_path_list, pred_type)

