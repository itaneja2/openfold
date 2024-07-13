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

from custom_openfold_utils.pdb_utils import superimpose_wrapper_monomer, superimpose_wrapper_multimer, renumber_chain_wrt_reference

asterisk_line = '******************************************************************************'

def get_template_str(input_str):
    search_str = "template="
    start_index = input_str.find(search_str)
    template_str = input_str[start_index:]
    template_str = template_str[0:template_str.find("/")]
    return template_str 

def compare_clustered_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = [] 

    template_str = get_template_str(conformation_info_dir)

    fname = '%s/cluster_representative_conformation_info.pkl' % conformation_info_dir
    with open(fname, 'rb') as f:
         conformation_info = pickle.load(f)
    for key in conformation_info:
        pdb_pred_path =  conformation_info[key][0] #this points to the original file
        rmsd_wrt_initial = round(conformation_info[key][1],2)
        mean_plddt = round(conformation_info[key][2],2)
        pdb_pred_path_named_by_cluster = conformation_info[key][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
        pdb_pred_path_list.append(pdb_pred_path)
        pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
        rmsd_wrt_initial_list.append(rmsd_wrt_initial)
        mean_plddt_list.append(mean_plddt)

    print('PDB files being evaluated:')
    print(pdb_pred_path_list)
        
    rmsd_list = [] 
    tm_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        rmsd, tm_score, _, _ = superimpose_wrapper_monomer(ref_pdb_id, None, 'pdb', 'pred_rw', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        tm_score = round(tm_score,2)
        rmsd_list.append(rmsd)
        tm_list.append(tm_score)

    out_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_iderence_structure': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list,
                          'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'tm_score': tm_list})
 
    return out_df


def compare_clustered_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = []  
    ptm_iptm_list = [] 

    template_str = get_template_str(conformation_info_dir)

    fname = '%s/cluster_representative_conformation_info.pkl' % conformation_info_dir
    with open(fname, 'rb') as f:
         conformation_info = pickle.load(f)
    for key in conformation_info:
        pdb_pred_path =  conformation_info[key][0] #this points to the original file
        rmsd_wrt_initial = round(conformation_info[key][1],2)
        mean_plddt = round(conformation_info[key][2],2)
        ptm_iptm = round(conformation_info[key][3],2)
        pdb_pred_path_named_by_cluster = conformation_info[key][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
        pdb_pred_path_list.append(pdb_pred_path)
        pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
        rmsd_wrt_initial_list.append(rmsd_wrt_initial)
        mean_plddt_list.append(mean_plddt)
        ptm_iptm_list.append(ptm_iptm)

    print('PDB files being evaluated:')
    print(pdb_pred_path_list)
        
    rmsd_list = [] 
    dockq_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        rmsd, pdb_path_output, _ = superimpose_wrapper_multimer(ref_pdb_id, None, 'pdb', 'pred_rw', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        rmsd_list.append(rmsd)
        num_chains = renumber_chain_wrt_reference(pdb_path_output, pdb_pred_path) #arg1 is the pdb to renumber, arg2 is the pdb to renumber w.r.t to 
        if num_chains == 2:
            dockq_info = calc_DockQ(pdb_pred_path, pdb_path_output)
            dockq_val = round(dockq_info['DockQ'],2)
        else:
            dockq_val = np.nan 
        dockq_list.append(dockq_val)

    out_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_iderence_structure': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list,
                          'mean_plddt': mean_plddt_list, 'ptm_iptm': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'dockq': dockq_list})
 
    return out_df


def compare_all_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = [] 

    template_str = get_template_str(conformation_info_dir)

    pattern = "%s/conformation_info.pkl" % conformation_info_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        pattern = "%s/**/conformation_info.pkl" % conformation_info_dir
        files = glob.glob(pattern, recursive=True)
        if len(files) == 0:
            print('conformation_info file(s) do not exist in dir: %s' % conformation_info_dir)
            sys.exit()
    else:
        print('conformation_info file(s) found:')
        print(files)
    
    for fname in files: 
        print('opening %s' % fname)
        with open(fname, 'rb') as f:
             conformation_info = pickle.load(f)
        for key in conformation_info:
            curr_conformation_info = conformation_info[key]
            for i in range(0,len(curr_conformation_info)):
                pdb_pred_path = curr_conformation_info[i][0] #this points to the original file
                rmsd_wrt_initial = round(curr_conformation_info[i][1],2) 
                mean_plddt = round(curr_conformation_info[i][2],2)
                pdb_pred_path_list.append(pdb_pred_path)
                rmsd_wrt_initial_list.append(rmsd_wrt_initial)
                mean_plddt_list.append(mean_plddt)

    print('%d total PDB files being evaluated' % len(pdb_pred_path_list))
        
    rmsd_list = [] 
    tm_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        rmsd, tm_score, _, _ = superimpose_wrapper_monomer(ref_pdb_id, None, 'pdb', 'pred_benchmark', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        tm_score = round(tm_score,2)
        rmsd_list.append(rmsd)
        tm_list.append(tm_score)

    out_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_iderence_structure': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 
                            'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'tm_score': tm_list})
 
    return out_df



def compare_all_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = []  
    ptm_iptm_list = [] 

    template_str = get_template_str(conformation_info_dir)

    pattern = "%s/conformation_info.pkl" % conformation_info_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        pattern = "%s/**/conformation_info.pkl" % conformation_info_dir
        files = glob.glob(pattern, recursive=True)
        if len(files) == 0:
            print('conformation_info file(s) do not exist in dir: %s' % conformation_info_dir)
            sys.exit()
    else:
        print('conformation_info file(s) found:')
        print(files)


    for fname in files:
        print('opening %s' % fname)
        with open(fname, 'rb') as f:
             conformation_info = pickle.load(f) 
        for key in conformation_info:
            curr_conformation_info = conformation_info[key]
            for i in range(0,len(curr_conformation_info)):
                pdb_pred_path = curr_conformation_info[i][0] #this points to the original file
                rmsd_wrt_initial = round(curr_conformation_info[i][1],2) 
                mean_plddt = round(curr_conformation_info[i][2],2)
                ptm_iptm = round(conformation_info[i][3],2)
                pdb_pred_path_list.append(pdb_pred_path)
                rmsd_wrt_initial_list.append(rmsd_wrt_initial)
                mean_plddt_list.append(mean_plddt)
                ptm_iptm_list.append(ptm_iptm)

    print('%d total PDB files being evaluated' % len(pdb_pred_path_list))
    
    rmsd_list = [] 
    dockq_list = [] 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        rmsd, pdb_path_output, _ = superimpose_wrapper_multimer(ref_pdb_id, None, 'pdb', 'pred_benchmark', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        rmsd_list.append(rmsd)
        num_chains = renumber_chain_wrt_reference(pdb_path_output, pdb_pred_path) #arg1 is the pdb to renumber, arg2 is the pdb to renumber w.r.t to 
        if num_chains == 2:
            dockq_info = calc_DockQ(pdb_pred_path, pdb_path_output)
            dockq_val = round(dockq_info['DockQ'],2)
        else:
            dockq_val = np.nan 
        dockq_list.append(dockq_val)

    out_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_iderence_structure': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 
                          'mean_plddt': mean_plddt_list, 'ptm_iptm': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'dockq': dockq_list})
 
    return out_df





def get_clustered_conformations_metrics(uniprot_id, ref_pdb_id, conformation_info_dir, monomer_or_multimer, method_str, ref_pdb_path=None, save=True):
    
    if monomer_or_multimer == 'monomer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 2:
            raise ValueError("specify chain in ref_pdb_id for predicting monomer")
    if monomer_or_multimer == 'multimer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 1:
            raise ValueError("do not specify chain in ref_pdb_id for predicting multimer")

    superimposed_structures_save_dir = '%s/superimpose-%s' % (conformation_info_dir, ref_pdb_id)
    if monomer_or_multimer == 'monomer':
        out_df = compare_clustered_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
    elif monomer_or_multimer == 'multimer':
        out_df = compare_clustered_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
        
    print(out_df)

    if save:
        Path(superimposed_structures_save_dir).mkdir(parents=True, exist_ok=True)
        out_df.to_csv('%s/metrics.csv' % superimposed_structures_save_dir, index=False)

    return out_df


def get_all_conformations_metrics(uniprot_id, ref_pdb_id, conformation_info_dir, monomer_or_multimer, method_str, ref_pdb_path=None, save=True):
    
    if monomer_or_multimer == 'monomer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 2:
            raise ValueError("specify chain in ref_pdb_id for predicting monomer")
    if monomer_or_multimer == 'multimer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 1:
            raise ValueError("do not specify chain in ref_pdb_id for predicting multimer")

    superimposed_structures_save_dir = '%s/superimpose-%s' % (conformation_info_dir, ref_pdb_id)
    if monomer_or_multimer == 'monomer':
        out_df = compare_all_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
    elif monomer_or_multimer == 'multimer':
        out_df = compare_all_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
        
    print(out_df)

    if save:
        Path(superimposed_structures_save_dir).mkdir(parents=True, exist_ok=True)
        out_df.to_csv('%s/metrics.csv' % superimposed_structures_save_dir, index=False)

    return out_df 




'''
monomer_or_multimer = 'multimer'
home_dir = '/gpfs/home/itaneja/af_conformational_states_multimer/openfold_multimer_experimental'
module_config_str = 'module_config_0'
train_hp_config_str = 'train-hp_config_1'
rw_hp_config_str = 'rw-hp_config_1'
num_clusters = 10 

conformation_info_dir_list = [] 
conformation_info_dir = '%s/P69905-P69905-P69905-P69905/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=None' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_dir_list.append(conformation_info_dir)
conformation_info_dir = '%s/P69905-P69905-P69905-P69905/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=0.7' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_dir_list.append(conformation_info_dir)



ref_pdb_id = '1qxe'
pdb_path = None
save_metrics(ref_pdb_id, pdb_path, conformation_info_dir_list, monomer_or_multimer)

ref_pdb_id = '2dn2'
pdb_path = None
save_metrics(ref_pdb_id, pdb_path, conformation_info_dir_list, monomer_or_multimer)'''


'''monomer_or_multimer = 'multimer'
home_dir = '/gpfs/home/itaneja/af_conformational_states_multimer/openfold_multimer_experimental'
module_config_str = 'module_config_1'
train_hp_config_str = 'train-hp_config_1'
rw_hp_config_str = 'rw-hp_config_0'
num_clusters = 5


conformation_info_dir_list = [] 
conformation_info_dir = '%s/CNPase-Nb8d/rw_v4/%s/%s/%s/rw/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=None' % (home_dir, module_config_str,train_hp_config_str,rw_hp_config_str, num_clusters)
conformation_info_dir_list.append(conformation_info_dir)


ref_pdb_id = 'H1140'
pdb_path = './casp15/H1140/H1140_wallner.pdb'
save_metrics(ref_pdb_id, pdb_path, conformation_info_dir_list, monomer_or_multimer)


print(asterisk_line)

monomer_or_multimer = 'monomer'
home_dir = '/gpfs/home/itaneja/openfold'
num_clusters = 10 


uniprot_id = 'P69441'
rw_conformation_info_dir = '%s/%s/rw/module_config_0/train-hp_config_1/rw-hp_config_0-0/rw/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (home_dir, uniprot_id, num_clusters)
benchmark_conformation_info_dir = '%s/%s/benchmark' % (home_dir, uniprot_id)

ref_pdb_id = '1ake_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)

ref_pdb_id = '4ake_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)

print(asterisk_line)

uniprot_id = 'P00533'
rw_conformation_info_dir = '%s/%s/rw/module_config_0/train-hp_config_1/rw-hp_config_0-0/rw/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (home_dir, uniprot_id, num_clusters)
benchmark_conformation_info_dir = '%s/%s/benchmark' % (home_dir, uniprot_id)

ref_pdb_id = '2itp_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)


ref_pdb_id = '2gs7_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)

print(asterisk_line)


uniprot_id = 'Q9BYF1'
rw_conformation_info_dir = '%s/%s/rw/module_config_0/train-hp_config_1/rw-hp_config_0-0/rw/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (home_dir, uniprot_id, num_clusters)
benchmark_conformation_info_dir = '%s/%s/benchmark' % (home_dir, uniprot_id)

ref_pdb_id = '1r42_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)


ref_pdb_id = '1r4l_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)

print(asterisk_line)

uniprot_id = 'P42866'
rw_conformation_info_dir = '%s/%s/rw/module_config_0/train-hp_config_1/rw-hp_config_0-0/rw/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (home_dir, uniprot_id, num_clusters)
benchmark_conformation_info_dir = '%s/%s/benchmark' % (home_dir, uniprot_id)

ref_pdb_id = '5c1m_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)


ref_pdb_id = '4dkl_A'
pdb_path = None
save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)

print(asterisk_line)

ref_pdb_id = None
pdb_path = './casp15_targets/T1123-D1.pdb'
uniprot_id = 'Q82452'
rw_conformation_info_dir = '%s/%s/rw/module_config_0/train-hp_config_1/rw-hp_config_0-0/rw/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (home_dir, uniprot_id, num_clusters)
benchmark_conformation_info_dir = '%s/%s/benchmark' % (home_dir, uniprot_id)

save_rw_metrics(uniprot_id, ref_pdb_id, pdb_path, rw_conformation_info_dir, monomer_or_multimer)
save_benchmark_metrics(uniprot_id, ref_pdb_id, pdb_path, benchmark_conformation_info_dir, monomer_or_multimer)
'''
