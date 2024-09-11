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
import re 

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO

import collections
import functools

from DockQ.DockQ import calc_DockQ

from custom_openfold_utils.pdb_utils import superimpose_wrapper_monomer, superimpose_wrapper_multimer, renumber_chain_wrt_reference, convert_pdb_to_mmcif, get_bfactor, get_pdb_path_seq, get_af_disordered_domains, get_af_disordered_residues, get_ca_pairwise_dist
from custom_openfold_utils.conformation_utils import get_rmsf_pdb_af_conformation, get_comparison_residues_idx_between_pdb_af_conformation

from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

asterisk_line = '******************************************************************************'

def get_template_str(input_str):
    search_str = "template="
    start_index = input_str.find(search_str)
    template_str = input_str[start_index:]
    template_str = template_str[0:template_str.find("/")]
    return template_str 

def get_cluster_num(cluster_str):
    cluster_num = cluster_str.split('/')[-1].split('.')[0]
    if 'cluster' in cluster_num:
        pattern = r'cluster_\d+'
        match = re.search(pattern, cluster_num)
        if match:
            cluster_num = match.group()
        else:
            cluster_num = 'none'
    elif 'initial' in cluster_num:
        cluster_num = 'initial_pred'
    return cluster_num

def compare_clustered_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = [] 

    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    pdb_pred_path_list.append(initial_pred_path)
    pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
    rmsd_wrt_initial_list.append(0)
    mean_plddt_list.append(mean_plddt)

    template_str = get_template_str(conformation_info_dir)

    fname = '%s/md_starting_structure_info.pkl' % conformation_info_dir
    with open(fname, 'rb') as f:
         conformation_info = pickle.load(f)
    for cluster_num in sorted(conformation_info.keys()):
        for cluster_idx in range(0,len(conformation_info[cluster_num])):
            pdb_pred_path = conformation_info[cluster_num][cluster_idx][0] #this points to the original file
            rmsd_wrt_initial = round(conformation_info[cluster_num][cluster_idx][1],2)
            mean_plddt = round(conformation_info[cluster_num][cluster_idx][2],2)
            pdb_pred_path_named_by_cluster = conformation_info[cluster_num][cluster_idx][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
            pdb_pred_path_list.append(pdb_pred_path)
            pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
            rmsd_wrt_initial_list.append(rmsd_wrt_initial)
            mean_plddt_list.append(mean_plddt)

    print('PDB files being evaluated:')
    print(pdb_pred_path_list[0:5])

    cluster_num_list = []    
    rmsd_list = [] 
    tm_list = []
    rmsf_df = []
    ca_pdist_all = [] 
 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):

        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))

        cluster_num = get_cluster_num(pdb_pred_path_named_by_cluster_list[i])
        cluster_num_list.append(cluster_num)

        rmsd, tm_score, ref_pdb_path_aligned, pdb_pred_path_aligned = superimpose_wrapper_monomer(ref_pdb_id, None, 'pdb', 'pred_rw', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        tm_score = round(tm_score,2)
        rmsd_list.append(rmsd)
        tm_list.append(tm_score)

        ca_pdist = get_ca_pairwise_dist(pdb_pred_path_aligned)
        if len(ca_pdist_all) == 0:
            ca_pdist_all = np.array(ca_pdist)
        else:
            ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))

        curr_rmsf_df = get_rmsf_pdb_af_conformation(ref_pdb_path_aligned, pdb_pred_path_aligned)
        curr_rmsf_df['uniprot_id'] = uniprot_id
        curr_rmsf_df['method'] = method_str
        curr_rmsf_df['template'] = template_str 
        curr_rmsf_df['ref_pdb_id'] = ref_pdb_id
        curr_rmsf_df['pdb_pred_path'] = pdb_pred_path_aligned
        curr_rmsf_df['cluster_num'] = cluster_num

        if len(rmsf_df) == 0:
            rmsf_df = curr_rmsf_df
        else:
            rmsf_df = pd.concat([rmsf_df, curr_rmsf_df], axis=0, ignore_index=True)
 
    summary_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_id': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list, 'cluster_num': cluster_num_list,
                          'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'tm_score': tm_list})

    print('Running PCA')
    #pdb_pred_path_list.append(ref_pdb_path_aligned_initial)
    #cluster_num_list.append('reference')
    #ca_pdist = get_ca_pairwise_dist(ref_pdb_path_aligned_initial)
    #ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))
    #print(ca_pdist_all)
    #print(ca_pdist_all.shape)
    scaler = StandardScaler()
    ca_pdist_all_scaled = scaler.fit_transform(ca_pdist_all)
    pca = PCA(n_components=2)
    pc_coords = pca.fit_transform(ca_pdist_all_scaled)

    pca_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_id': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 'cluster_num': cluster_num_list, 'pc1': pc_coords[:,0], 'pc2': pc_coords[:,1]})

    print(pca_df)

 
    return summary_df, rmsf_df, pca_df


def compare_clustered_conformations_pca(uniprot_id, ref_pdb_id, rw_initial_pred_path, benchmark_initial_pred_path, rw_conformation_info_dir, benchmark_conformation_info_dir):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = [] 
    method_list = []

    try: 
        af_seq = get_pdb_path_seq(rw_initial_pred_path, None)
        af_disordered_domains_idx, _ = get_af_disordered_domains(rw_initial_pred_path)     
        af_disordered_residues_idx = get_af_disordered_residues(af_disordered_domains_idx)
        af_disordered_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_disordered_residues_idx)))
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % initial_pred_path) 
        af_disordered_residues_idx_complement = None

    print('Index of residues that are not disordered based on initial AF prediction')
    print(af_disordered_residues_idx_complement) 
 
    for i,initial_pred_path in enumerate([rw_initial_pred_path, benchmark_initial_pred_path]):
        cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
        _, mean_plddt = get_bfactor(cif_path)
        mean_plddt = round(mean_plddt,2)
        os.remove(cif_path)
        pdb_pred_path_list.append(initial_pred_path)
        pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
        rmsd_wrt_initial_list.append(0)
        mean_plddt_list.append(mean_plddt)
        if i == 0:
            method_list.append('initial-rw')
        else:
            method_list.append('initial-benchmark')

    template_str = get_template_str(rw_conformation_info_dir)

    fname = '%s/md_starting_structure_info.pkl' % rw_conformation_info_dir
    with open(fname, 'rb') as f:
         rw_conformation_info = pickle.load(f)
    fname = '%s/md_starting_structure_info.pkl' % benchmark_conformation_info_dir
    with open(fname, 'rb') as f:
         benchmark_conformation_info = pickle.load(f)

    for i,conformation_info in enumerate([rw_conformation_info, benchmark_conformation_info]):
        for cluster_num in sorted(conformation_info.keys()):
            for cluster_idx in range(0,len(conformation_info[cluster_num])):
                pdb_pred_path = conformation_info[cluster_num][cluster_idx][0] #this points to the original file
                rmsd_wrt_initial = round(conformation_info[cluster_num][cluster_idx][1],2)
                mean_plddt = round(conformation_info[cluster_num][cluster_idx][2],2)
                pdb_pred_path_named_by_cluster = conformation_info[cluster_num][cluster_idx][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
                pdb_pred_path_list.append(pdb_pred_path)
                pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
                rmsd_wrt_initial_list.append(rmsd_wrt_initial)
                mean_plddt_list.append(mean_plddt)
                if i == 0:
                    method_list.append('rw')
                else:
                    method_list.append('benchmark')

    cluster_num_list = []    
    ca_pdist_all = []
 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):

        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))

        cluster_num = get_cluster_num(pdb_pred_path_named_by_cluster_list[i])
        cluster_num_list.append(cluster_num)

        ca_pdist = get_ca_pairwise_dist(pdb_pred_path, residues_include_idx=af_disordered_residues_idx_complement)
        if len(ca_pdist_all) == 0:
            ca_pdist_all = np.array(ca_pdist)
        else:
            ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))

    print('Running PCA')
    #pdb_pred_path_list.append(ref_pdb_path_aligned_initial)
    #cluster_num_list.append('reference')
    #ca_pdist = get_ca_pairwise_dist(ref_pdb_path_aligned_initial)
    #ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))
    #print(ca_pdist_all)
    #print(ca_pdist_all.shape)
    scaler = StandardScaler()
    ca_pdist_all_scaled = scaler.fit_transform(ca_pdist_all)
    pca = PCA(n_components=2)
    pc_coords = pca.fit_transform(ca_pdist_all_scaled)

    pca_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_list, 'template': template_str, 'pdb_pred_path': pdb_pred_path_list, 'cluster_num': cluster_num_list, 'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'pc1': pc_coords[:,0], 'pc2': pc_coords[:,1]})

    print(pca_df)

 
    return pca_df




def compare_clustered_conformations_pca_w_ref(uniprot_id, ref_pdb_id, initial_pred_path, rw_conformation_info_dir, benchmark_conformation_info_dir):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = [] 
    method_list = [] 

    superimposed_structures_save_dir = '%s/superimpose-%s/pdb_ref_structure' % (rw_conformation_info_dir, ref_pdb_id)
    ref_pdb_path = '%s/%s_clean.pdb' % (superimposed_structures_save_dir, ref_pdb_id)
    ref_pdb_comparison_residues_idx, af_comparison_residues_idx = get_comparison_residues_idx_between_pdb_af_conformation(ref_pdb_path, initial_pred_path)

    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    pdb_pred_path_list.append(initial_pred_path)
    pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
    rmsd_wrt_initial_list.append(0)
    mean_plddt_list.append(mean_plddt)
    method_list.append('initial')

    template_str = get_template_str(rw_conformation_info_dir)

    fname = '%s/md_starting_structure_info.pkl' % rw_conformation_info_dir
    with open(fname, 'rb') as f:
         rw_conformation_info = pickle.load(f)
    fname = '%s/md_starting_structure_info.pkl' % benchmark_conformation_info_dir
    with open(fname, 'rb') as f:
         benchmark_conformation_info = pickle.load(f)

    for i,conformation_info in enumerate([rw_conformation_info, benchmark_conformation_info]):
        for cluster_num in sorted(conformation_info.keys()):
            for cluster_idx in range(0,len(conformation_info[cluster_num])):
                pdb_pred_path = conformation_info[cluster_num][cluster_idx][0] #this points to the original file
                rmsd_wrt_initial = round(conformation_info[cluster_num][cluster_idx][1],2)
                mean_plddt = round(conformation_info[cluster_num][cluster_idx][2],2)
                pdb_pred_path_named_by_cluster = conformation_info[cluster_num][cluster_idx][-1] #this is the same structure as pdb_pred_path except named by the cluster it belongs to  
                pdb_pred_path_list.append(pdb_pred_path)
                pdb_pred_path_named_by_cluster_list.append(pdb_pred_path_named_by_cluster)
                rmsd_wrt_initial_list.append(rmsd_wrt_initial)
                mean_plddt_list.append(mean_plddt)
                if i == 0:
                    method_list.append('rw')
                else:
                    method_list.append('benchmark')

    cluster_num_list = []    
    ca_pdist_all = []
 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):

        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))

        cluster_num = get_cluster_num(pdb_pred_path_named_by_cluster_list[i])
        cluster_num_list.append(cluster_num)

        ca_pdist = get_ca_pairwise_dist(pdb_pred_path, residues_include_idx=af_comparison_residues_idx)
        if len(ca_pdist_all) == 0:
            ca_pdist_all = np.array(ca_pdist)
        else:
            ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))

    ref_pdb_ca_pdist = get_ca_pairwise_dist(ref_pdb_path, ref_pdb_comparison_residues_idx)
    ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))
    method_list.append('reference')
    pdb_pred_path_list.append(ref_pdb_path)
    cluster_num_list.append('reference')
    mean_plddt_list.append(-1)
    rmsd_wrt_initial_list.append(0)

    print('Running PCA')
    #pdb_pred_path_list.append(ref_pdb_path_aligned_initial)
    #cluster_num_list.append('reference')
    #ca_pdist = get_ca_pairwise_dist(ref_pdb_path_aligned_initial)
    #ca_pdist_all = np.vstack((ca_pdist_all, np.array(ca_pdist)))
    #print(ca_pdist_all)
    #print(ca_pdist_all.shape)
    scaler = StandardScaler()
    ca_pdist_all_scaled = scaler.fit_transform(ca_pdist_all)
    pca = PCA(n_components=2)
    pc_coords = pca.fit_transform(ca_pdist_all_scaled)

    pca_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_list, 'template': template_str, 'pdb_pred_path': pdb_pred_path_list, 'cluster_num': cluster_num_list, 'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'pc1': pc_coords[:,0], 'pc2': pc_coords[:,1]})

    print(pca_df)

 
    return pca_df



def compare_clustered_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    pdb_pred_path_named_by_cluster_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = []  
    ptm_iptm_list = [] 

    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    pdb_pred_path_list.append(initial_pred_path)
    pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
    rmsd_wrt_initial_list.append(0)
    mean_plddt_list.append(mean_plddt)
    ptm_iptm_list.append(0)


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
    cluster_num_list = [] 
 
    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        cluster_num = get_cluster_num(pdb_pred_path_named_by_cluster_list[i])
        cluster_num_list.append(cluster_num)
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

    summary_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_id': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 'pdb_pred_path_named_by_cluster': pdb_pred_path_named_by_cluster_list, 'cluster_num': cluster_num_list,
                          'mean_plddt': mean_plddt_list, 'ptm_iptm': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'dockq': dockq_list})
 
    return summary_df


def compare_all_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = []

    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    pdb_pred_path_list.append(initial_pred_path)
    pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
    rmsd_wrt_initial_list.append(0)
    mean_plddt_list.append(mean_plddt)
 
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
    rmsf_df = []  

    for i,pdb_pred_path in enumerate(pdb_pred_path_list):
        print('completion percentage: %.2f' % (i/len(pdb_pred_path_list)))
        print('superimposing %s and %s' % (ref_pdb_id, pdb_pred_path))
        rmsd, tm_score, ref_pdb_path_aligned, pdb_pred_path_aligned = superimpose_wrapper_monomer(ref_pdb_id, None, 'pdb', 'pred_benchmark', ref_pdb_path, pdb_pred_path, superimposed_structures_save_dir)
        rmsd = round(rmsd,2)
        tm_score = round(tm_score,2)
        rmsd_list.append(rmsd)
        tm_list.append(tm_score)

        curr_rmsf_df = get_rmsf_pdb_af_conformation(ref_pdb_path_aligned, pdb_pred_path_aligned)
        curr_rmsf_df['uniprot_id'] = uniprot_id
        curr_rmsf_df['method'] = method_str
        curr_rmsf_df['template'] = template_str 
        curr_rmsf_df['ref_pdb_id'] = ref_pdb_id
        curr_rmsf_df['pdb_pred_path'] = pdb_pred_path_aligned
        curr_rmsf_df['pdb_cluster'] = get_cluster_num(pdb_pred_path_named_by_cluster_list[i]) 

        if len(rmsf_df) == 0:
            rmsf_df = curr_rmsf_df
        else:
            rmsf_df = pd.concat([rmsf_df, curr_rmsf_df], axis=0, ignore_index=True)


    summary_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_id': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 
                            'mean_plddt': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'tm_score': tm_list})
 
    return summary_df, rmsf_df



def compare_all_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str):

    pdb_pred_path_list = []
    rmsd_wrt_initial_list = [] 
    mean_plddt_list = []  
    ptm_iptm_list = []

    cif_path = convert_pdb_to_mmcif(initial_pred_path, './cif_temp')
    _, mean_plddt = get_bfactor(cif_path)
    mean_plddt = round(mean_plddt,2)
    os.remove(cif_path)
    pdb_pred_path_list.append(initial_pred_path)
    pdb_pred_path_named_by_cluster_list.append(initial_pred_path)
    rmsd_wrt_initial_list.append(0)
    mean_plddt_list.append(mean_plddt)
    ptm_iptm_list.append(0)

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

    summary_df = pd.DataFrame({'uniprot_id': uniprot_id, 'method': method_str, 'template': template_str, 'ref_pdb_id': ref_pdb_id, 'pdb_pred_path': pdb_pred_path_list, 
                          'mean_plddt': mean_plddt_list, 'ptm_iptm': mean_plddt_list, 'rmsd_wrt_initial': rmsd_wrt_initial_list, 'rmsd': rmsd_list, 'dockq': dockq_list})
 
    return summary_df





def get_clustered_conformations_metrics(uniprot_id, ref_pdb_id, conformation_info_dir, initial_pred_path, monomer_or_multimer, method_str, ref_pdb_path=None, save=True):
    
    if monomer_or_multimer == 'monomer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 2:
            raise ValueError("specify chain in ref_pdb_id for predicting monomer")
    if monomer_or_multimer == 'multimer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 1:
            raise ValueError("do not specify chain in ref_pdb_id for predicting multimer")

    superimposed_structures_save_dir = '%s/superimpose-%s' % (conformation_info_dir, ref_pdb_id)
    if monomer_or_multimer == 'monomer':
        summary_df, rmsf_df, pca_df = compare_clustered_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
    elif monomer_or_multimer == 'multimer':
        summary_df = compare_clustered_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, initial_pred_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
        
    print(summary_df)

    if save:
        Path(superimposed_structures_save_dir).mkdir(parents=True, exist_ok=True)
        summary_df.to_csv('%s/metrics.csv' % superimposed_structures_save_dir, index=False)
        if monomer_or_multimer == 'monomer':
            print(rmsf_df)
            rmsf_df.to_csv('%s/rmsf_metrics.csv' % superimposed_structures_save_dir, index=False)
            pca_df.to_csv('%s/pca_metrics.csv' % superimposed_structures_save_dir, index=False)


    return summary_df


def get_all_conformations_metrics(uniprot_id, ref_pdb_id, conformation_info_dir, monomer_or_multimer, method_str, ref_pdb_path=None, save=True):
    
    if monomer_or_multimer == 'monomer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 2:
            raise ValueError("specify chain in ref_pdb_id for predicting monomer")
    if monomer_or_multimer == 'multimer':
        if ref_pdb_id is not None and len(ref_pdb_id.split('_')) != 1:
            raise ValueError("do not specify chain in ref_pdb_id for predicting multimer")

    superimposed_structures_save_dir = '%s/superimpose-%s' % (conformation_info_dir, ref_pdb_id)
    if monomer_or_multimer == 'monomer':
        summary_df = compare_all_conformations_to_reference_monomer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
    elif monomer_or_multimer == 'multimer':
        summary_df = compare_all_conformations_to_reference_multimer(uniprot_id, ref_pdb_id, ref_pdb_path, conformation_info_dir, superimposed_structures_save_dir, method_str)
        
    print(summary_df)

    if save:
        Path(superimposed_structures_save_dir).mkdir(parents=True, exist_ok=True)
        summary_df.to_csv('%s/metrics.csv' % superimposed_structures_save_dir, index=False)
        if monomer_or_multimer == 'monomer':
            rmsf_df.to_csv('%s/rmsf_metrics.csv' % superimposed_structures_save_dir, index=False)


    return summary_df 




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
