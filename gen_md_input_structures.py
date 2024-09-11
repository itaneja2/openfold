import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO
import glob
import pandas as pd
import os
import pickle
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import collections
from collections import defaultdict
import sys
import shutil 
from scipy.spatial.distance import pdist
from pre_md_refine import refine_multiple_rounds 
import json 

from custom_openfold_utils.pdb_utils import get_pdb_path_seq, get_af_disordered_domains, get_af_disordered_residues, get_ca_pairwise_dist

asterisk_line = '******************************************************************************'

def remove_files(file_list):
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)


def create_clustering_dist_matrix(ca_pdist_all):

    #method derived from
    #https://www.biorxiv.org/content/10.1101/2023.07.13.545008v2.supplementary-material

    clustering_dist_matrix = np.zeros((ca_pdist_all.shape[0],ca_pdist_all.shape[0]))
    for i in range(0,ca_pdist_all.shape[0]):
        for j in range(i+1,ca_pdist_all.shape[0]):
            pairwise_dist_diff = np.abs(ca_pdist_all[j,:] - ca_pdist_all[i,:])
            pairwise_dist_diff[pairwise_dist_diff <= 1] = 0.
            pairwise_dist_diff = np.sum(pairwise_dist_diff)
            clustering_dist_matrix[i,j] = pairwise_dist_diff
    
    return clustering_dist_matrix
           

def gen_md_input(conformation_info_dir, initial_pred_path, num_clusters, num_md_structures, plddt_threshold, overwrite):

    if plddt_threshold is None:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=None' % (conformation_info_dir, num_clusters)
    else:
        md_structures_dir = '%s/md_starting_structures/num_clusters=%d/plddt_threshold=%s' % (conformation_info_dir, num_clusters, str(plddt_threshold))

    os.makedirs(md_structures_dir, exist_ok=True)

    try: 
        af_seq = get_pdb_path_seq(initial_pred_path, None)
        af_disordered_domains_idx, _ = get_af_disordered_domains(initial_pred_path)     
        af_disordered_residues_idx = get_af_disordered_residues(af_disordered_domains_idx)
        af_disordered_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_disordered_residues_idx)))
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % initial_pred_path) 
        af_disordered_residues_idx_complement = None 

    md_starting_structure_info_fname = '%s/md_starting_structure_info.pkl' % md_structures_dir
    pdb_files = glob.glob('%s/*.pdb' % md_structures_dir) 
    if os.path.exists(md_starting_structure_info_fname):
        if not(overwrite):
            print("SKIPPING CLUSTERING PROCEDURE, %s ALREADY EXISTS:" % md_starting_structure_info_fname)
            sys.exit(0)
        else:
            remove_files(pdb_files)
    else:
        remove_files(pdb_files)


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


    #combine into single dictionary
    for i in range(0,len(files)):
        with open(files[i], 'rb') as f:
            curr_conformation_info = pickle.load(f)
        if i == 0:
            conformation_info = curr_conformation_info.copy()
        else:
            conformation_info.update(curr_conformation_info)

    conformation_info_all = [] 
    ca_pdist_all = [] 
    for key in conformation_info:
        for i,val in enumerate(conformation_info[key]):
            pdb_path = val[0]
            rmsd = val[1]
            mean_plddt = val[2]

            if plddt_threshold is None:
                conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_residues_idx_complement)
                ca_pdist_all.append(ca_pdist)
            else:
                if mean_plddt >= plddt_threshold:
                    conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                    ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_residues_idx_complement)
                    ca_pdist_all.append(ca_pdist)
        
    ca_pdist_all = np.array(ca_pdist_all) 
    clustering_dist_matrix = create_clustering_dist_matrix(ca_pdist_all)
 
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage="average").fit(clustering_dist_matrix)

    print(asterisk_line)
    print('CLUSTER FREQUENCIES:')
    counter = collections.Counter(clustering.labels_)
    print(counter.most_common())
    print(asterisk_line)

    clustering_info_dict = {} 
    for idx,cluster_num in enumerate(clustering.labels_):
        if cluster_num in clustering_info_dict:
            clustering_info_dict[cluster_num].append(conformation_info_all[idx])
        else:
            clustering_info_dict[cluster_num] = [conformation_info_all[idx]] 

    for cluster_num in clustering_info_dict:
        clustering_info_dict[cluster_num] = sorted(clustering_info_dict[cluster_num], key=lambda x: x[2], reverse=True) #sort by plddt 

    cluster_idx_dict = {}
    for cluster_num,freq in counter.most_common():
        cluster_idx_dict[cluster_num] = 0 

    md_starting_structure_info_dict = defaultdict(list)
    curr_md_structures_count = 0
    
    while curr_md_structures_count < num_md_structures: 
        for cluster_num,freq in counter.most_common():
            curr_cluster_idx = cluster_idx_dict[cluster_num] 
            if len(clustering_info_dict[cluster_num]) > curr_cluster_idx and curr_md_structures_count < num_md_structures:
                md_starting_structure_info_dict[cluster_num].append(clustering_info_dict[cluster_num][curr_cluster_idx])
                cluster_idx_dict[cluster_num] += 1 
                curr_md_structures_count += 1
                print("Adding structure from cluster %d, index %d" % (cluster_num, cluster_idx_dict[cluster_num]-1)) 

    print(asterisk_line)
    print('MD STARTING STRUCTURE INFO:')
    for key, value in md_starting_structure_info_dict.items():
        print(f"Cluster number: {key}, Number of elements: {len(value)}")
    print(asterisk_line)

    structural_issues_info_summary = {} 

    for cluster_num in md_starting_structure_info_dict:
        for i in range(0,len(md_starting_structure_info_dict[cluster_num])):
            pdb_source_path = md_starting_structure_info_dict[cluster_num][i][0]
            plddt = md_starting_structure_info_dict[cluster_num][i][2]
            output_fname = 'cluster_%d_idx_%d_plddt_%s' % (cluster_num, i, str(round(plddt)))
            pdb_target_path = '%s/%s.pdb' % (md_structures_dir, output_fname)
            shutil.copyfile(pdb_source_path, pdb_target_path)
            
            md_unrefined_structure_path = os.path.abspath(pdb_target_path)
            md_refined_structure_path = md_unrefined_structure_path.replace('.pdb', '_openmm_refinement.pdb')
            print('****************************************')
            print('****************************************')
            print("ATTEMPTING TO RESOLVE ANY STRUCTURAL ISSUES WITH %s" % md_unrefined_structure_path)
            print('****************************************')
            print('****************************************')
            structural_issues_info = refine_multiple_rounds(md_unrefined_structure_path, md_refined_structure_path)
            md_starting_structure_info_dict[cluster_num][i].append(md_refined_structure_path)
            structural_issues_info_summary[md_refined_structure_path] = structural_issues_info

    print(structural_issues_info_summary)

    output_file = '%s/structural_issues_info_summary.json' % md_structures_dir
    print('saving %s' % output_file)
    with open(output_file, 'w') as json_file:
        json.dump(structural_issues_info_summary, json_file, sort_keys=True, indent=4)

    md_starting_structure_info_fname = '%s/md_starting_structure_info.pkl' % md_structures_dir
    with open(md_starting_structure_info_fname, 'wb') as f:
        pickle.dump(md_starting_structure_info_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conformation_info_dir", type=str, default=None,
        help="Either points directly to the parent directory containing conformation_info.pkl or the grandparent directory if multiple conformation_info.pkl are present",
    )
    parser.add_argument(
        "--initial_pred_path", type=str, default=None,
        help="Points directly to the vanila AF prediction",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10,
        help="",
    )
    parser.add_argument(
        "--num_md_structures", type=int, default=50,
        help="number of structures to use as starting points for MD"
    )
    parser.add_argument(
        "--plddt_threshold", type=float, default=None,
        help="only clusters conformations whose plddt score is greater than this threshold (which is a value between 0-100)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False
    )

    args = parser.parse_args()

    gen_md_input(args.conformation_info_dir, args.initial_pred_path, args.num_clusters, args.num_md_structures, args.plddt_threshold, args.overwrite)


