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
import sys
import shutil 
from scipy.spatial.distance import pdist
import ml_collections as mlc
import torch 

from openfold.utils.script_utils import relax_protein
from openfold.np.protein import from_pdb_string 

from custom_openfold_utils.pdb_utils import get_pdb_path_seq, get_af_disordered_domains, get_af_disordered_residues, get_ca_pairwise_dist

asterisk_line = '******************************************************************************'

def remove_files(file_list):
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)


def create_clustering_dist_matrix(ca_pdist_all):

    #method derived from
    #https://www.biorxiv.org/content/10.1101/2023.07.13.545008v2.supplementary-material (pg. 3)

    clustering_dist_matrix = np.zeros((ca_pdist_all.shape[0],ca_pdist_all.shape[0]))
    for i in range(0,ca_pdist_all.shape[0]):
        for j in range(i+1,ca_pdist_all.shape[0]):
            pairwise_dist_diff = np.abs(ca_pdist_all[j,:] - ca_pdist_all[i,:])
            pairwise_dist_diff[pairwise_dist_diff <= 1] = 0.
            pairwise_dist_diff = np.sum(pairwise_dist_diff)
            clustering_dist_matrix[i,j] = pairwise_dist_diff
    
    return clustering_dist_matrix
           
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
        "--plddt_threshold", type=float, default=None,
        help="only clusters conformations whose plddt score is greater than this threshold (which is a value between 0-100)",
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False
    )

    args = parser.parse_args()

    config = mlc.ConfigDict(
        {
            "relax": {
                "max_iterations": 0,  # no max
                "tolerance": 2.39,
                "stiffness": 10.0,
                "max_outer_iterations": 20,
                "exclude_residues": [],
            }
        }
    )

    if args.plddt_threshold is None:
        cluster_dir = '%s/cluster_representative_structures/num_clusters=%d/plddt_threshold=None' % (args.conformation_info_dir, args.num_clusters)
    else:
        cluster_dir = '%s/cluster_representative_structures/num_clusters=%d/plddt_threshold=%s' % (args.conformation_info_dir, args.num_clusters, str(args.plddt_threshold))

    os.makedirs(cluster_dir, exist_ok=True)

    cluster_representative_conformation_info_fname = '%s/cluster_representative_conformation_info.pkl' % cluster_dir
    pdb_files = glob.glob('%s/*.pdb' % cluster_dir) 
    if os.path.exists(cluster_representative_conformation_info_fname):
        if not(args.overwrite):
            print("SKIPPING CLUSTERING PROCEDURE, %s ALREADY EXISTS:" % cluster_representative_conformation_info_fname)
            sys.exit(0)
        else:
            remove_files(pdb_files)
    else:
        remove_files(pdb_files)

    pattern = "%s/conformation_info.pkl" % args.conformation_info_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        pattern = "%s/**/conformation_info.pkl" % args.conformation_info_dir
        files = glob.glob(pattern, recursive=True)
        if len(files) == 0:
            print('conformation_info file(s) do not exist in dir: %s' % args.conformation_info_dir)
            sys.exit()
    else:
        print('conformation_info file(s) found:')
        print(files)

    try: 
        af_seq = get_pdb_path_seq(initial_pred_path, None)
        af_disordered_domains_idx, _ = get_af_disordered_domains(initial_pred_path)     
        nterm_disorder_start_idx, nterm_disorder_end_idx = get_af_nterm_disorder_tail_idx(af_disordered_domains_idx)
        cterm_disorder_start_idx, cterm_disorder_end_idx = get_af_cterm_disorder_tail_idx(af_disordered_domains_idx, len(af_seq))
        print('nterminal disordered tail idx: %s-%s' % (str(nterm_disorder_start_idx),str(nterm_disorder_end_idx)))
        print('cterminal disordered tail idx: %s-%s' % (str(cterm_disorder_start_idx),str(cterm_disorder_end_idx)))
        af_disordered_tail_residues_idx = get_af_disordered_tail_residues(af_disordered_domains_idx, len(af_seq))
        print('disordered tail residues idx:')
        print(af_disordered_tail_residues_idx)
        af_disordered_tail_residues_idx_complement = sorted(list(set(range(len(af_seq))) - set(af_disordered_tail_residues_idx)))
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION %s' % initial_pred_path) 
        af_disordered_tail_residues_idx_complement = None 


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
            if args.plddt_threshold is None:
                conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                ca_pdist_all.append(ca_pdist)
            else:
                if mean_plddt >= args.plddt_threshold:
                    conformation_info_all.append([pdb_path,rmsd,mean_plddt])
                    ca_pdist = get_ca_pairwise_dist(pdb_path, residues_include_idx=af_disordered_tail_residues_idx_complement)
                    ca_pdist_all.append(ca_pdist)
        
    ca_pdist_all = np.array(ca_pdist_all) 
    clustering_dist_matrix = create_clustering_dist_matrix(ca_pdist_all)
 
    clustering = AgglomerativeClustering(n_clusters=args.num_clusters, metric="precomputed", linkage="average").fit(clustering_dist_matrix)

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

    cluster_representative_conformation_info_dict = {} 
    for cluster_num,freq in counter.most_common():
        num_structures_in_cluster = len(clustering_info_dict[cluster_num])
        halfway_idx = num_structures_in_cluster//2
        cluster_representative_conformation_info_dict[cluster_num] = clustering_info_dict[cluster_num][halfway_idx] 

    print(cluster_representative_conformation_info_dict)

    for cluster_num in cluster_representative_conformation_info_dict:
        pdb_source_path = cluster_representative_conformation_info_dict[cluster_num][0]
        plddt = cluster_representative_conformation_info_dict[cluster_num][2]
        output_fname = 'cluster_%d_plddt_%s' % (cluster_num, str(round(plddt)))
        pdb_target_path = '%s/%s.pdb' % (cluster_dir, output_fname)
        shutil.copyfile(pdb_source_path, pdb_target_path)
        
        if not(args.skip_relaxation):
            with open(pdb_target_path, "r") as f:
                pdb_str = f.read()
            unrelaxed_protein = from_pdb_string(pdb_str)
            print('RELAXING PROTEIN FROM CLUSTER %d' % cluster_num)
            if torch.cuda.is_available():
                relax_protein(config, 'cuda:0', 
                              unrelaxed_protein, cluster_dir, 
                              output_fname)
            else:
                relax_protein(config, 'cpu', 
                              unrelaxed_protein, cluster_dir, 
                              output_fname)


        cluster_representative_conformation_info_dict[cluster_num].append(os.path.abspath(pdb_target_path))


    cluster_representative_conformation_info_fname = '%s/cluster_representative_conformation_info.pkl' % cluster_dir
    with open(cluster_representative_conformation_info_fname, 'wb') as f:
        pickle.dump(cluster_representative_conformation_info_dict, f)




 
