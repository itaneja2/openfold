import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO
import glob
import pandas as pd
import os
import sys 
import pickle
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import collections
import shutil

asterisk_line = '******************************************************************************'

def remove_files_in_dir(path):
    file_list = glob.glob('%s/*.pdb' % path)
    for f in file_list:
        print('removing old file: %s' % f)
        os.remove(f)


def get_structure_ca_coords(pdb_fname, md_conformation_str=None):  
    # Initialize a PDB parser
  
    if 'step-agg' in pdb_fname:
        temp = pdb_fname[pdb_fname.index('step-agg'):]
        agg_step_num = int(temp.split('-')[1][3:])
    else:
        agg_step_num = md_conformation_str

    if '.pdb' in pdb_fname:
        parser = PDBParser(QUIET=True)
    elif '.cif' in pdb_fname:
        parser = MMCIFParser(QUIET=True)

    # Load the CIF file
    structure = parser.get_structure('structure', pdb_fname)

    model = structure[0]
    # Extract CA coordinates
    ca_coordinates = []
    for chain in model:
        for residue in chain:
            if residue.has_id('CA'):
                ca_atom = residue['CA']
                ca_coordinates.append(list(ca_atom.get_coord()))
              
    ca_coordinates = np.array(ca_coordinates)            
    return ca_coordinates



           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conformation_dir", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10,
        help="",
    )
    parser.add_argument(
        "--ptm_iptm_threshold", type=float, default=None,
        help="only clusters conformations whose ptm_iptm score is greater than this threshold (which is a value between 0-1)",
    )

    '''parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )''' 

    args = parser.parse_args()

    '''cluster_representative_conformation_info_fname = '%s/cluster_representative_conformation_info.pkl' % args.conformation_dir
    with open(cluster_representative_conformation_info_fname, 'rb') as f:
        s = pickle.load(f)''' 

    pattern = "%s/**/conformation_info.pkl" % args.conformation_dir
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        print('conformation_info files do not exist')
        sys.exit()
    else:
        print('conformation_info files')
        print(files)

    '''for i in range(0,len(files)):
        with open(files[i], 'rb') as f:
            curr_conformation_info = pickle.load(f)
        
        for key in curr_conformation_info:
            for j in range(0,len(curr_conformation_info[key])):
                print(j)
                curr_conformation_info[key][j] = list(curr_conformation_info[key][j])
                pdb_path = curr_conformation_info[key][j][0]
                pdb_path = pdb_path.replace('ACCEPTED', 'exploration/ACCEPTED')
                curr_conformation_info[key][j][0] = pdb_path
                curr_conformation_info[key][j] = tuple(curr_conformation_info[key][j])

        with open(files[i], 'wb') as f:
            pickle.dump(curr_conformation_info, f)'''

    #combine into single dictionary
    for i in range(0,len(files)):
        with open(files[i], 'rb') as f:
            curr_conformation_info = pickle.load(f)
        if i == 0:
            conformation_info = curr_conformation_info.copy()
        else:
            conformation_info.update(curr_conformation_info)
 
    print(conformation_info.keys())
    for key in conformation_info:
        print(len(conformation_info[key]))
 
    conformation_info_all = [] 
    ca_coords_all = [] 
    for key in conformation_info:
        print('KEY: %s' % key)
        for i,val in enumerate(conformation_info[key]):
            print(i)
            pdb_path = val[0]
            rmsd = val[1]
            mean_plddt = val[2]
            ptm_iptm = float(val[3])
            if args.ptm_iptm_threshold is None:
                conformation_info_all.append([pdb_path,rmsd,mean_plddt,ptm_iptm])
                ca_coords = get_structure_ca_coords(pdb_path)
                ca_coords_all.append(ca_coords)
            else:
                if ptm_iptm >= args.ptm_iptm_threshold:
                    conformation_info_all.append([pdb_path,rmsd,mean_plddt,ptm_iptm])
                    ca_coords = get_structure_ca_coords(pdb_path)
                    ca_coords_all.append(ca_coords)
        
    ca_coords_all = np.array(ca_coords_all)
    ca_coords_all = np.reshape(ca_coords_all,(ca_coords_all.shape[0],ca_coords_all.shape[1]*ca_coords_all.shape[2])) 
       
    clustering = AgglomerativeClustering(n_clusters=args.num_clusters).fit(ca_coords_all)

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
        clustering_info_dict[cluster_num] = sorted(clustering_info_dict[cluster_num], key=lambda x: x[3], reverse=True) #sort by ptm_iptm

    cluster_representative_conformation_info_dict = {} 
    for cluster_num,freq in counter.most_common():
        num_structures_in_cluster = len(clustering_info_dict[cluster_num])
        halfway_idx = num_structures_in_cluster//2
        cluster_representative_conformation_info_dict[cluster_num] = clustering_info_dict[cluster_num][halfway_idx] 

    print(cluster_representative_conformation_info_dict)

    if args.ptm_iptm_threshold is None:
        cluster_dir = '%s/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=None' % (args.conformation_dir, args.num_clusters)
    else:
        cluster_dir = '%s/cluster_representative_structures/num_clusters=%d/ptm_iptm_threshold=%s' % (args.conformation_dir, args.num_clusters, str(round(args.ptm_iptm_threshold,2)))

    os.makedirs(cluster_dir, exist_ok=True)
    remove_files_in_dir(cluster_dir)

    for cluster_num in cluster_representative_conformation_info_dict:
        pdb_source_path = cluster_representative_conformation_info_dict[cluster_num][0]
        ptm_iptm = cluster_representative_conformation_info_dict[cluster_num][3]
        pdb_target_path = '%s/cluster_%d_ptm_iptm_%s.pdb' % (cluster_dir, cluster_num, str(round(ptm_iptm)))
        shutil.copyfile(pdb_source_path, pdb_target_path)
        #add this path to cluster_representative_conformation_info_dict[clsuter_num]
        cluster_representative_conformation_info_dict[cluster_num].append(pdb_target_path)

    cluster_representative_conformation_info_fname = '%s/cluster_representative_conformation_info.pkl' % cluster_dir
    with open(cluster_representative_conformation_info_fname, 'wb') as f:
        pickle.dump(cluster_representative_conformation_info_dict, f)


    
    
    
