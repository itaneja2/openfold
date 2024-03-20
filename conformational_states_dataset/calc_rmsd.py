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
import boto3
from datetime import date
import io 
import requests
import urllib.request
from pymol import cmd
import json 
from joblib import Parallel, delayed
import numpy as np

sys.path.insert(0, '../')

from pdb_utils.pdb_utils import save_pdb_chain, get_model_name, get_pymol_cmd_superimpose, get_pymol_cmd_save, clean_pdb


############################3


def align_and_get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    #print('super %s,%s' % (s2,s1))

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom
    #print('RMSD: %.3f' % rmsd)

    if rmsd < 0:
        print("RMSD < 0")
        rmsd = 0 

    s2 = get_pymol_cmd_save(pdb2_model_name)
    cmd.save(pdb2_path, s2)
    cmd.delete('all')

    return rmsd 


def fetch_pdb(pdb_id: str, save_dir: str, clean=False) -> str:

    """
    Args:
        pdb_id: e.g 1xyz_A or 1xyz
        save_dir: e.g ./pdb_raw_structures_folder
    """ 

    pdb_save_path = "%s/%s.pdb" % (save_dir, pdb_id)

    if os.path.exists(pdb_save_path):
        return pdb_save_path

    cmd.reinitialize()
    if len(pdb_id.split('_')) > 1:
        pdb_id_wo_chain = pdb_id.split('_')[0]
        chain_id = pdb_id.split('_')[1]
        cmd.fetch(pdb_id_wo_chain, async_=0)
        cmd.save(pdb_save_path, "chain %s and %s" % (chain_id, pdb_id_wo_chain))
        chain_id_list = [chain_id]
    else:
        cmd.fetch(pdb_id, async_=0)
        cmd.save(pdb_save_path, pdb_id)
        chain_id_list = None
    cmd.delete('all')
 
    if clean:
        with open(pdb_save_path, "r") as f:
            pdb_str = f.read()
        clean_pdb(pdb_save_path, pdb_str)

    return pdb_save_path



def superimpose_wrapper_monomer(pdb1_full_id: str, pdb2_full_id: str, save_dir: str, parallel=False):

    """
    Args:
        pdb1_full_id: e.g 1xf2_A
        pdb2_full_id: e.g 1ya3_B
    """ 
 
    pdb1_id = pdb1_full_id.split('_')[0]
    pdb1_chain = pdb1_full_id.split('_')[1] 
    pdb2_id = pdb2_full_id.split('_')[0]
    pdb2_chain = pdb2_full_id.split('_')[1] 

    pdb_ref_struc_folder = '%s/pdb_ref_structure' % save_dir
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_superimposed_folder = '%s/pdb_superimposed_structures' % save_dir 
    Path(pdb_superimposed_folder).mkdir(parents=True, exist_ok=True)

    pdb1_path = fetch_pdb(pdb1_full_id, pdb_ref_struc_folder, clean=True)
    pdb1_chain_path = pdb1_path.replace('%s.pdb' % pdb1_id, '%s.pdb' % pdb1_full_id)
    pdb1_path = save_pdb_chain(pdb1_path, pdb1_chain_path, pdb1_chain) 
    pdb2_path = fetch_pdb(pdb2_full_id, pdb_superimposed_folder, clean=True) 
    pdb2_chain_path = pdb2_path.replace('%s.pdb' % pdb2_id, '%s.pdb' % pdb2_full_id)
    pdb2_path = save_pdb_chain(pdb2_path, pdb2_chain_path, pdb2_chain) 

    rmsd = round(align_and_get_rmsd(pdb1_path, pdb2_path, 'A', 'A'),3)
    print('SAVING ALIGNED PDB AT %s, RMSD=%.2f' % (pdb2_path,rmsd))

    return rmsd, pdb1_path, pdb2_path


def get_rmsd_dict(chunk_num):
    
    print('reading conformation_states_dict.pkl')
    with open("./data/conformation_states_dict.pkl", "rb") as f:
        conformation_states_dict = pickle.load(f)

    chunked_dict = np.array_split(conformation_states_dict, 100)

    rmsd_dict = {} 
    for i,conformation_info in enumerate(chunked_dict[chunk_num]):
        completion_percentage = round((i/len(chunked_dict[chunk_num]))*100,3)
        print('on instance %d, completion percentage %.3f' % (i, completion_percentage))
        print(conformation_info)
        for uniprot_id in conformation_info:
            rmsd_dict[uniprot_id] = {} 
            for segment_id in conformation_info[uniprot_id]:
                pdb_id_list = conformation_info[uniprot_id][segment_id]
                pdb_ref = pdb_id_list[0]
                rmsd_list = [] #calculate rmsd w.r.t the first pdb in the list 
                for i in range(1,len(pdb_id_list)):
                    rmsd, _, _ = superimpose_wrapper_monomer(pdb_ref, pdb_id_list[i], './pdb_structures')
                    rmsd_list.append(rmsd)
                rmsd_dict[uniprot_id][segment_id] = rmsd_list
        print(rmsd_dict[uniprot_id])

    print('saving rmsd_dict')
    output_fname = './data/rmsd_dict_chunk_%d.pkl' % chunk_num
    with open(output_fname, "wb") as f:
        pickle.dump(rmsd_dict, f)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_num", type=int, default=None
    )
    args = parser.parse_args()
    
    get_rmsd_dict(args.chunk_num)
