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

sys.path.insert(0, '../')

from pdb_utils.pdb_utils import save_pdb_chain, get_model_name, get_pymol_cmd_superimpose, get_pymol_cmd_save, clean_pdb

def get_all_pdb_openprotein():
    s3 = boto3.client('s3')
    bucket_name = 'openfold'
    prefix = 'pdb/'
    paginator = s3.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    all_pdb_list = []
    all_pdb_dict = {} 
     
    for response in response_iterator:
        for common_prefix in response.get('CommonPrefixes', []):
            pdb_id_w_chain = common_prefix['Prefix'].split('/')[1]
            all_pdb_list.append(pdb_id_w_chain)
            all_pdb_dict[pdb_id_w_chain] = 1 

    print(all_pdb_list)    
    print(len(all_pdb_list))
    sdf
    os.makedirs('./data', exist_ok=True)
    with open("./data/pdb_list_openprotein.pkl", "wb") as f:
        pickle.dump(all_pdb_list, f)
    with open("./data/pdb_dict_openprotein.pkl", "wb") as f:
        pickle.dump(all_pdb_dict, f)

get_all_pdb_openprotein()
 

def parse_superposition_info(json_dict, uniprot_id, all_pdb_dict):
    parsed_dict = {} 
    parsed_dict[uniprot_id] = {} 
    for key in json_dict:
        cluster_info = json_dict[key]
        for info in cluster_info:
            seg_start = info['segment_start']
            seg_end = info['segment_end']
            cluster_data = info['clusters']
            seg_key = '%d-%d' % (seg_start,seg_end)
            if len(cluster_data) == 1:
                continue 
            pdb_list = []  
            for curr_cluster_data in cluster_data: #each element in cluster_data corresponds to a list of pdbs 
                for i in range(0,len(curr_cluster_data)):
                    pdb_id = curr_cluster_data[i]['pdb_id'] 
                    auth_asym_id = curr_cluster_data[i]['auth_asym_id'] #pymol uses this as the chain_id 
                    pdb_id_w_chain = '%s_%s' % (pdb_id, auth_asym_id)
                    if pdb_id_w_chain in all_pdb_dict:
                        pdb_list.append(pdb_id_w_chain)
                        break 
            if len(pdb_list) == len(cluster_data):
                parsed_dict[uniprot_id][seg_key] = pdb_list
            #else:
            #    print('not all values of pdb_list present in openprotein')
            #    print(pdb_list)

    if parsed_dict[uniprot_id] == {}: 
        return None    
    else:  
        return parsed_dict  
       
def get_superposition_info(uniprot_id, all_pdb_dict):
    base_api_uri = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/superposition/"
    query = f"{base_api_uri}{uniprot_id}"
    request = requests.get(query, allow_redirects=True)
    if request.status_code != 200:
        #print('%s not found w/ superposition info' % uniprot_id)
        return None 
    else:
        json_dict = json.loads(request.content.decode('utf-8'))
        parsed_dict = parse_superposition_info(json_dict, uniprot_id, all_pdb_dict)
        return parsed_dict

def get_superposition_info_openprotein(all_uniprot_id, all_pdb_dict, i):
    uniprot_id = all_uniprot_id[i]
    if i % 100 == 0:
        completion_percentage = round((i/len(all_uniprot_id))*100,3)
        print('on uniprot_id %s, completion percentage %.3f' % (uniprot_id, completion_percentage))
    superposition_info = get_superposition_info(uniprot_id, all_pdb_dict)
    return superposition_info

def gen_conformation_states_dict():

    """
    Generates list of dictionaries, where each entry corresponds to 
    'uniprot_id': {'seg_start-seg_end: ['pdb_id_1','pdb_id_2'] 
    (e.g: {'A0A022MRT4': {'1-436': ['6siw_B', '6siw_A']}}). 
    
    Dataset is derived from https://www.biorxiv.org/content/10.1101/2023.07.13.545008v2.full.pdf
    """ 
 
    print('reading pdb_openprotein.pkl')
    with open("./data/pdb_dict_openprotein.pkl", "rb") as f:
        all_pdb_dict = pickle.load(f)

    print('reading uniprot_pdb.csv')
    uniprot_pdb_df = pd.read_csv('uniprot_pdb.csv', skiprows=1)
    all_uniprot_id = list(uniprot_pdb_df['SP_PRIMARY'])
    print(len(all_uniprot_id))

    #IO bound, so use threads 
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(get_superposition_info_openprotein)(all_uniprot_id, all_pdb_dict, i) for i in range(0,len(all_uniprot_id)))
    results = [v for v in results if v is not None]

    with open("./data/conformation_states_dict.pkl", "wb") as f:
        pickle.dump(results, f)

    print(results)
    print(len(results))

#gen_conformation_states_dict()


'''

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


def fetch_pdb(pdb_id: str, save_dir: str, clean=False, parallel=False) -> str:

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

    if not(parallel):
        fetch_path = './%s.cif' % pdb_id_wo_chain
        if os.path.exists(fetch_path):
            os.remove(fetch_path)    
    
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

    pdb1_path = fetch_pdb(pdb1_full_id, pdb_ref_struc_folder, clean=True, parallel=parallel)
    pdb1_chain_path = pdb1_path.replace('%s.pdb' % pdb1_id, '%s.pdb' % pdb1_full_id)
    pdb1_path = save_pdb_chain(pdb1_path, pdb1_chain_path, pdb1_chain) 
    pdb2_path = fetch_pdb(pdb2_full_id, pdb_superimposed_folder, clean=True, parallel=parallel) 
    pdb2_chain_path = pdb2_path.replace('%s.pdb' % pdb2_id, '%s.pdb' % pdb2_full_id)
    pdb2_path = save_pdb_chain(pdb2_path, pdb2_chain_path, pdb2_chain) 

    if parallel:
        rmsd = round(align_and_get_rmsd(pdb1_path, pdb2_path, 'A', 'A'),3)
        print('SAVING ALIGNED PDB AT %s, RMSD=%.2f' % (pdb2_path,rmsd))
    else:
        rmsd = None

    return rmsd, pdb1_path, pdb2_path

def get_rmsd_info(conformation_states_dict, i):
    
    rmsd_dict = {} 
    conformation_info = conformation_states_dict[i]
    
    completion_percentage = round((i/len(conformation_states_dict))*100,3)
    print('on instance %d, completion percentage %.3f' % (i, completion_percentage))
    print(conformation_info)
    for uniprot_id in conformation_info:
        rmsd_dict[uniprot_id] = {} 
        for segment_id in conformation_info[uniprot_id]:
            pdb_id_list = conformation_info[uniprot_id][segment_id]
            pdb_ref = pdb_id_list[0]
            rmsd_list = [] 
            for i in range(1,len(pdb_id_list)):
                rmsd, _, _ = superimpose_wrapper_monomer(pdb_ref, pdb_id_list[i], './pdb_structures', True)
                rmsd_list.append(rmsd)
            rmsd_dict[uniprot_id][segment_id] = rmsd_list
    print(rmsd_dict[uniprot_id])

    return rmsd_dict 
        
def get_rmsd_dict_parallel():

    print('reading conformation_states_dict.pkl')
    with open("./data/conformation_states_dict.pkl", "rb") as f:
        conformation_states_dict = pickle.load(f)

    #use loky, because this causes conflicts in pymol if we use threads
    rmsd_dict = Parallel(n_jobs=-1,backend='loky')(delayed(get_rmsd_info)(conformation_states_dict, i) for i in range(0,len(conformation_states_dict)))

    with open("./data/rmsd_dict.pkl", "wb") as f:
        pickle.dump(rmsd_dict, f)
''' 
