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

    os.makedirs('./dataset', exist_ok=True)
    with open("./dataset/pdb_list_openprotein.pkl", "wb") as f:
        pickle.dump(all_pdb_list, f)
    with open("./dataset/pdb_dict_openprotein.pkl", "wb") as f:
        pickle.dump(all_pdb_dict, f)
 

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
            #    print('not all clusters have a pdb present in openprotein dataset')
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
    with open("./dataset/pdb_dict_openprotein.pkl", "rb") as f:
        all_pdb_dict = pickle.load(f)

    print('reading uniprot_pdb.csv')
    uniprot_pdb_df = pd.read_csv('uniprot_pdb.csv', skiprows=1)
    all_uniprot_id = list(uniprot_pdb_df['SP_PRIMARY'])
    print(len(all_uniprot_id))

    #IO bound, so use threads 
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(get_superposition_info_openprotein)(all_uniprot_id, all_pdb_dict, i) for i in range(0,len(all_uniprot_id)))
    results = [v for v in results if v is not None]

    with open("./dataset/conformation_states_dict.pkl", "wb") as f:
        pickle.dump(results, f)

    print(results)
    print(len(results))

gen_conformation_states_dict()

