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
                found = False 
                for i in range(0,len(curr_cluster_data)):
                    pdb_id = curr_cluster_data[i]['pdb_id'] 
                    auth_asym_id = curr_cluster_data[i]['auth_asym_id'] #pymol uses this as the chain_id 
                    representative = curr_cluster_data[i]['is_representative']
                    if representative:
                        rep_pdb_id_w_chain = '%s_%s' % (pdb_id, auth_asym_id)
                    pdb_id_w_chain = '%s_%s' % (pdb_id, auth_asym_id)
                    if pdb_id_w_chain in all_pdb_dict:
                        pdb_list.append((pdb_id_w_chain,1))
                        found = True 
                        break
                if not(found):
                    pdb_list.append((rep_pdb_id_w_chain,0))

            parsed_dict[uniprot_id][seg_key] = pdb_list

                 
    if parsed_dict[uniprot_id] == {}: 
        return None    
    else:  
        return parsed_dict  
       
def get_superposition_info(uniprot_id, all_pdb_dict):
    base_api_uri = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/superposition/"
    query = f"{base_api_uri}{uniprot_id}"
    print(query)
    request = requests.get(query, allow_redirects=True)
    if request.status_code != 200:
        print('%s not found w/ superposition info' % uniprot_id)
        return None 
    else:
        json_dict = json.loads(request.content.decode('utf-8'))
        parsed_dict = parse_superposition_info(json_dict, uniprot_id, all_pdb_dict)
        return parsed_dict

def get_superposition_info_openprotein(all_uniprot_id, all_pdb_dict, i):
    uniprot_id = all_uniprot_id[i]
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
    with open("../conformational_states_dataset/dataset/pdb_dict_openprotein.pkl", "rb") as f:
        all_pdb_dict = pickle.load(f)

    conformational_states_df = pd.read_csv('./dataset/conformational_states_testing_data_manual.csv')
    conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

    all_uniprot_id = list(conformational_states_df['uniprot_id'])
    print(len(all_uniprot_id))

    #IO bound, so use threads 
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(get_superposition_info_openprotein)(all_uniprot_id, all_pdb_dict, i) for i in range(0,len(all_uniprot_id)))
    results = [v for v in results if v is not None]

    #with open("./dataset/conformation_states_dict.pkl", "wb") as f:
    #    pickle.dump(results, f)

    print(results)

    conformational_states_dataset = []
    for i,conformation_info in enumerate(results):
        for uniprot_id in conformation_info:
            for segment_id in conformation_info[uniprot_id]:
                seg_start = segment_id.split('-')[0]
                seg_end = segment_id.split('-')[1]
                pdb_id_list = conformation_info[uniprot_id][segment_id]
                pdb_ref, pdb_ref_openprotein_status = pdb_id_list[0]
                for i in range(1,len(pdb_id_list)):
                    pdb_i, pdb_i_openprotein_status = pdb_id_list[i]
                    if pdb_ref_openprotein_status == 1:
                        pdb_id_msa = pdb_ref
                        openprotein_present = True
                    elif pdb_i_openprotein_status == 1:
                        pdb_id_msa = pdb_i
                        openprotein_present = True
                    else:
                        pdb_id_msa = pdb_ref
                        openprotein_present = False
                    curr_data = [uniprot_id, seg_start, seg_end, pdb_ref, pdb_i, pdb_id_msa, openprotein_present]
                    conformational_states_dataset.append(curr_data)

    df = pd.DataFrame(conformational_states_dataset, columns=['uniprot_id', 'seg_start', 'seg_end', 'pdb_id_ref', 'pdb_id_state_i', 'pdb_id_msa', 'openprotein_present'])
    df.to_csv('./dataset/conformational_states_testing_data_processed.csv', index=False)

    print(df)

#gen_conformation_states_dict()

conformational_states_df = pd.read_csv('./dataset/conformational_states_testing_data_processed.csv')
conformational_states_df = conformational_states_df[conformational_states_df['pdb_id_msa'] != '2mq9_A'].reset_index(drop=True) #there are two instances for this uniprot, other one corresponding to segment 1-437 is correct
conformational_states_df = conformational_states_df[conformational_states_df['pdb_id_msa'] != '5ho9_A'].reset_index(drop=True) #5ho0_A is the closed conformation, not 5ho9_A 
conformational_states_df['seg_len'] = conformational_states_df['seg_end']-conformational_states_df['seg_start']+1
conformational_states_df = conformational_states_df[conformational_states_df['seg_len'] < 800]

conformational_states_df.to_csv('./dataset/conformational_states_testing_data_processed_adjudicated.csv', index=False)
