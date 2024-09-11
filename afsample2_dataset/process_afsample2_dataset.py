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
import io 
import requests
import urllib.request
import json 
from joblib import Parallel, delayed
from datetime import datetime, timezone 

 
cutoff_date = datetime(2019, 8, 28, tzinfo=timezone.utc)

def get_deposit_dates(pdb_ids):
    base_url = "https://data.rcsb.org/rest/v1/core/entry/"
    deposit_dates = {}
    structure_info = {} 
    
    for pdb_id in pdb_ids:
        try:
            response = requests.get(base_url + pdb_id)
            response.raise_for_status()
            data = response.json()
            deposit_date_str = data['rcsb_accession_info']['deposit_date']
            experimental_method = data['rcsb_entry_info']['experimental_method']
            resolution = 'None' 
            if 'resolution_combined' in data['rcsb_entry_info']:
                resolution = float(data['rcsb_entry_info']['resolution_combined'][0])
            structure_info[pdb_id] = [experimental_method, resolution] 

            deposit_date = datetime.strptime(deposit_date_str, "%Y-%m-%dT%H:%M:%S%z")
            # Convert to UTC for consistent comparison
            deposit_date = deposit_date.astimezone(timezone.utc)
            formatted_deposit_date = deposit_date.strftime("%Y-%m-%d")
            deposit_dates[pdb_id] = [deposit_date, formatted_deposit_date]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for PDB ID {pdb_id}: {e}")
        except KeyError:
            print(f"Deposit date not found for PDB ID {pdb_id}")
    
    return deposit_dates, structure_info 


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

def process_af2sample_dataset():

    """
    Generates list of dictionaries, where each entry corresponds to 
    'uniprot_id': {'seg_start-seg_end: ['pdb_id_1','pdb_id_2'] 
    (e.g: {'A0A022MRT4': {'1-436': ['6siw_B', '6siw_A']}}). 
    
    Dataset is derived from https://www.biorxiv.org/content/10.1101/2023.07.13.545008v2.full.pdf
    """ 
 
    print('reading pdb_openprotein.pkl')
    with open("../conformational_states_dataset/dataset/pdb_dict_openprotein.pkl", "rb") as f:
        all_pdb_dict = pickle.load(f)

    afsample2_df = pd.read_csv('./afsample2_dataset_raw.csv')
    print(afsample2_df)
    afsample2_df = afsample2_df[afsample2_df['exclude'] == 'n'].reset_index(drop=True)
    #afsample2_df = afsample2_df[afsample2_df['uniprot_id'] == 'A2RJ53'].reset_index(drop=True)

    all_uniprot_id = list(afsample2_df['uniprot_id'])
    print(len(all_uniprot_id))

    #IO bound, so use threads 
    results = Parallel(n_jobs=-1,prefer='threads')(delayed(get_superposition_info_openprotein)(all_uniprot_id, all_pdb_dict, i) for i in range(0,len(all_uniprot_id)))
    results = [v for v in results if v is not None]

    print(results)

    afsample2_dataset_processed = []
    for i,conformation_info in enumerate(results):
        print(results[i])
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

                    pdb_model_name_ref = pdb_ref.split('_')[0]
                    pdb_model_name_i = pdb_i.split('_')[0]

                    deposit_dates, structure_info = get_deposit_dates([pdb_model_name_ref,pdb_model_name_i])

                    print(deposit_dates)
                    print(structure_info)

                    use_instance = True 
                    for key in structure_info:
                        experimental_method = structure_info[key][0]
                        resolution = structure_info[key][1]
                        if 'nmr' in experimental_method.lower() or 'nuclear' in experimental_method.lower():
                            print('NOT USING THIS ROW - NMR STRUCTURE')
                            use_instance = False
                            break  
                        if resolution == 'None':
                            print('NOT USING THIS ROW - NO RESOLUTION')
                            use_instance = False
                            break  
                        elif resolution > 3.5:
                            print('NOT USING THIS ROW - RESOLUTION > 3.5')
                            use_instance = False
                            break  

                    if not(use_instance):
                        continue  
                           
                    use_instance = False 
                    if len(deposit_dates) > 0:
                        for key in deposit_dates:
                            if deposit_dates[key][0] > cutoff_date: 
                                use_instance = True 
                                print('KEEPING THIS ROW')
                    if not(use_instance):
                        print('NOT USING THIS ROW - DEPOSIT DATES')
                        continue 

                    pdb_id_ref_date = deposit_dates[pdb_model_name_ref][1]
                    pdb_id_ref_exp_method = structure_info[pdb_model_name_ref][0]
                    pdb_id_ref_resolution = structure_info[pdb_model_name_ref][1]

                    pdb_id_state_i_date = deposit_dates[pdb_model_name_i][1]
                    pdb_id_state_i_exp_method = structure_info[pdb_model_name_i][0]
                    pdb_id_state_i_resolution = structure_info[pdb_model_name_i][1]
 
                    curr_data = [uniprot_id, seg_start, seg_end, pdb_ref, pdb_i, pdb_id_msa, openprotein_present, pdb_id_ref_date, pdb_id_ref_exp_method, pdb_id_ref_resolution, pdb_id_state_i_date, pdb_id_state_i_exp_method, pdb_id_state_i_resolution]
                    afsample2_dataset_processed.append(curr_data)

    colnames = ['uniprot_id', 'seg_start', 'seg_end', 'pdb_id_ref', 'pdb_id_state_i', 'pdb_id_msa', 'openprotein_present', 'pdb_id_ref_date', 'pdb_id_ref_exp_method', 'pdb_id_ref_resolution', 'pdb_id_state_i_date', 'pdb_id_state_i_exp_method', 'pdb_id_state_i_resolution']
    df = pd.DataFrame(afsample2_dataset_processed, columns=colnames)
    df['seg_len'] = afsample2_df['seg_end']-afsample2_df['seg_start']+1
    df.to_csv('./afsample2_dataset_processed.csv', index=False)

    print(df)

process_af2sample_dataset()

