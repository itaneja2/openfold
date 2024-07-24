import requests
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
import json 
import re 
from datetime import datetime, timezone 

#get instances where rmsd between states > 30
#at least one structure is past AF training date
#no NMR structures 
#no structures w/o any resolution info 

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


conformational_states_df = pd.read_csv('../conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)
conformational_states_df = conformational_states_df[conformational_states_df['rmsd_wrt_pdb_id_ref'] > 25].reset_index(drop=True)
#conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'P09167'].reset_index(drop=True)

conformational_states_post_AF_training = pd.DataFrame()

for index,row in conformational_states_df.iterrows():

    completion_percentage = round((index/len(conformational_states_df))*100,3)
    print(completion_percentage)

    new_row = row.to_dict()
    print(row)

    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref']).split('_')[0]
    pdb_model_name_state_i = str(row['pdb_id_state_i']).split('_')[0]
    
    deposit_dates, structure_info = get_deposit_dates([pdb_model_name_ref,pdb_model_name_state_i])

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
    else: 
        new_row['pdb_id_ref_date'] = deposit_dates[pdb_model_name_ref][1]
        new_row['pdb_id_ref_exp_method'] = structure_info[pdb_model_name_ref][0]
        new_row['pdb_id_ref_resolution'] = structure_info[pdb_model_name_ref][1]

        new_row['pdb_id_state_i_date'] = deposit_dates[pdb_model_name_state_i][1]
        new_row['pdb_id_state_i_exp_method'] = structure_info[pdb_model_name_state_i][0]
        new_row['pdb_id_state_i_resolution'] = structure_info[pdb_model_name_state_i][1]

        curr_df = pd.DataFrame(new_row, index=[0])
        conformational_states_post_AF_training = pd.concat([conformational_states_post_AF_training, curr_df], axis=0, ignore_index=True)

        print(conformational_states_post_AF_training) 


conformational_states_post_AF_training.to_csv('./dataset/conformational_states_filtered_adjudicated_post_AF_training.csv', index=False)
print(conformational_states_post_AF_training)

