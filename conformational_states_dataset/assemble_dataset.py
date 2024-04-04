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



def get_rmsd_list(rmsd_dict_all, uniprot_id, segment_id):
    if uniprot_id in rmsd_dict_all:
        if segment_id in rmsd_dict_all[uniprot_id]:
            return rmsd_dict_all[uniprot_id][segment_id]
    return None 

rmsd_dict_all = {} 
for i in range(0,100):
    output_fname = './dataset/rmsd_dict_chunk_%d.pkl' % i 
    with open(output_fname, "rb") as f:
        rmsd_dict_chunk_i = pickle.load(f)
    rmsd_dict_all.update(rmsd_dict_chunk_i)


conformational_states_dataset = [] 

print('reading conformation_states_dict.pkl')
with open("./dataset/conformation_states_dict.pkl", "rb") as f:
    conformation_states_dict = pickle.load(f)

for i,conformation_info in enumerate(conformation_states_dict):
    completion_percentage = round((i/len(conformation_states_dict))*100,3)
    if i % 100 == 0:
        print('on instance %d, completion percentage %.3f' % (i, completion_percentage))
    for uniprot_id in conformation_info:
        for segment_id in conformation_info[uniprot_id]:
            seg_start = segment_id.split('-')[0]
            seg_end = segment_id.split('-')[1]
            rmsd_list = get_rmsd_list(rmsd_dict_all, uniprot_id, segment_id) #rmsd is taken w.r.t first pdb in list
            if rmsd_list is not None:
                pdb_id_list = conformation_info[uniprot_id][segment_id]
                pdb_ref = pdb_id_list[0]
                for i in range(1,len(pdb_id_list)):
                    pdb_i = pdb_id_list[i]
                    rmsd_wrt_pdb_id_ref = rmsd_list[i-1] 
                    curr_data = [uniprot_id, seg_start, seg_end, pdb_ref, pdb_i, rmsd_wrt_pdb_id_ref]
                    conformational_states_dataset.append(curr_data)


df = pd.DataFrame(conformational_states_dataset, columns=['uniprot_id', 'seg_start', 'seg_end', 'pdb_id_ref', 'pdb_id_state_i', 'rmsd_wrt_pdb_id_ref'])
df.to_csv('./dataset/conformational_states_df.csv', index=False)
print(df)
