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
import numpy as np
import re

from custom_openfold_utils.pdb_utils import get_rmsd, get_cif_string_from_pdb

def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs

def save_conformation_vectorfield_training_data(data, output_fname):
    output_dir = './conformation_vectorfield_data' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


template_id_rw_conformation_path = './conformation_vectorfield_data/template_id_rw_conformation_path_dict.pkl'
residues_mask_path = './conformation_vectorfield_data/af_conformations_residues_mask_dict.pkl'


with open(template_id_rw_conformation_path, 'rb') as f:
    template_id_rw_conformation_path_dict = pickle.load(f) 

with open(residues_mask_path, 'rb') as f:
    residues_mask_dict = pickle.load(f) 


print(len(template_id_rw_conformation_path_dict.keys()))

for key in list(template_id_rw_conformation_path_dict.keys()):
    if len(template_id_rw_conformation_path_dict[key]) == 0: 
        print('deleting %s' % key)
        del template_id_rw_conformation_path_dict[key]

for key in list(template_id_rw_conformation_path_dict.keys()):
    uniprot_id, template_pdb_id = key.split('-') 
    curr_alignment_dir = './alignment_data/%s/%s' % (uniprot_id, template_pdb_id)
    fasta_file = "%s/%s.fasta" % (curr_alignment_dir, template_pdb_id)
    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seq = parse_fasta(fasta_data)
    seq = seq[0]

    if 'X' in seq:
        print('X present')
        print(key)
        del template_id_rw_conformation_path_dict[key]

print(len(template_id_rw_conformation_path_dict.keys()))

print('here')

for idx,key in enumerate(list(template_id_rw_conformation_path_dict.keys())):
 
    all_rw_conformation_paths = template_id_rw_conformation_path_dict[key]
    rw_conformation_path = all_rw_conformation_paths[0]

    for rw_conformation_path in all_rw_conformation_paths:
        if rw_conformation_path not in residues_mask_dict:
            print('%s not in residues_mask_dict' % key)
            print('deleting')
            del template_id_rw_conformation_path_dict[key]
            break 
    

print(len(template_id_rw_conformation_path_dict.keys()))


#for cases with 3 conformations 
for key in list(template_id_rw_conformation_path_dict.keys()):
    uniprot_id, template_id = key.split('-')
    all_rw_conformation_paths = template_id_rw_conformation_path_dict[key]
    rw_conformation_path = all_rw_conformation_paths[0]
    if template_id not in rw_conformation_path:
        print(key)
        del template_id_rw_conformation_path_dict[key]
     
print(len(template_id_rw_conformation_path_dict.keys()))


for idx,key in enumerate(list(template_id_rw_conformation_path_dict.keys())):

    if idx % 50 == 0:
        print(idx)

    uniprot_id, template_id = key.split('-')
    all_rw_conformation_paths = template_id_rw_conformation_path_dict[key]
    rw_conformation_path = all_rw_conformation_paths[0]
    
    try:
        get_cif_string_from_pdb(rw_conformation_path)
    except ValueError:
        print('cant read %s' % rw_conformation_path)
        del template_id_rw_conformation_path_dict[key]

print(len(template_id_rw_conformation_path_dict.keys()))         

print('saving template_id_rw_conformation_path_dict')
save_conformation_vectorfield_training_data(template_id_rw_conformation_path_dict, 'template_id_rw_conformation_path_dict')

