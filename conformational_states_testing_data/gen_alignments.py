import numpy as np
import pandas as pd 
import os
import shutil
import json
import re
import glob  
import sys
import itertools
import subprocess 
import pickle

from custom_openfold_utils.pdb_utils import get_uniprot_id

gen_msa_monomer_path = '../msa_utils/gen_msa_monomer.py' 

asterisk_line = '******************************************************************************'

def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


conformational_states_df = pd.read_csv('./dataset/conformational_states_testing_data_processed_adjudicated.csv')

arg2 = '--msa_save_dir=./alignment_data'
arg3 = '--batch_processing'
arg4 = '--template_mmcif_dir=/dev/shm/pdb_mmcif/mmcif_files'
arg5= '--uniref90_database_path=/dev/shm/uniref90/uniref90.fasta'
arg6 = '--mgnify_database_path=/dev/shm/mgnify/mgy_clusters_2022_05.fa'
arg7 = '--pdb70_database_path=/dev/shm/pdb70/pdb70'
arg8 = '--uniclust30_database_path=/dev/shm/uniref30/UniRef30_2021_03'
arg9 = '--bfd_database_path=/dev/shm/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
arg10 = '--jackhmmer_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/jackhmmer'
arg11 = '--hhblits_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhblits'
arg12 = '--hhsearch_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhsearch'
arg13 = '--kalign_binary_path=/opt/applications/kalign/2.04/gnu/bin/kalign' 

print('getting all existing feature pkl paths')
existing_features_pkl_paths = glob.glob('./alignment_data/**/**/features.pkl')
print("%d feature.pkl already exist" % len(existing_features_pkl_paths))

for index,row in conformational_states_df.iterrows():

    print('On row %d of %d' % (index, len(conformational_states_df)))   
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_id = str(row['pdb_id_msa'])

    features_path = './alignment_data/%s/%s/features.pkl' % (uniprot_id, pdb_id)
    
    if features_path not in existing_features_pkl_paths:
        arg1 = '--pdb_id=%s' % pdb_id
        script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13]
        cmd_to_run = ["python", gen_msa_monomer_path] + script_arguments
        cmd_to_run_str = s = ' '.join(cmd_to_run)
        print(asterisk_line)
        print("RUNNING THE FOLLOWING COMMAND:")
        print(cmd_to_run_str)
        subprocess.run(cmd_to_run)

        uniprot_id_from_sifts = get_uniprot_id(pdb_id)
        
        if uniprot_id != uniprot_id_from_sifts:
            curr_folder_name =  './alignment_data/%s' % uniprot_id_from_sifts
            new_folder_name = './alignment_data/%s' % uniprot_id
            pdb_curr_folder = '%s/%s' % (curr_folder_name, pdb_id)
            pdb_new_folder = '%s/%s' % (new_folder_name, pdb_id)
            if not(os.path.exists(new_folder_name)):
                print('renaming %s to %s' % (curr_folder_name,new_folder_name))
                os.rename(curr_folder_name, new_folder_name)
            else:
                print('copying %s to %s' % (pdb_curr_folder, new_folder_name))
                shutil.copytree(pdb_curr_folder, pdb_new_folder, dirs_exist_ok=True)
                shutil.rmtree(curr_folder_name)
    else:
        print('%s already exists' % features_path)
        continue 

    fasta_file = './alignment_data/%s/%s/%s.fasta' % (uniprot_id, pdb_id, pdb_id)

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seq = parse_fasta(fasta_data)
    seq = seq[0]
    print(seq)
    print('sequence length: %d' % len(seq)) 

