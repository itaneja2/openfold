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

sys.path.insert(0, '../')
from openfold.utils.script_utils import parse_fasta
from pdb_utils.pdb_utils import get_uniprot_id

gen_msa_monomer_path = '../msa_utils/gen_msa_monomer.py' 

asterisk_line = '******************************************************************************'


conformational_states_df = pd.read_csv('./data/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

arg2 = '--msa_save_dir=./alignment_data'
arg3 = '--template_mmcif_dir=/dev/shm/pdb_mmcif/mmcif_files'
arg4 = '--uniref90_database_path=/dev/shm/uniref90/uniref90.fasta'
arg5 = '--mgnify_database_path=/dev/shm/mgnify/mgy_clusters_2022_05.fa'
arg6 = '--pdb70_database_path=/dev/shm/pdb70/pdb70'
arg7 = '--uniclust30_database_path=/dev/shm/uniref30/UniRef30_2021_03'
arg8 = '--bfd_database_path=/dev/shm/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
arg9 = '--jackhmmer_binary_path=/opt/applications/hmmer/3.3.2/gnu/bin/jackhmmer'
arg10 = '--hhblits_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhblits'
arg11 = '--hhsearch_binary_path=/opt/applications/hhsuite/3.3.0/gnu/bin/hhsearch'
arg12 = '--kalign_binary_path=/opt/applications/kalign/2.04/gnu/bin/kalign' 


for index,row in conformational_states_df.iterrows():

    print('On row %d of %d' % (index, len(conformational_states_df)))   
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_id_ref = str(row['pdb_id_ref'])
    #pdb_id_state_i = str(row['pdb_id_state_i'])
    seg_len = int(row['seg_len'])
    uniprot_id_from_sifts = get_uniprot_id(pdb_id_ref)

    features_path = './alignment_data/%s/%s/features.pkl' % (uniprot_id, pdb_id_ref)
    
    if not(os.path.exists(features_path)):
        arg1 = '--pdb_id=%s' % pdb_id_ref 
        script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12]
        cmd_to_run = ["python", gen_msa_monomer_path] + script_arguments
        cmd_to_run_str = s = ' '.join(cmd_to_run)
        print(asterisk_line)
        print("RUNNING THE FOLLOWING COMMAND:")
        print(cmd_to_run_str)
        subprocess.run(cmd_to_run)
        
        curr_folder_name =  './alignment_data/%s' % uniprot_id_from_sifts
        new_folder_name = './alignment_data/%s' % uniprot_id
        print('renaming %s to %s' % (curr_folder_name,new_folder_name))
        os.rename(curr_folder_name, new_folder_name)
    else:
        print('%s already exists' % features_path)

    fasta_file = './alignment_data/%s/%s/%s.fasta' % (uniprot_id, pdb_id_ref, pdb_id_ref)

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seq = parse_fasta(fasta_data)
    seq = seq[0]
    print(seq)
    print('sequence length: %d, seg_len: %d' % (len(seq), seg_len))



