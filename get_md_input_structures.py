import pandas as pd 
import os
import shutil
import json
import re
import glob  
import sys



conformational_states_df = pd.read_csv('./conformational_states_testing_data/dataset/conformational_states_testing_data_processed_adjudicated.csv')

conformational_states_df = conformational_states_df[conformational_states_df['any_structures_present_in_training_set'] == 'n']
conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 

relevant_uniprot_id = list(conformational_states_df['uniprot_id']) 
print(relevant_uniprot_id)

template_str = 'template=none' 

for uniprot_id in relevant_uniprot_id:

    rw_relaxed_structures_pattern = './conformational_states_testing_data/rw_predictions/%s/*/%s/**/*_relaxed.pdb' % (uniprot_id, template_str)
    benchmark_relaxed_structures_pattern = './conformational_states_testing_data/benchmark_predictions/%s/*/%s/**/*_relaxed.pdb' % (uniprot_id, template_str)

    rw_files = sorted(glob.glob(rw_relaxed_structures_pattern, recursive=True))
    benchmark_files = sorted(glob.glob(benchmark_relaxed_structures_pattern, recursive=True))

    print(rw_files)
    print(len(rw_files))
    if len(rw_files) != 10:
        sys.exit()
    print(benchmark_files)
    print(len(benchmark_files))
    if len(benchmark_files) != 10:
        sys.exit()

    rw_save_dir = '/gpfs/home/itaneja/md_conformation_input/%s/rw' % uniprot_id
    benchmark_save_dir = '/gpfs/home/itaneja/md_conformation_input/%s/benchmark' % uniprot_id
    os.makedirs(rw_save_dir, exist_ok=True)
    os.makedirs(benchmark_save_dir, exist_ok=True)

    for f in sorted(rw_files):
        output_fname = f.split('/')[-1]
        pdb_target_path = '%s/%s' % (rw_save_dir, output_fname)
        shutil.copyfile(f, pdb_target_path)

    for f in sorted(benchmark_files):
        output_fname = f.split('/')[-1]
        pdb_target_path = '%s/%s' % (benchmark_save_dir, output_fname)
        shutil.copyfile(f, pdb_target_path)

