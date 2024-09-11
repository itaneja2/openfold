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

    rw_openmm_structures_pattern = './conformational_states_testing_data/rw_predictions/%s/*/%s/**/*_openmm_refinement.pdb' % (uniprot_id, template_str)
    benchmark_openmm_structures_pattern = './conformational_states_testing_data/benchmark_predictions/%s/*/%s/**/*_openmm_refinement.pdb' % (uniprot_id, template_str)

    print(rw_openmm_structures_pattern)

    rw_files = sorted(glob.glob(rw_openmm_structures_pattern, recursive=True))
    benchmark_files = sorted(glob.glob(benchmark_openmm_structures_pattern, recursive=True))

    print(rw_files)
    
    if len(rw_files) > 0:
        rw_structural_issues_info_summary_file = '%s/structural_issues_info_summary.json' % rw_files[0][0:rw_files[0].rindex('/')]
        rw_files.append(rw_structural_issues_info_summary_file)
    if len(benchmark_files) > 0:
        benchmark_structural_issues_info_summary_file = '%s/structural_issues_info_summary.json' % benchmark_files[0][0:benchmark_files[0].rindex('/')]
        benchmark_files.append(benchmark_structural_issues_info_summary_file)


    #print(rw_structural_issues_info_summary_file)
    #print(benchmark_structural_issues_info_summary_file)


    print(rw_files)
    print(len(rw_files))
    print(benchmark_files)
    print(len(benchmark_files))

    rw_save_dir = '/gpfs/home/itaneja/md_conformation_input/%s/rw' % uniprot_id
    benchmark_save_dir = '/gpfs/home/itaneja/md_conformation_input/%s/benchmark' % uniprot_id
    os.makedirs(rw_save_dir, exist_ok=True)
    os.makedirs(benchmark_save_dir, exist_ok=True)

    for f in sorted(rw_files):
        if 'initial' not in f:
            output_fname = f.split('/')[-1]
            pdb_target_path = '%s/%s' % (rw_save_dir, output_fname)
            shutil.copyfile(f, pdb_target_path)

    for f in sorted(benchmark_files):
        if 'initial' not in f:
            output_fname = f.split('/')[-1]
            pdb_target_path = '%s/%s' % (benchmark_save_dir, output_fname)
            shutil.copyfile(f, pdb_target_path)


