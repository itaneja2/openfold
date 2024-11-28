import pandas as pd 
import os
import shutil
import json
import re
import glob  
import sys


testset_dir = './metadynamics_testset_results' #./conformational_states_testset_results

conformational_states_df_path = './metadynamics_dataset/metadynamics_dataset_processed.csv' #'./afsample2_dataset/afsample2_dataset_processed_adjudicated.csv'

conformational_states_df = pd.read_csv(conformational_states_df_path)
conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 
uniprot_ids = list(conformational_states_df['uniprot_id']) 
print(uniprot_ids)

template_str = 'template=none' 

for uniprot_id in uniprot_ids:

    print('*****')
    print(uniprot_id)
    print('*****')

    rw_openmm_folder_pattern = '%s/rw_predictions/%s/*/**/openmm_refined_structures' % (testset_dir,uniprot_id)
    benchmark_openmm_folder_pattern = '%s/benchmark_predictions/%s/**/openmm_refined_structures' % (testset_dir,uniprot_id)

    rw_openmm_folder = glob.glob(rw_openmm_folder_pattern, recursive=True)[0]
    benchmark_openmm_folder = glob.glob(benchmark_openmm_folder_pattern, recursive=True)[0]

    print(rw_openmm_folder)
    print(benchmark_openmm_folder)

    rw_structural_issues_info_summary_file = '%s/structural_issues_info_summary.json' % rw_openmm_folder[0:rw_openmm_folder.rindex('/')]
    benchmark_structural_issues_info_summary_file = '%s/structural_issues_info_summary.json' % benchmark_openmm_folder[0:benchmark_openmm_folder.rindex('/')]

    print(rw_structural_issues_info_summary_file)
    print(benchmark_structural_issues_info_summary_file)

    rw_save_dir = '/gpfs/home/itaneja/openfold/md_conformation_input/%s/rw' % uniprot_id
    benchmark_save_dir = '/gpfs/home/itaneja/openfold/md_conformation_input/%s/benchmark' % uniprot_id
    os.makedirs(rw_save_dir, exist_ok=True)
    os.makedirs(benchmark_save_dir, exist_ok=True)

    print("copying %s to %s" % (rw_openmm_folder, rw_save_dir))
    print("copying %s to %s" % (benchmark_openmm_folder, benchmark_save_dir))

    shutil.copytree(rw_openmm_folder, rw_save_dir, dirs_exist_ok=True)
    shutil.copytree(benchmark_openmm_folder, benchmark_save_dir, dirs_exist_ok=True) 

    f = rw_structural_issues_info_summary_file 
    output_fname = f.split('/')[-1]
    output_path = '%s/%s' % (rw_save_dir, output_fname)
    shutil.copyfile(f, output_path)

    f = benchmark_structural_issues_info_summary_file 
    output_fname = f.split('/')[-1]
    output_path = '%s/%s' % (benchmark_save_dir, output_fname)
    shutil.copyfile(f, output_path)

