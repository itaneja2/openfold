import sys
import os
import subprocess
import pandas as pd
import argparse
import glob  
import re 
import shutil 

def submit_job(script_path):         
    try:
        result = subprocess.run(["sbatch", script_path], 
                                check=True, 
                                capture_output=True, 
                                text=True)
        print(f"Submitted {script_path}: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {filename}: {e}")


def gen_unbiased_md_bash_script_forlipartition(input_receptor_path, water_model, production_steps, output_dir, job_name, job_dir):

    script = f"""#!/bin/sh  
#SBATCH --job-name={job_name}
#SBATCH --partition=forli
#SBATCH --exclude=nodea0110,nodea0111,nodea0112,nodea0113,nodea0114,nodea0115,nodea0116,nodea0117,nodea0118,nodea0119,nodea0120,nodeb0417,nodeb0419,nodeb0423,nodeb0425,nodeb0427,nodea0423,nodea0410,nodec0819,nodec0821
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=115:00:00
#SBATCH -o {job_dir}/{job_name}.out
#SBATCH -e {job_dir}/{job_name}.err

source /gpfs/home/itaneja/.bashrc
micromamba activate cosolvkit

python -u /gpfs/home/itaneja/openfold/unbiased_md.py --input_receptor_path={input_receptor_path} --water_model={water_model} --production_steps={production_steps} --output_dir={output_dir}

"""

    return script 


def gen_unbiased_md_bash_script_afpartition(input_receptor_path, water_model, production_steps, output_dir, job_name, job_dir):

    script = f"""#!/bin/sh  
#SBATCH --job-name={job_name}
#SBATCH --partition=alphafold
#SBATCH --exclude=nodeb201,nodeb203,nodeb205,nodeb207,nodeb209,nodeb211,nodeb213,nodeb301,nodeb303,nodeb305,nodeb307,nodeb309,nodeb311,nodeb313,nodeb601,nodeb603,nodeb605,nodeb607,nodeb609,nodeb611,nodeb613
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=115:00:00
#SBATCH -o {job_dir}/{job_name}.out
#SBATCH -e {job_dir}/{job_name}.err

source /gpfs/home/itaneja/.bashrc
micromamba activate cosolvkit

python -u /gpfs/home/itaneja/openfold/unbiased_md.py --input_receptor_path={input_receptor_path} --water_model={water_model} --production_steps={production_steps} --output_dir={output_dir}

"""

    return script 


 

uniprot_ids = os.listdir('/gpfs/home/itaneja/openfold/md_conformation_input')
print(uniprot_ids)

partition_type = 'forli'

#these files are to be ignored because their disulfides are not consistent with corresponding initial prediction made by AF  
files_to_ignore_dict = {'P21589_benchmark': ['cluster_1_idx_1_plddt_93_openmm_refinement.pdb', 'cluster_3_idx_5_plddt_90_openmm_refinement.pdb'], 
                   'P21589_rw': ['cluster_0_idx_1_plddt_66_openmm_refinement.pdb', 'cluster_2_idx_0_plddt_66_openmm_refinement.pdb', 
                                 'cluster_3_idx_3_plddt_68_openmm_refinement.pdb', 'cluster_3_idx_7_plddt_62_openmm_refinement.pdb',
                                 'cluster_7_idx_9_plddt_94_openmm_refinement.pdb', 'cluster_9_idx_1_plddt_70_openmm_refinement.pdb'],
                    'Q53W80_rw': ['cluster_0_idx_18_plddt_95_openmm_refinement.pdb']
                  }

for uniprot_id in uniprot_ids:
    
    if uniprot_id != 'M1VAN7':
        continue 

    print('on uniprot_id: %s' % uniprot_id)

    benchmark_dir = '/gpfs/home/itaneja/openfold/md_conformation_input/%s/benchmark' % uniprot_id
    rw_dir = '/gpfs/home/itaneja/openfold/md_conformation_input/%s/rw' % uniprot_id

    benchmark_pdb_paths = sorted(glob.glob('%s/*openmm_refinement.pdb' % benchmark_dir))
    rw_pdb_paths = sorted(glob.glob('%s/*openmm_refinement.pdb' % rw_dir))

    benchmark_pdb_files = [f[f.rindex('/')+1:] for f in benchmark_pdb_paths]    
    rw_pdb_files = [f[f.rindex('/')+1:] for f in rw_pdb_paths]

    file_info_dict = {}
    file_info_dict['rw'] = {}
    file_info_dict['benchmark'] = {}

    for p in rw_pdb_paths:
        f = p[p.rindex('/')+1:]
        print(f)
        pattern = r'cluster_(\d+).*plddt_(\d+)'
        match = re.search(pattern, f)
        if match:
            cluster_num = int(match.group(1))
            plddt_score = int(match.group(2))
            print(f"Cluster number: {cluster_num}")
            print(f"pLDDT score: {plddt_score}")
            if cluster_num in file_info_dict['rw']:
                file_info_dict['rw'][cluster_num].append((plddt_score, f, p))
            else:
                file_info_dict['rw'][cluster_num] = [(plddt_score, f, p)]
        else:
            file_info_dict['rw']['initial'] = (f,p) 

    for p in benchmark_pdb_paths:
        f = p[p.rindex('/')+1:]
        print(f)
        pattern = r'cluster_(\d+).*plddt_(\d+)'
        match = re.search(pattern, f)
        if match:
            cluster_num = int(match.group(1))
            plddt_score = int(match.group(2))
            print(f"Cluster number: {cluster_num}")
            print(f"pLDDT score: {plddt_score}")
            if cluster_num in file_info_dict['benchmark']:
                file_info_dict['benchmark'][cluster_num].append((plddt_score, f, p))
            else:
                file_info_dict['benchmark'][cluster_num] = [(plddt_score, f, p)]
        else:
            file_info_dict['benchmark']['initial'] = (f,p)


    for cluster_num in file_info_dict['rw']:
        if cluster_num != 'initial':
            file_info_dict['rw'][cluster_num].sort(key=lambda x: x[0], reverse=True)        

    for cluster_num in file_info_dict['benchmark']:
        if cluster_num != 'initial':
            file_info_dict['benchmark'][cluster_num].sort(key=lambda x: x[0], reverse=True)        


    for method in ['rw', 'benchmark']:

        files_to_ignore_key = '%s_%s' % (uniprot_id, method)
        if files_to_ignore_key in files_to_ignore_dict:
            curr_files_to_ignore = files_to_ignore_dict[files_to_ignore_key]
        else:
            curr_files_to_ignore = [] 

        for cluster_num in file_info_dict[method]:

            if cluster_num != 'initial':
                input_receptor_filename = file_info_dict[method][cluster_num][0][1]
                if input_receptor_filename in curr_files_to_ignore:
                    print('skipping cluster %d because file %s is to be ignored' % (cluster_num, input_receptor_filename))
                    continue
                input_receptor_path = file_info_dict[method][cluster_num][0][2]
                output_dir = '/gpfs/home/itaneja/openfold/unbiased_md_output/%s/%s/cluster%d' % (uniprot_id, method, cluster_num)
                os.makedirs(output_dir, exist_ok=True)
                copied_receptor_path = '%s/%s' % (output_dir, input_receptor_filename)
                shutil.copyfile(input_receptor_path, copied_receptor_path)
                job_name = 'c%d_%s' % (cluster_num, method)
            else:
                input_receptor_filename = file_info_dict[method][cluster_num][0]
                if input_receptor_filename in curr_files_to_ignore:
                    print('skipping cluster %d because file %s is to be ignored' % (cluster_num, input_receptor_filename))
                    continue
                input_receptor_path = file_info_dict[method][cluster_num][1]
                output_dir = '/gpfs/home/itaneja/openfold/unbiased_md_output/%s/%s/initial' % (uniprot_id, method)
                os.makedirs(output_dir, exist_ok=True)
                copied_receptor_path = '%s/%s' % (output_dir, input_receptor_filename)
                shutil.copyfile(input_receptor_path, copied_receptor_path)
                job_name = 'i_%s' % method

            water_model = 'tip3p'

            if uniprot_id in ['P62495','M1VAN7']:
                production_steps = 50000000 #100 ns
            elif uniprot_id in ['P01116', 'P02925']:
                production_steps = 250000000 #500 ns
            else:
                production_steps = 125000000 #250 ns

            job_dir = output_dir

            if partition_type == 'forli':
                script_str = gen_unbiased_md_bash_script_forlipartition(copied_receptor_path, water_model, production_steps, output_dir, job_name, job_dir) 
            elif partition_type == 'alphafold':
                script_str = gen_unbiased_md_bash_script_afpartition(copied_receptor_path, water_model, production_steps, output_dir, job_name, job_dir) 

 
            script_path = '%s/run_md.sh' % (output_dir)
            with open(script_path, 'w') as f:
                f.write(script_str)
            os.chmod(script_path, 0o755)

            print(script_path)
            #if method == 'benchmark':
            submit_job(script_path)




    
