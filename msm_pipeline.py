import argparse
import numpy as np
import glob
import pandas as pd
import os
import subprocess
import re  
import shutil
import pickle
import argparse
from functools import reduce
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import self_distance_array
from MDAnalysis.analysis import rms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from custom_openfold_utils.pdb_utils import superimpose_wrapper_monomer


asterisk_line = '******************************************************************************'

path_to_msm_clustering_exe = '/gpfs/home/itaneja/openfold/Clustering/build/clustering' 

def remove_files_in_dir(path):
    file_list = sorted(glob.glob('%s/*' % path))
    for f in file_list:
        if not os.path.isdir(f):
            print('removing old file: %s' % f)
            os.remove(f)


def get_msm_input_data(input_prmtop_path, input_traj_path, output_dir, max_frames=None, initial_pred_path=None, write_reimaged_traj=False):

    os.makedirs(output_dir, exist_ok=True)

    print('reimaging trajectory %s' % input_traj_path)

    whole_system_u = mda.Universe(input_prmtop_path, input_traj_path, select='protein and not resname HOH')

    num_frames = len(whole_system_u.trajectory)
    print('total frames %d' % num_frames)
    timestep_ns = whole_system_u.trajectory.dt/1000

    if max_frames is not None:
        if max_frames <= num_frames:
            num_frames = max_frames
        else:
            print('ERROR: max frames greater than num frames')
            sys.exit()

    protein = whole_system_u.select_atoms('protein')

    protein_coords = np.empty((num_frames, protein.atoms.n_atoms, 3))
    boxes = np.empty((num_frames, 6))
    for ts in whole_system_u.trajectory[0:num_frames]:
        boxes[ts.frame] = ts.dimensions
        protein_coords[ts.frame] = protein.atoms.positions
    boxes = boxes[0]

    print('protein coordinates shape')
    print(protein_coords.shape)

    protein_u = mda.Merge(protein.atoms).load_new(protein_coords, dimensions=boxes)
    protein_u.add_TopologyAttr('tempfactors')

    #these transformations are lazy
    transforms = [
        trans.unwrap(protein_u.atoms),
        trans.center_in_box(protein_u.atoms, center='mass'),
        trans.wrap(protein_u.atoms, compound='fragments'),
    ]

    print('reimaging protein')
    protein_u.trajectory.add_transformations(*transforms)

    protein_u.add_TopologyAttr("chainID")
    for atom in protein_u.atoms:
        atom.chainID = 'A'


    rmsd_df = [] 

    if initial_pred_path is not None:
        print('aligning to %s' % initial_pred_path)
        ref = mda.Universe(initial_pred_path)
        ref_protein = ref.select_atoms("protein")
        aligner = align.AlignTraj(protein_u, ref_protein, select='name CA', in_memory=True)
        aligner.run() 

        #rmsd and rmsf calculated w.r.t initial_pred_path 
        print('calculating rsmd')
        rmsd_calc = rms.RMSD(protein_u, ref_protein, select='name CA', ref_frame=0)
        rmsd_calc.run()
        rmsd_results = rmsd_calc.results.rmsd 
 
        print('calculating rmsf')
        mobile = protein_u.select_atoms('name CA')
        rmsf_calc = rms.RMSF(mobile).run()
        rmsf_results = rmsf_calc.results.rmsf
        rmsf_dict = dict(zip(mobile.resids, rmsf_results))

        cluster_num = output_dir[output_dir.rindex('/')+1:]

        aligned_frames_save_dir = '%s/aligned_frames_wrt_initial_pred' % output_dir
        os.makedirs(aligned_frames_save_dir, exist_ok=True)
        remove_files_in_dir(aligned_frames_save_dir)
        
        for ts in protein_u.trajectory: 
            frame_num = ts.frame 
            rmsd = rmsd_results[frame_num][2]
            rmsd_str = str(round(rmsd,2)).replace('.','-')        
            output_pdb_path = '%s/frame%d-rmsd_wrt_initial-%s.pdb' % (aligned_frames_save_dir, frame_num, rmsd_str)
            protein_u.atoms.tempfactors = 0.0
            for residue in protein_u.residues:
                ca_atom = residue.atoms.select_atoms('name CA')
                if len(ca_atom) > 0:
                    residue.atoms.tempfactors = rmsf_dict[residue.resid] 
            if frame_num % 10 == 0:
                print('saving %s' % output_pdb_path)
            protein_u.atoms.write(output_pdb_path)
            rmsd_df.append([frame_num, cluster_num, initial_pred_path, output_pdb_path, round(rmsd,3)])
 
    if write_reimaged_traj:
        print('writing reimaged trajectory')
        output_traj_path = '%s/trajectory-centered.nc' % output_dir
        with mda.Writer(output_traj_path, protein_u.atoms.n_atoms) as W:
            for ts in protein_u.trajectory:
                if ts.frame == 0:
                    box_dimensions = ts.dimensions
                    print(f"Box dimensions (a, b, c, alpha, beta, gamma) = {box_dimensions}")
                if ts.frame % 100 == 0:
                    print(f"Writing frame {ts.frame}/{num_frames}")
                W.write(protein_u.atoms)

    print('calculating ca pairwise distances and contacts')
    ca_atoms = protein_u.select_atoms('name CA')
    num_atoms = len(ca_atoms)

    ca_pdist = np.zeros((num_frames, num_atoms * (num_atoms - 1) // 2))

    for ts in protein_u.trajectory:
        pdist = self_distance_array(ca_atoms.positions) 
        ca_pdist[ts.frame] = pdist

    print('pairwise distance matrix shape:')
    print(ca_pdist.shape)

    ca_contacts = (ca_pdist <= 8.0).astype(int)

    #https://www.nature.com/articles/s41467-024-53170-z#Sec9
    #####contacts with a peak-to-peak value (range) versus mean ratio of less than 0.2 are considered static
    dist_ratio = np.ptp(ca_pdist, axis=0) / (np.mean(ca_pdist, axis=0) + 1e-10)
    static_contacts_idx = np.where(dist_ratio < .2)[0] 

    print('%d static contacts found out of %d total contacts' % (len(static_contacts_idx), ca_contacts.shape[1]))
    
    protein_u.trajectory[0]
    output_pdb_path = '%s/protein-reimaged.pdb' % output_dir
    print('saving %s' % output_pdb_path)
    protein_u.atoms.write(output_pdb_path)


    return ca_pdist, ca_contacts, static_contacts_idx, rmsd_df, num_frames, timestep_ns 


def run_pca(input_data, output_dir, output_fname):

    print('Running PCA')
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    pca = PCA(n_components=10)
    pc_coords = pca.fit_transform(input_data_scaled)
    
    pca_df = pd.DataFrame({'pc1': pc_coords[:,0], 'pc2': pc_coords[:,1], 'pc3': pc_coords[:,2]})
    explained_variance_df = pd.DataFrame({
       'component_num': [f'PC{i+1}' for i in range(pca.n_components_)],
       'explained_var': pca.explained_variance_ratio_
    })

    print('explained variance:')
    print(explained_variance_df)

    explained_var_path = '%s/%s_explained_var.csv' % (output_dir, output_fname)
    print('saving %s' % explained_var_path)
    explained_variance_df.to_csv(explained_var_path, index=False)

    pca_output_path = '%s/%s_numpc=3' % (output_dir, output_fname)
    print('saving %s' % pca_output_path)
    pca_df.to_csv(pca_output_path, sep=' ', index=False, header=False)

    return pca_output_path 


def run_command(command):

    result = subprocess.run(command, capture_output=True, text=True, check=True)
 
    print("Output:")
    print(result.stdout)
    
    if result.stderr:
        print("Warnings/Errors:")
        print(result.stderr)

def run_all_commands(pca_data_path, num_frames, num_trajectories, lag_steps, min_population, output_dir):

    os.chdir(output_dir) 
    
    free_energy_output_path = '%s/free_energy' % output_dir
    nearest_neighbor_output_path = '%s/nearest_neighbor' % output_dir 
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-pop",               
        "-d", free_energy_output_path,
        "-b", nearest_neighbor_output_path,
        "-v"              
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    cluster_info_output_dir = '%s/cluster_info' % output_dir
    os.makedirs(cluster_info_output_dir, exist_ok=True)
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-T", "-1",
        "-D", free_energy_output_path,
        "-B", nearest_neighbor_output_path,
        "-o", '%s/cluster' % cluster_info_output_dir,
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    print('changing directory to: %s' % cluster_info_output_dir)
    os.chdir(cluster_info_output_dir)
    command = [
        path_to_msm_clustering_exe,
        "network",
        "-p", str(min_population), 
        "-b", "cluster",
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)

    os.chdir(output_dir)
    microstate_traj_output_dir = '%s/microstate_traj' % output_dir 
    os.makedirs(microstate_traj_output_dir, exist_ok=True)
    microstates_output_path = '%s/microstates_pc3_minpopulation=%d' % (microstate_traj_output_dir, min_population)
    command = [
        path_to_msm_clustering_exe,
        "density",             
        "-f", pca_data_path,
        "-i", '%s/network_end_node_traj.dat' % cluster_info_output_dir,
        "-D", free_energy_output_path,
        "-B", nearest_neighbor_output_path,
        "-o", microstates_output_path,
        "-v" 
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)


    mpp_output_path = '%s/microstates_pc3_minpopulation=%d_lagsteps=%d' % (microstate_traj_output_dir, min_population, lag_steps) 
    command = [
        path_to_msm_clustering_exe,
        "mpp",             
        "-s", microstates_output_path,
        "-l", str(lag_steps), 
        "-D", free_energy_output_path,
        "-o", mpp_output_path,
        "--concat-nframes", str(num_frames), 
        "--concat-limits", str(num_trajectories),
        "-v"
    ]
    command_str = " ".join(command)
    print('running %s' % command_str)
    run_command(command)


def run_pipeline(uniprot_id, traj_paths, initial_pred_path, save_dir, exp_pdb_ids=[], max_frames=None):

    ca_pdist_all_traj = []
    ca_contacts_all_traj = []
    rmsd_df_all_traj = []
    static_contacts_idx_all_traj = []

    for i in range(0,len(traj_paths)):

        print('on path %d/%d' % (i,len(traj_paths)))
        print('loading %s' % traj_paths[i])

        input_traj_path = traj_paths[i]
        input_prmtop_path = input_traj_path.replace('trajectory.nc', 'minimization_round2.prmtop')
        output_dir = input_traj_path[0:input_traj_path.rindex('/')] 

        ca_pdist, ca_contacts, static_contacts_idx, rmsd_df, num_frames, timestep_ns = get_msm_input_data(input_prmtop_path, input_traj_path, output_dir, max_frames=max_frames, initial_pred_path=initial_pred_path)
        prev_num_frames = num_frames
        
        static_contacts_idx_all_traj.append(static_contacts_idx)
        rmsd_df_all_traj.extend(rmsd_df)
                
        if i == 0:
            ca_pdist_all_traj.append(ca_pdist)
            ca_contacts_all_traj.append(ca_contacts)
        else:
            if prev_num_frames != num_frames:
                print('WARNING: %s does not match number of frames (%d vs %d) from previous trajectory, so ignoring from analysis' % (input_traj_path, prev_num_frames, num_frames))
            else:
                ca_pdist_all_traj.append(ca_pdist)
                ca_contacts_all_traj.append(ca_contacts)
     
    
    if len(rmsd_df_all_traj) > 0:
        rmsd_df_all_traj = pd.DataFrame(rmsd_df_all_traj, columns = ['frame_num', 'cluster_num', 'initial_pred_path', 'md_frame_path', 'rmsd_wrt_initial_pred'])
        rmsd_df_all_traj.insert(0, 'uniprot_id', uniprot_id)
        print(rmsd_df_all_traj)
        msm_output_dir = '%s/msm_pipeline_output' % save_dir
        if os.path.exists(msm_output_dir):
            shutil.rmtree(msm_output_dir)
        os.makedirs(msm_output_dir, exist_ok=True)
        output_path = '%s/rmsd_df.csv' % msm_output_dir 
        print('saving %s' % output_path)
        rmsd_df_all_traj.to_csv(output_path, index=False) 


    for exp_pdb_id in exp_pdb_ids:
        rmsd_list = [] 
        for idx,md_frame_path in enumerate(list(rmsd_df_all_traj['md_frame_path'])):
            frame_num = list(rmsd_df_all_traj['frame_num'])[idx]
            if frame_num == 0:
                print('aligning %s to %s' % (md_frame_path, exp_pdb_id))
                aligned_frames_wrt_exp_pdb_save_dir = md_frame_path[0:md_frame_path.rindex('/')]
                aligned_frames_wrt_exp_pdb_save_dir = aligned_frames_wrt_exp_pdb_save_dir.replace('aligned_frames_wrt_initial_pred', 'aligned_frames_wrt_%s' % exp_pdb_id)
                os.makedirs(aligned_frames_wrt_exp_pdb_save_dir, exist_ok=True)
                remove_files_in_dir(aligned_frames_wrt_exp_pdb_save_dir)
            else:
                print('aligning %s to %s' % (md_frame_path, exp_pdb_path)) 
            if frame_num == 0:
                exp_pdb_path = None 
                rmsd, _, exp_pdb_path, md_frame_path_aligned = superimpose_wrapper_monomer(exp_pdb_id, None, 'pdb', 'md', exp_pdb_path, md_frame_path, aligned_frames_wrt_exp_pdb_save_dir, clean=True)
            else:
                #we use cleaned exp_pdb_path from frame_num=0 instead of fetching and cleaning pdb again 
                rmsd, _, _, md_frame_path_aligned = superimpose_wrapper_monomer(exp_pdb_id, None, 'pdb', 'md', exp_pdb_path, md_frame_path, aligned_frames_wrt_exp_pdb_save_dir, clean=False)
            rmsd = round(rmsd,3)
            rmsd_str = str(round(rmsd,2)).replace('.','-')
            md_frame_path_aligned_renamed = '%s/frame%d-rmsd_wrt_%s-%s.pdb' % (aligned_frames_wrt_exp_pdb_save_dir, frame_num, exp_pdb_id, rmsd_str)
            os.rename(md_frame_path_aligned, md_frame_path_aligned_renamed) 
            rmsd_list.append(rmsd)
        colname = 'rmsd_wrt_%s' % exp_pdb_id
        rmsd_df_all_traj[colname] = rmsd_list
    if len(exp_pdb_ids) > 0:
        msm_output_dir = '%s/msm_pipeline_output' % save_dir
        output_path = '%s/rmsd_df.csv' % msm_output_dir 
        print('saving %s' % output_path)
        rmsd_df_all_traj.to_csv(output_path, index=False) 

        
    num_trajectories = len(traj_paths)
    lag_steps = int(10/timestep_ns) #corresponds to lagtime of 10ns  
    min_population = int(num_frames/10) #corresponds to the minimum number of frames per microstate  
 
    #static contacts that exist among all simulations 
    print('calculating common subset of static contacts among all trajectories')
    static_contacts_idx_intersection = reduce(np.intersect1d, static_contacts_idx_all_traj)

    ca_pdist_all_traj = np.array(ca_pdist_all_traj)
    ca_contacts_all_traj = np.array(ca_contacts_all_traj) 

    ca_pdist_all_traj = np.concatenate(ca_pdist_all_traj, axis=0)
    ca_contacts_all_traj = np.concatenate(ca_contacts_all_traj, axis=0)

    print(ca_contacts_all_traj.shape)
    print('removing %d static contacts out of %d total contacts' % (len(static_contacts_idx_intersection), ca_contacts_all_traj.shape[1])) 
    ca_dynamic_contacts_all_traj = np.delete(ca_contacts_all_traj, static_contacts_idx_intersection, axis=1) 
    print(ca_dynamic_contacts_all_traj.shape)
                 
    pca_output_dir = '%s/pca_output' % save_dir
    os.makedirs(pca_output_dir, exist_ok=True)

    pca_info_fname = '%s/pca_info.txt' % pca_output_dir
    with open(pca_info_fname, 'w') as f:
        f.write("shape of ca_pairwise_dist matrix: \n")
        f.write(str(ca_pdist_all_traj.shape))
        f.write("\n")
        f.write("shape of ca_contacts matrix: \n")
        f.write(str(ca_dynamic_contacts_all_traj.shape))
        f.write("\n")
         
    output_fname = 'pca_pdist'
    pca_pdist_path = run_pca(ca_pdist_all_traj, pca_output_dir, output_fname)               
    output_fname = 'pca_ca_contacts'
    pca_contacts_path = run_pca(ca_dynamic_contacts_all_traj, pca_output_dir, output_fname)

    msm_output_dir = '%s/msm_pipeline_output/pca_pdist' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_all_commands(pca_pdist_path, num_frames, num_trajectories, lag_steps, min_population, msm_output_dir)

    msm_output_dir = '%s/msm_pipeline_output/pca_contacts' % save_dir
    if os.path.exists(msm_output_dir):
        shutil.rmtree(msm_output_dir)
    os.makedirs(msm_output_dir, exist_ok=True)
    run_all_commands(pca_contacts_path, num_frames, num_trajectories, lag_steps, min_population, msm_output_dir)




def run_pipeline_wrapper():
    
    uniprot_ids = os.listdir('/gpfs/home/itaneja/openfold/unbiased_md_output')
    print(uniprot_ids)

    #uniprot_ids.remove('Q53W80')
    #print(uniprot_ids)

    for uniprot_id in uniprot_ids:
         
        if uniprot_id != 'Q53W80':
            continue

        #if uniprot_id == 'P71447':
        #    max_frames = 300 
        #else:
        #    max_frames = 400 

        print('ON %s' % uniprot_id)
        print('********')

        conformational_states_df = pd.read_csv('/gpfs/home/itaneja/openfold/afsample2_dataset/afsample2_dataset_processed_adjudicated.csv')
        conformational_states_df_subset = conformational_states_df[conformational_states_df['uniprot_id'] == uniprot_id]
        pdb_id_ref = str(conformational_states_df_subset.iloc[0]['pdb_id_ref'])
        pdb_id_state_i = str(conformational_states_df_subset.iloc[0]['pdb_id_state_i'])

        #exp_pdb_ids = [pdb_id_ref, pdb_id_state_i]
    
        benchmark_dir = '/gpfs/home/itaneja/openfold/unbiased_md_output/%s/benchmark' % uniprot_id
        rw_dir = '/gpfs/home/itaneja/openfold/unbiased_md_output/%s/rw' % uniprot_id

        initial_pred_path = '%s/initial/initial_pred_openmm_refinement.pdb' % rw_dir

        benchmark_traj_paths = sorted(glob.glob('%s/**/trajectory.nc' % benchmark_dir))
        rw_traj_paths = sorted(glob.glob('%s/**/trajectory.nc' % rw_dir))

        for i in [0,1]:

            if i == 0:
                traj_paths = rw_traj_paths
                parent_dir = rw_dir
            else:
                traj_paths = benchmark_traj_paths
                parent_dir = benchmark_dir
    
            run_pipeline(uniprot_id, traj_paths, initial_pred_path, parent_dir)        

            


run_pipeline_wrapper()

 


