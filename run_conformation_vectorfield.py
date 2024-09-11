import argparse
import logging
import math
import numpy as np
import pandas as pd 
import os
import shutil
import json
from collections import Counter
import re
import glob  
import sys
from datetime import date
import itertools
import time 

from openfold.utils.script_utils import parse_fasta, prep_output, \
    update_timings, relax_protein
from openfold.model.conformation_vectorfield_model import ConformationVectorField


import subprocess 
import pickle

import random
import torch
from torch import nn

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args
import pandas as pd 
pd.set_option('display.max_rows', 500)

from custom_openfold_utils.pdb_utils import get_ca_coords_matrix, save_ca_coords, get_cif_string_from_pdb
from rw_helper_functions import write_timings, remove_files, calc_disordered_percentage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./run_conformation_vectorfield.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def load_conformation_vectorfield(config, model_device, conformation_vectorfield_checkpoint_path):
 
    model = ConformationVectorField(config)
    model = model.eval()
   
    ckpt_path = conformation_vectorfield_checkpoint_path
    d = torch.load(ckpt_path)
    sd = d["state_dict"]
    sd = {k.replace('model.',''):v for k,v in sd.items()}
    import_openfold_weights_(model=model, state_dict=sd)
    logger.info(
        f"Loaded ConformationVectorField parameters at {ckpt_path}..."
    )
    model = model.to(model_device)

    return model


def eval_model(model, args, config, feature_dict, alt_conformation_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    alt_conformation_ca_coords = get_ca_coords_matrix(alt_conformation_path)
 
    with torch.no_grad():
        t = time.perf_counter()
        out = model(feature_dict)
        inference_time = time.perf_counter() - t

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    vectorfield_gt_feats = tensor_tree_map(lambda x: np.array(x), vectorfield_gt_feats)

    # [N,2]
    norm_phi_pred = out["normalized_phi_theta"][..., 0, :]
    norm_theta_pred = out["normalized_phi_theta"][..., 1, :]

    #phi is between 0 and Pi, so y-val is between 0 and 1
    norm_phi_pred[:,1] = np.abs(norm_phi_pred[:,1]) 

    #convert normalized phi/theta to radians 
    raw_phi_pred = np.arctan2(np.clip(norm_phi_pred[..., 1], a_min=-1, a_max=1),np.clip(norm_phi_pred[..., 0], a_min=-1,a_max=1))
    raw_theta_pred = np.arctan2(np.clip(norm_theta_pred[..., 1], a_min=-1, a_max=1),np.clip(norm_theta_pred[..., 0], a_min=-1,a_max=1))

    delta_x_pred = 1.0*np.cos(raw_theta_pred)*np.sin(raw_phi_pred)
    delta_y_pred = 1.0*np.sin(raw_theta_pred)*np.sin(raw_phi_pred)
    delta_z_pred = 1.0*np.cos(raw_phi_pred)

    delta_xyz_pred = np.transpose(np.array([delta_x_pred,delta_y_pred,delta_z_pred]))
    ca_coords_pred = rw_conformation_ca_coords + delta_xyz_pred

    alt_conformation_name = alt_conformation_path.split('/')[-1].split('.')[0]
    output_name = '%s-CVF.pdb' % alt_conformation_name
    pdb_output_path = '%s/%s' % (output_dir, output_name)

    save_ca_coords(alt_conformation_path, ca_coords_pred, pdb_output_path)

    out_df = pd.DataFrame({'raw_phi_pred': raw_phi_pred, 'raw_theta_pred' raw_theta_pred,
                           'norm_phi_pred_x': norm_phi_pred[:,0], 'norm_phi_pred_y': norm_phi_pred[:,1], 
                           'norm_theta_pred_x': norm_theta_pred[:,0], 'norm_theta_pred_y': norm_theta_pred[:,1],
                           'rw_conformation_path': rw_conformation_path}) 

    out_df.to_csv('%s/phi_theta_pred.csv' % output_dir, index=False)
 
    


def eval_model(model, args, config, feature_dict, alt_conformation_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    alt_conformation_ca_coords = get_ca_coords_matrix(alt_conformation_path)
 
    with torch.no_grad():
        t = time.perf_counter()
        out = model(feature_dict)
        inference_time = time.perf_counter() - t

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    vectorfield_gt_feats = tensor_tree_map(lambda x: np.array(x), vectorfield_gt_feats)

    # [N,2]
    norm_phi_pred = out["normalized_phi_theta"][..., 0, :]
    norm_theta_pred = out["normalized_phi_theta"][..., 1, :]

    #phi is between 0 and Pi, so y-val is between 0 and 1
    norm_phi_pred[:,1] = np.abs(norm_phi_pred[:,1]) 

    #convert normalized phi/theta to radians 
    raw_phi_pred = np.arctan2(np.clip(norm_phi_pred[..., 1], a_min=-1, a_max=1),np.clip(norm_phi_pred[..., 0], a_min=-1,a_max=1))
    raw_theta_pred = np.arctan2(np.clip(norm_theta_pred[..., 1], a_min=-1, a_max=1),np.clip(norm_theta_pred[..., 0], a_min=-1,a_max=1))

    norm_phi_gt = np.squeeze(vectorfield_gt_feats["normalized_phi_theta_gt"][..., 0, :], axis=-1)
    norm_theta_gt = np.squeeze(vectorfield_gt_feats["normalized_phi_theta_gt"][..., 1, :], axis=-1)
    r_gt = vectorfield_gt_feats["r_gt"]

    raw_phi_gt = vectorfield_gt_feats["raw_phi_theta_gt"][..., 0, :]
    raw_theta_gt = vectorfield_gt_feats["raw_phi_theta_gt"][..., 1, :]

    #angle between two vectors in spherical coordinates where
    #phi is inclination angle (from positive z-axis)
    #and theta is azimuthal angle from x-axis in xy plane:
    ##sin(phi_1)*sin(phi_2)*cos(theta_1-theta_2) + cos(phi_1)*cos(phi_2)
    sin_phi_prod = np.sin(raw_phi_pred)*np.sin(raw_phi_gt)
    cos_theta_diff = np.cos(raw_theta_pred-raw_theta_gt)
    cos_phi_prod = np.cos(raw_phi_pred)*np.cos(raw_phi_gt)
    vector_angle = np.arccos(np.clip(sin_phi_prod*cos_theta_diff + cos_phi_prod, a_min=-1, a_max=1))

    delta_x_pred = 1.0*np.cos(raw_theta_pred)*np.sin(raw_phi_pred)
    delta_y_pred = 1.0*np.sin(raw_theta_pred)*np.sin(raw_phi_pred)
    delta_z_pred = 1.0*np.cos(raw_phi_pred)

    delta_x_gt = r_gt*np.cos(raw_theta_gt)*np.sin(raw_phi_gt)
    delta_y_gt = r_gt*np.sin(raw_theta_gt)*np.sin(raw_phi_gt)
    delta_z_gt = r_gt*np.cos(raw_phi_gt)

    delta_xyz_pred = np.transpose(np.array([delta_x_pred,delta_y_pred,delta_z_pred]))
    delta_xyz_gt = np.transpose(np.array([delta_x_gt,delta_y_gt,delta_z_gt]))

    ca_coords_pred = rw_conformation_ca_coords + delta_xyz_pred
    ca_coords_gt = rw_conformation_ca_coords + delta_xyz_gt

    rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
    output_name = '%s-CVF.pdb' % rw_conformation_name
    pdb_output_path = '%s/%s' % (output_dir, output_name)

    save_ca_coords(rw_conformation_path, ca_coords_pred, pdb_output_path)

    residues_mask = np.squeeze(vectorfield_gt_feats["residues_mask"], axis=-1)

    out_df = pd.DataFrame({'norm_phi_pred_x': norm_phi_pred[:,0], 'norm_phi_pred_y': norm_phi_pred[:,1], 
                           'norm_phi_gt_x': norm_phi_gt[:,0], 'norm_phi_gt_y': norm_phi_gt[:,1],
                           'norm_theta_pred_x': norm_theta_pred[:,0], 'norm_theta_pred_y': norm_theta_pred[:,1],
                           'norm_theta_gt_x': norm_theta_gt[:,0], 'norm_theta_gt_y': norm_theta_gt[:,1],
                           'vector_angle': vector_angle, 
                           'residues_mask': residues_mask, 'rw_conformation_path': rw_conformation_path, 
                           'nearest_aligned_gtc_path': nearest_aligned_gtc_path})  
    out_df_subset = out_df[out_df['residues_mask'] == 1]
    rel_cols = list(out_df.columns[0:-3])
    print(out_df_subset[rel_cols])

    
    return out_df


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)
    model = load_conformation_vectorfield(config, args.model_device, args.conformation_vectorfield_checkpoint_path)
    
    seq_embedding_dict = pickle.load(args.seq_embeddings_path)
    seq_id = seq_embeddings_dict.keys()[0]
    seq = seq_embeddings_dict[seq_id][0]
    seq_embedding = {"s": seq_embeddings_dict[seq_id][1]}

    if args.alt_conformation_data_dir and args.alt_conformation_path:
        raise ValueError("Only one of alt_conformation_data_dir/alt_conformation_path should be set")
    elif args.alt_conformation_data_dir:
        all_alt_conformations_paths =  glob.glob('%s/*.pdb' % args.alt_conformation_data_dir)
        if len(all_alt_conformations_path) == 0:
            all_alt_conformations_paths =  glob.glob('%s/*.cif' % args.alt_conformation_data_dir)
        if len(all_alt_conformations_paths) == 0:
            raise ValueError('No pdb or cif files found in %s' % args.alt_conformation_data_dir)
    elif args.alt_conformation_path:
        all_alt_conformations_paths = [args.alt_conformation_path]

    
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=0,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path
    )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    tensor_feature_names = config.common.unsupervised_features
    tensor_feature_names += config.common.template_features
    tensor_feature_names += ['template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask']

    for alt_conformation_path in all_alt_conformations_paths: 

        alt_conformation_fname = alt_conformation_path.split('/')[-1].split('.')[0]
        alt_conformation_cif_string = get_cif_string_from_pdb(alt_conformation_path) 
        conformation_feats = data_processor.process_conformation(
            cif_string=alt_conformation_cif_string,
            file_id=alt_conformation_fname,
            tensor_feature_names=tensor_feature_names
        )

        seq_features = data_pipeline.make_sequence_features(seq, seq_id, len(seq)) 
        to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
        seq_features = {
            k: to_tensor(v).squeeze(0) for k, v in seq_features.items() if k in tensor_feature_names 
        }

        feature_dict = {**conformation_feats, **seq_features, **seq_embedding}

        eval_model(model, args, config, feature_dict, alt_conformation_path, args.output_dir_base)


    ########################

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    children_dirs = glob.glob('%s/*/' % args.alignment_dir) #UNIPROT_ID
    children_dirs = [f[0:-1] for f in children_dirs] #remove trailing forward slash
    unique_uniprot_ids = [f[f.rindex('/')+1:] for f in children_dirs] #extract UNIPROT_ID 
    rw_conformations = [] #path to rw_conformations across all uniprot_ids 
    uniprot_ids = [] #corresponding uniprot_id for each rw_conformation
    for uniprot_id in unique_uniprot_ids:
        rw_data_dir_curr_uniprot_id = os.path.join(args.rw_data_dir, uniprot_id)
        if os.path.exists(rw_data_dir_curr_uniprot_id):
            rw_conformations_curr_uniprot_id = glob.glob('%s/*/*/*/*/bootstrap/ACCEPTED/*.pdb' % rw_data_dir_curr_uniprot_id)
            rw_conformations.extend(rw_conformations_curr_uniprot_id)
            uniprot_ids.extend([uniprot_id]*len(rw_conformations_curr_uniprot_id))

    conformation_vectorfield_path = os.path.join(args.ground_truth_data_dir, 'conformation_vectorfield_dict.pkl')
    residues_mask_path = os.path.join(args.ground_truth_data_dir, 'residues_mask_dict.pkl')
   
    with open(conformation_vectorfield_path, 'rb') as f:
        conformation_vectorfield_dict = pickle.load(f) 
    with open(residues_mask_path, 'rb') as f:
        residues_mask_dict = pickle.load(f) 

    out_df_all = [] 

    for i,rw_conformation_path in enumerate(rw_conformations):
        
        rw_conformation_path = os.path.abspath(rw_conformation_path)

        logger.info('rw_conformation_path: %s' % rw_conformation_path)
        uniprot_id = uniprot_ids[i]

        match = re.search(r'template=(\w+)', rw_conformation_path)
        template_pdb_id = match.group(1) #this is used as the template and the MSA is derived from this PDB_ID
        alignment_dir = '%s/%s/%s' % (args.alignment_dir, uniprot_id, template_pdb_id)

        pattern = "%s/*.fasta" % alignment_dir
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            fasta_file = files[0]
        else: 
            raise FileNotFoundError("Either >1 or 0 .fasta files found in alignment_dir -- should only be one")

        with open(fasta_file, "r") as fp:
            fasta_data = fp.read()
        _, seq = parse_fasta(fasta_data)
        seq = seq[0]
        logger.info("PROTEIN SEQUENCE:")
        logger.info(seq)

        rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_name = rw_conformation_name.replace('_unrelaxed','')
        rw_conformation_name = rw_conformation_name.replace('_relaxed','')
        rw_conformation_parent_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]
        rw_conformation_input = '%s/structure_module_intermediates/%s_sm_output_dict.pkl' % (rw_conformation_parent_dir, rw_conformation_name)

        with open(rw_conformation_input, 'rb') as f:
            conformation_module_input = pickle.load(f) 

        residues_mask = residues_mask_dict[rw_conformation_path]
        conformation_vectorfield_spherical_coords, nearest_aligned_gtc_path, nearest_pdb_model_name = conformation_vectorfield_dict[rw_conformation_path]

        logger.info('nearest_aligned_gtc_path: %s' % nearest_aligned_gtc_path)

        seq_features = data_pipeline.make_sequence_features(seq, template_pdb_id, len(seq)) 
        seq_features = feature_processor.process_features(
            seq_features, 'predict'
        )

        phi = conformation_vectorfield_spherical_coords[1]
        theta = conformation_vectorfield_spherical_coords[2]
        r = conformation_vectorfield_spherical_coords[3]

        phi_norm = np.stack((np.cos(phi), np.sin(phi)), axis=-1) # [N,2]
        theta_norm = np.stack((np.cos(theta), np.sin(theta)), axis=-1) # [N,2]
        normalized_phi_theta = np.stack((phi_norm,theta_norm), axis=-1) # [N,2,2]
        raw_phi_theta = np.stack((phi,theta), axis=-1) # [N,2]

        vectorfield_gt_feats = {} 
        vectorfield_gt_feats['normalized_phi_theta_gt'] = torch.from_numpy(normalized_phi_theta).to(torch.float32)
        vectorfield_gt_feats['raw_phi_theta_gt'] = torch.from_numpy(raw_phi_theta).to(torch.float32)
        vectorfield_gt_feats['r_gt'] = torch.from_numpy(r).to(torch.float32)
        vectorfield_gt_feats['residues_mask'] = torch.tensor(residues_mask, dtype=torch.int)
      
        vectorfield_gt_feats = {k: torch.unsqueeze(v, dim=-1) for k, v in vectorfield_gt_feats.items()} 

        conformation_module_input['single'] = torch.unsqueeze(conformation_module_input['single'], dim=-1)
        conformation_module_input['rigid_rotation'] = torch.unsqueeze(conformation_module_input['rigid_rotation'], dim=-1) #add recycling dimension
        conformation_module_input['rigid_translation'] = torch.unsqueeze(conformation_module_input['rigid_translation'], dim=-1) #add recycling dimension


        feature_dict = {**seq_features, **conformation_module_input}

        output_dir = '%s/conformation_vectorfield_predictions/gtc=%s' % (rw_conformation_parent_dir, nearest_pdb_model_name)
        out_df = eval_model(model, args, config, feature_dict, vectorfield_gt_feats, rw_conformation_path, nearest_aligned_gtc_path, output_dir)
        out_df_all.append(out_df)

    output_dir = '%s/conformation_vectorfield_predictions' % rw_conformation_parent_dir
    os.makedirs(output_dir, exist_ok=True)
    out_df_all = pd.concat(out_df_all)
    out_df_all.to_csv('%s/phi_norm_theta_pred.csv' % output_dir, index=False)

           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_data_dir", type=str, default = None, 
        help="Directory containing training mmCIF files -- corresponds two conformations 1 and 2."
    )
    parser.add_argument(
        "--alt_conformation_data_dir", type=str, default = None, 
        help="Directory containing files corresponding to alternative conformations"
    )
    parser.add_argument(
        "--alt_conformation_path", type=str, default = None,
        help="Path to alternative conformation"
    )
    parser.add_argument(
        "--seq_embeddings_path", type=str, default = None
    )
    
    parser.add_argument(
        "--alignment_dir", type=str, required=True,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--output_dir_base", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default="conformation_vectorfield",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--conformation_vectorfield_checkpoint_path", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=1,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )



    add_data_args(parser)
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)

