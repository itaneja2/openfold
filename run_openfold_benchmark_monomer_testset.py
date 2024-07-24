import argparse
from argparse import Namespace
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
import ml_collections as mlc
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union

from openfold.utils.script_utils import load_model_w_intrinsic_param, parse_fasta, run_model_w_intrinsic_dim, prep_output, \
    update_timings, relax_protein

import subprocess 
import pickle

import random
import torch
from torch import nn

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

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

from random_corr_sap import gen_randcorr_sap
from custom_openfold_utils.pdb_utils import align_and_get_rmsd
import rw_helper_functions

from run_openfold_benchmark_monomer import (
    run_msa_mask 
)

FeatureDict = MutableMapping[str, np.ndarray]

logger = logging.getLogger('run_openfold_rw_monomer_testset')
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./rw_monomer_testset.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

finetune_openfold_path = './finetune_openfold.py'

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def gen_args(alignment_dir, output_dir_base, seed, use_templates=True):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark_method", type=str, 
    )
    parser.add_argument(
        "--use_templates", type=bool
    )
    parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--alignment_dir", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
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
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--msa_mask_fraction", type=float, default=0.15
    )
    parser.add_argument(
        "--num_predictions_per_model", type=int, default=10
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

    args.benchmark_method = 'msa_mask'
    args.template_mmcif_dir = '/dev/shm/pdb_mmcif/mmcif_files'
    args.use_templates = use_templates 
    args.alignment_dir = alignment_dir
    args.output_dir_base = output_dir_base 
    args.config_preset = 'model_1_ptm'
    args.openfold_checkpoint_path = '/opt/databases/openfold/openfold_params/finetuning_ptm_2.pt'
    args.model_device = 'cuda:0'
    args.data_random_seed = seed 
    args.msa_mask_fraction = 0.20 
    args.num_predictions_per_model = 300 
        
    if(args.jax_param_path is None and args.openfold_checkpoint_path is None):
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    return args  



def restart_incomplete_iterations(output_dir, args):

    should_run_rw = True 

    pdb_files = glob.glob('%s/*.pdb' % output_dir)    
    if len(pdb_files) != (args.num_predictions_per_model+1): #incomplete job -- +1 for initial_pred 
        if len(pdb_files) > 0:
            logger.info('removing %d pdb files in %s' % (len(pdb_files),output_dir))
            rw_helper_functions.remove_files(pdb_files)
    else:
        should_run_rw = False 

    return should_run_rw 


def run_benchmark_af2sample_dataset():

    conformational_states_df = pd.read_csv('./conformational_states_testing_data/dataset/conformational_states_testing_data_processed_adjudicated.csv')
    
    conformational_states_df = conformational_states_df[conformational_states_df['any_structures_present_in_training_set'] == 'n']
    conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'].isin(['Q53W80'])]
    conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 

    for index,row in conformational_states_df.iterrows():

        logger.info('On row %d of %d' % (index, len(conformational_states_df)))  
        logger.info(asterisk_line)
        logger.info(row)
 
        uniprot_id = str(row['uniprot_id'])
        pdb_id_msa = str(row['pdb_id_msa'])

        for j in [1]:

            if j == 0:
                use_templates = True
                template_str = 'template=default' 
            else:
                use_templates = False
                template_str = 'template=none' 
            
            alignment_dir = './conformational_states_testing_data/alignment_data/%s/%s' % (uniprot_id,pdb_id_msa)
            seed = index #keep seed constant per uniprot_id  
            logger.info(asterisk_line)
            logger.info('TEMPLATE = %s' % template_str)
            logger.info('SEED = %d' % seed) 
            logger.info(asterisk_line)
            output_dir_base = './conformational_states_testing_data/benchmark_predictions/%s' % uniprot_id 
            args = gen_args(alignment_dir, output_dir_base, seed, use_templates)
            output_dir = '%s/msa_mask/%s' % (output_dir_base, template_str)
            should_run_rw = restart_incomplete_iterations(output_dir, args)
            if should_run_rw:
                logger.info("RUNNING %s" % output_dir)
                run_msa_mask(args)
            else:
                logger.info("SKIPPING %s BECAUSE ALREADY EVALUATED" % output_dir) 


def run_benchmark_custom_dataset():


    conformational_states_df = pd.read_csv('./conformational_states_dataset/dataset/conformational_states_filtered_adjudicated_post_AF_training_final.csv')

    conformational_states_df = conformational_states_df[conformational_states_df['any_structures_present_in_training_set'] == 'n']
    conformational_states_df = conformational_states_df.sort_values('seg_len').reset_index(drop=True) 

    for index,row in conformational_states_df.iterrows():

        logger.info('On row %d of %d' % (index, len(conformational_states_df)))  
        logger.info(asterisk_line)
        logger.info(row)
 
        uniprot_id = str(row['uniprot_id'])
        pdb_id_msa = str(row['pdb_id_outside_training_set'])
        if pdb_id_msa == 'both':
            pdb_id_msa = str(row['pdb_id_ref'])

        for j in [1]:

            if j == 0:
                use_templates = True
                template_str = 'template=default' 
            else:
                use_templates = False
                template_str = 'template=none' 
            
            alignment_dir = './conformational_states_testing_data/alignment_data/%s/%s' % (uniprot_id,pdb_id_msa)
            seed = index #keep seed constant per uniprot_id  
            logger.info(asterisk_line)
            logger.info('TEMPLATE = %s' % template_str)
            logger.info('SEED = %d' % seed) 
            logger.info(asterisk_line)
            output_dir_base = './conformational_states_testing_data/benchmark_predictions/%s' % uniprot_id 
            args = gen_args(alignment_dir, output_dir_base, seed, use_templates)
            output_dir = '%s/msa_mask/%s' % (output_dir_base, template_str)
            should_run_rw = restart_incomplete_iterations(output_dir, args)
            if should_run_rw:
                logger.info("RUNNING %s" % output_dir)
                run_msa_mask(args)
            else:
                logger.info("SKIPPING %s BECAUSE ALREADY EVALUATED" % output_dir) 



run_benchmark_af2sample_dataset() 
