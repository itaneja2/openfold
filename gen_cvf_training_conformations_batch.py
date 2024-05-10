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

from random_corr_sap import gen_randcorr_sap
from custom_openfold_utils.pdb_utils import align_and_get_rmsd
import rw_helper_functions

from gen_cvf_training_conformations import (
    run_rw_pipeline 
)

FeatureDict = MutableMapping[str, np.ndarray]

logger = logging.getLogger('gen_cvf_training_conformations_batch')
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./cvf_training_conformations_batch.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

finetune_openfold_path = './finetune_openfold.py'

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def gen_args(template_pdb_id, alignment_dir, output_dir_base, seed):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_file", type=str, default=None,
        help="Path to FASTA file, one sequence per file. By default assumes that .fasta file is located in alignment_dir "
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--custom_template_pdb_id", type=str, default=None,
        help="""String of the format PDB-ID_CHAIN-ID (e.g 4ake_A). If provided,
              this structure is used as the only template."""
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
        "--num_steps", type=int, default=50
    )
    parser.add_argument(
        "--num_hp_tuning_steps", type=int, default=10
    )

    parser.add_argument(
        "--num_rw_steps", type=int, default=100
    )
    parser.add_argument(
        "--num_rw_hp_tuning_steps_per_round", type=int, default=10
    )
    parser.add_argument(
        "--num_rw_hp_tuning_rounds_total", type=int, default=2
    )
    parser.add_argument(
        "--early_stop_rw_hp_tuning", action="store_true", default=False,
    )
    parser.add_argument(
        "--num_training_conformations", type=int, default=5
    )
    parser.add_argument(
        "--save_training_conformations", action="store_true", default=False
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--save_structure_module_intermediates", action="store_true", default=False,
        help="Whether to save s (i.e first row of MSA) and backbone frames representation"
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
        "--module_config", type=str, default='model_config_0',
        help=(
            "module_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--rw_hp_config", type=str, default='hp_config_0',
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--train_hp_config", type=str, default='hp_config_2',
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--use_local_context_manager", action="store_true", default=False,
        help=(
            """whether to use local context manager
             when generating proposals. this means 
             that the same set of intrinsic_param
             will be produced within that context
             block.""" 
            )
    )
    parser.add_argument(
        "--overwrite_pred", action="store_true", default=False
    )
    parser.add_argument(
        "--mean_plddt_threshold", type=int, default=60
    )
    parser.add_argument(
        "--disordered_percentage_threshold", type=int, default=80
    )
    parser.add_argument(
        "--log_level", type=str, default='INFO'
    )

    add_data_args(parser)
    args = parser.parse_args()

    args.template_mmcif_dir = '/dev/shm/pdb_mmcif/mmcif_files'
    args.custom_template_pdb_id = template_pdb_id 
    args.alignment_dir = alignment_dir
    args.output_dir_base = output_dir_base 
    args.save_structure_module_intermediates = True
    args.config_preset = 'model_1_ptm'
    args.openfold_checkpoint_path = '/opt/databases/openfold/openfold_params/finetuning_ptm_2.pt'
    args.module_config = 'module_config_0'
    args.rw_hp_config = 'hp_config_0'
    args.skip_relaxation = True
    args.model_device = 'cuda:0'
    args.data_random_seed = seed 
    args.use_local_context_manager = True 
    args.num_rw_hp_tuning_steps = 5
    args.num_rw_steps = 100 
        
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



def restart_incomplete_iterations(uniprot_id, pdb_id_ref, pdb_id_state_i, args):

    should_run_rw = True 

    template_str = 'template=%s' % pdb_id_ref
    output_dir_base = './conformational_states_training_data/rw_predictions/%s/%s' % (uniprot_id, template_str) 
    output_dir_ref = '%s/%s/%s/rw-%s' % (output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config)
    conformation_info_ref_fname = '%s/training_conformations/conformation_info.pkl' % output_dir_ref

    template_str = 'template=%s' % pdb_id_state_i
    output_dir_base = './conformational_states_training_data/rw_predictions/%s/%s' % (uniprot_id, template_str) 
    output_dir_state_i = '%s/%s/%s/rw-%s' % (output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config)
    conformation_info_ref_state_i_fname = '%s/training_conformations/conformation_info.pkl' % output_dir_state_i

    pdb_files_ref = glob.glob('%s/*/*/*.pdb' % output_dir_ref)
    pdb_files_state_i = glob.glob('%s/*/*/*.pdb' % output_dir_state_i)

    #if either reference or state_i didn't finish we should restart 
    if os.path.exists(conformation_info_ref_fname) and os.path.exists(conformation_info_ref_state_i_fname):
        if len(pdb_files_ref) > 0 and len(pdb_files_state_i) > 0:
            logger.info('SKIPPING RW FOR: %s --%d files already exist--' % (output_dir_ref, len(pdb_files_ref)))      
            logger.info('SKIPPING RW FOR: %s --%d files already exist--' % (output_dir_state_i, len(pdb_files_state_i)))       
            should_run_rw = False 
    elif len(pdb_files_ref) > 0: #incomplete job
        logger.info('removing %d pdb files in %s' % (len(pdb_files_ref),output_dir_ref))
        logger.info('removing %d pdb files in %s' % (len(pdb_files_state_i),output_dir_state_i))
        rw_helper_functions.remove_files(pdb_files_ref)
        rw_helper_functions.remove_files(pdb_files_state_i)

    return should_run_rw 


def run_rw_all_custom_template(num_top_rmsd: int = 40):
 
    conformational_states_df = pd.read_csv('./conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
    conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

    #conformational_states_df = conformational_states_df[conformational_states_df['uniprot_id'] == 'P69441'].reset_index(drop=True)

    for index,row in conformational_states_df.iterrows():

        logger.info('On row %d of %d' % (index, len(conformational_states_df)))  
        logger.info(asterisk_line)
        logger.info(row)
 
        uniprot_id = str(row['uniprot_id'])
        pdb_id_ref = str(row['pdb_id_ref'])
        pdb_id_state_i = str(row['pdb_id_state_i'])
        seg_len = int(row['seg_len'])

        for j,template_pdb_id in enumerate([pdb_id_ref, pdb_id_state_i]):
            
            alignment_dir = './conformational_states_training_data/alignment_data/%s/%s' % (uniprot_id,template_pdb_id)
            seed = index #keep seed constant between conformations 
            logger.info(asterisk_line)
            logger.info('j = %d' % j)
            logger.info('SEED = %d' % seed) 
            logger.info(asterisk_line)
            template_str = 'template=%s' % template_pdb_id
            output_dir_base = './conformational_states_training_data/rw_predictions/%s/%s' % (uniprot_id, template_str) 
            args = gen_args(template_pdb_id, alignment_dir, output_dir_base, seed)
            output_dir = '%s/%s/%s/rw-%s' % (output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config)

            if j == 0:
                should_run_rw = restart_incomplete_iterations(uniprot_id, pdb_id_ref, pdb_id_state_i, args)
                if should_run_rw:
                    logger.info("RUNNING %s" % output_dir)
                    scaling_factor, conformation_info, candidate_conformations = run_rw_pipeline(args)
                else:
                    logger.info("SKIPPING %s BECAUSE ALREADY EVALUATED" % output_dir) 
            else:
                if should_run_rw:
                    logger.info("RUNNING %s" % output_dir)
                    #use results from same sequence with different template
                    #to save computational time 
                    scaling_factor, conformation_info, candidate_conformations = run_rw_pipeline(args, scaling_factor, conformation_info, candidate_conformations, num_top_rmsd)
                    if conformation_info is None: #no conformations were accepted
                        logger.info('NO CONFORMATIONS WERE ACCEPTED, RETRYING FROM SCRATCH') 
                        rejected_pdb_files = glob.glob('%s/*/*/*.pdb' % output_dir)
                        logger.info('removing %d rejected pdb files in %s' % (len(rejected_pdb_files),output_dir))
                        rw_helper_functions.remove_files(rejected_pdb_files)
                        args.mean_plddt_threshold = args.mean_plddt_threshold-5  
                        run_rw_pipeline(args)
                else:
                    logger.info("SKIPPING %s BECAUSE ALREADY EVALUATED" % output_dir)
                scaling_factor = None
                conformation_info = None 
                candidate_conformations = None 



run_rw_all_custom_template() 
