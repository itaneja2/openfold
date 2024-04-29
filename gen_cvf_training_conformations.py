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
import ml_collections as mlc
from typing import Tuple, List, Mapping, Optional, Sequence, Any, MutableMapping, Union
import contextlib 
import copy 

from openfold.utils.script_utils import load_model_w_intrinsic_param, load_model_w_cvf_and_intrinsic_param, parse_fasta, run_model_w_intrinsic_dim, prep_output, \
    update_timings, relax_protein
from openfold.data.data_pipeline import make_template_features 

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
from pdb_utils.pdb_utils import align_and_get_rmsd, get_ca_coords_matrix, save_ca_coords
import rw_helper_functions

from run_openfold_rw_monomer import (
    local_np_seed,
    eval_model,
    propose_new_intrinsic_param_vanila_rw,
    get_scaling_factor_bootstrap,
    summarize_rw 
)


FeatureDict = MutableMapping[str, np.ndarray]

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
    console_handler = logging.StreamHandler() 
    console_handler.setLevel(logging.INFO) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler('./cvf_training_conformations.log', mode='w') 
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    logger = logging.getLogger("gen_cvf_training_conformations_batch.gen_cvf_training_conformations")


finetune_openfold_path = './finetune_openfold.py'

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'



def run_rw_monomer(
    initial_pred_path: str, 
    intrinsic_dim: int, 
    rw_type: str, 
    rw_hp_dict: Mapping[str, List[float]], 
    num_total_steps: int, 
    L: np.ndarray, 
    cov_type: str, 
    model: nn.Module, 
    config: mlc.ConfigDict, 
    feature_processor: feature_pipeline.FeaturePipeline, 
    feature_dict: FeatureDict, 
    processed_feature_dict: FeatureDict, 
    output_dir: str, 
    phase: str, 
    relevant_bootstrap_idx: List[int], 
    args: argparse.Namespace,
    save_intrinsic_param: bool = False, 
    early_stop: bool = False,
):
    """Generates new conformations by modifying parameter weights via a random walk on episilon. 
    
       For a given perturbation to epsilon, model weights are updated via fastfood transform. 
       The model is then evaluated and the output conformation is either accepted or rejected
       according to the acceptance criteria. If the conformation is rejected, epsilon is
       reinitialized to the zero vector and the process repeats until num_total_steps have
       been run.  
    """

    if args.bootstrap_phase_only:
        base_tag = '%s_rw-%s' % (args.module_config, args.rw_hp_config)
    else:
        base_tag = '%s_train-%s_rw-%s' % (args.module_config, args.train_hp_config, args.rw_hp_config)
    base_tag = base_tag.replace('=','-')

    intrinsic_param_prev = np.zeros(intrinsic_dim)
    if rw_type == 'rw_w_momentum':
        velocity_param_prev = np.zeros(intrinsic_dim)
    num_rejected_steps = 0
    num_accepted_steps = 0  
    iter_n = 0 
    curr_step_iter_n = 0
    curr_step_aggregate = 0 
    state_history = [] 
    conformation_info = [] 

    while (num_rejected_steps+num_accepted_steps) < num_total_steps:

        if rw_type == 'vanila':
            intrinsic_param_proposed = propose_new_intrinsic_param_vanila_rw(intrinsic_param_prev, intrinsic_dim, 
                                                                             rw_hp_dict['epsilon_scaling_factor'], cov_type, L)
        elif rw_type == 'discrete_ou':
            intrinsic_param_proposed = propose_new_intrinsic_param_discrete_ou(intrinsic_param_prev, intrinsic_dim, 
                                                                               rw_hp_dict['epsilon_scaling_factor'], rw_hp_dict['alpha'], 
                                                                               cov_type, L)
        elif rw_type == 'rw_w_momentum':
            intrinsic_param_proposed = propose_new_intrinsic_param_rw_w_momentum(intrinsic_param_prev, velocity_param_prev, 
                                                                                 intrinsic_dim, rw_hp_dict['epsilon_scaling_factor'], 
                                                                                 rw_hp_dict['gamma'], cov_type, L)

        should_eval = True
        if relevant_bootstrap_idx is not None and curr_step_aggregate not in relevant_bootstrap_idx:
            should_eval = False
        
        if should_eval:
            curr_tag = '%s_iter%d_step-iter%d_step-agg%d' % (base_tag, iter_n, curr_step_iter_n, curr_step_aggregate) 
            mean_plddt, disordered_percentage, inference_time, accept_conformation, pdb_path_rw = eval_model(model, config, intrinsic_param_proposed, feature_processor, feature_dict, processed_feature_dict, curr_tag, output_dir, phase, args)    
            state_history.append(accept_conformation)
            logger.info('pLDDT: %.3f, disordered percentage: %.3f, step: %d' % (mean_plddt, disordered_percentage, curr_step_aggregate)) 

            if accept_conformation:
                logger.info('STEP %d: ACCEPTED' % curr_step_aggregate)
                intrinsic_param_prev = intrinsic_param_proposed
                curr_step_iter_n += 1
                num_accepted_steps += 1
                rmsd = align_and_get_rmsd(initial_pred_path, pdb_path_rw) 
                if save_intrinsic_param:
                    conformation_info.append((pdb_path_rw, rmsd, mean_plddt, disordered_percentage, inference_time, intrinsic_param_proposed, rw_hp_dict['epsilon_scaling_factor'], curr_step_aggregate))
                else:
                    conformation_info.append((pdb_path_rw, rmsd, mean_plddt, disordered_percentage, inference_time, None, rw_hp_dict['epsilon_scaling_factor'], curr_step_aggregate)) #if this function is called within a local context manager, curr_step_aggregate can be used to reproduce intrinsic_param_proposed
            else:
                logger.info('STEP %d: REJECTED' % curr_step_aggregate)
                if early_stop:
                    return state_history, conformation_info
                else:
                    intrinsic_param_prev = np.zeros(intrinsic_dim)
                    if rw_type == 'rw_w_momentum':
                        velocity_param_prev = np.zeros(intrinsic_dim)
                    iter_n += 1
                    curr_step_iter_n = 0 
                    num_rejected_steps += 1
        else:
            logger.info('STEP %d: SKIPPED BECAUSE NOT PRESENT IN relevant_bootstrap_idx' % curr_step_aggregate)
            iter_n += 1
            curr_step_iter_n = 0
            num_rejected_steps += 1 

        curr_step_aggregate += 1

    return state_history, conformation_info



def get_bootstrap_candidate_conformations(
    conformation_info: List[Tuple[Any,...]],
    args: argparse.Namespace
):
    """Generates a set of candidate conformations to use for the gradient descent phase
       of the pipeline. 
       
       The candidate conformations are derived from the bootstrap phase. More specifically,
       the candidate conformations correspond to those that were outputted one step prior 
       to a rejected conformation.   
    """ 

    bootstrap_conformations = {}
    iter_num_list = [] 
    for i in range(0,len(conformation_info)):
        f = conformation_info[i][0]
        rmsd = conformation_info[i][1]
        match = re.search(r'_iter(\d+)', f)
        iter_num = int(match.group(1)) #this corresponds to a given iteration (i.e a sequence of conformations that terminates in a rejection)
        match = re.search(r'step-iter(\d+)', f) 
        step_num = int(match.group(1)) #this corresponds to the step_num for a given iteration 
        if iter_num not in bootstrap_conformations:
            bootstrap_conformations[iter_num] = [(step_num,rmsd,i,f)]
        else:
            bootstrap_conformations[iter_num].append((step_num,rmsd,i,f)) 
        iter_num_list.append(iter_num)

    del bootstrap_conformations[max(iter_num_list)] #delete key corresponding to max(iter_num_list) because this iteration did not yield a rejected conformation (i.e it did not 'finish') 

    for key in bootstrap_conformations:
        bootstrap_conformations[key] = sorted(bootstrap_conformations[key], key=lambda x:x[0], reverse=True) #sort by step_num in reverse order 

    bootstrap_candidate_conformations = []
    for key in bootstrap_conformations:
        bootstrap_candidate_conformations.append(bootstrap_conformations[key][0]) #get last conformation in each iteration (i.e last conformation prior to rejection)
        
    bootstrap_candidate_conformations = sorted(bootstrap_candidate_conformations, key=lambda x:x[1], reverse=True) #sort by rmsd in reverse order 
    
    logger.debug('BOOTSTRAP CONFORMATIONS ALL:')
    logger.debug(bootstrap_conformations)    

    logger.info('BOOTSTRAP CANDIDATE CONFORMATIONS:')
    logger.info(bootstrap_candidate_conformations)

    return bootstrap_candidate_conformations



def run_rw_pipeline(args, scaling_factor_bootstrap=None, bootstrap_candidate_conformations=None):

    if args.log_level.lower() == 'debug':
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference, use_conformation_vectorfield_module=args.use_conformation_vectorfield_module, save_structure_module_intermediates=args.save_structure_module_intermediates)

    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    if args.use_conformation_vectorfield_module and not(args.conformation_vectorfield_checkpoint_path):
        raise ValueError("If using conformation_vectorfield_module, then conformation_vectorfield_checkpoint_path must be set")
 
    output_dir = '%s/%s/%s/rw-%s' % (args.output_dir_base, 'alternative_conformations-verbose', args.module_config, args.rw_hp_config)
    l0_output_dir = '%s/%s' % (args.output_dir_base, 'alternative_conformations-verbose')
    l1_output_dir = '%s/%s/%s' % (args.output_dir_base, 'alternative_conformations-verbose', args.module_config)

    output_dir = os.path.abspath(output_dir)
    l0_output_dir = os.path.abspath(l0_output_dir)
    l1_output_dir = os.path.abspath(l1_output_dir)
    
    logger.info("OUTPUT DIRECTORY: %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    alignment_dir = args.alignment_dir
    msa_files = glob.glob('%s/*.a3m' % alignment_dir)
    if len(msa_files) == 0: 
        file_id = os.listdir(alignment_dir)
        if len(file_id) > 1:
            raise ValueError("should only be a single directory under %s" % alignment_dir)
        else:
            file_id = file_id[0] #e.g 1xyz_A
            file_id_wo_chain = file_id.split('_')[0]
        alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
    else:
        alignment_dir_w_file_id = alignment_dir
    logger.info("alignment directory with file_id: %s" % alignment_dir_w_file_id)

    if args.fasta_file is None:
        pattern = "%s/*.fasta" % alignment_dir_w_file_id
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            fasta_file = files[0]
        else: 
            raise FileNotFoundError("Either >1 or 0 .fasta files found in alignment_dir -- should only be one")
    else:
        fasta_file = args.fasta_file

    with open(fasta_file, "r") as fp:
        fasta_data = fp.read()
    _, seq = parse_fasta(fasta_data)
    seq = seq[0]
    logger.info("PROTEIN SEQUENCE:")
    logger.info(seq)

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    with open('./rw_monomer_config.json') as f:
        rw_config_data = json.load(f)

    module_config_data = rw_config_data['SAID'][args.module_config]
    rw_hp_config_data = rw_config_data['hyperparameter']['rw'][args.rw_hp_config]
    train_hp_config_data = rw_config_data['hyperparameter']['train'][args.train_hp_config]
    intrinsic_dim = module_config_data['intrinsic_dim']

    pattern = "%s/features.pkl" % alignment_dir_w_file_id
    files = glob.glob(pattern, recursive=True)
    if len(files) == 1:
        features_output_path = files[0]
        logger.info('features.pkl path: %s' % features_output_path)
    else:
        features_output_path = ''

    if os.path.isfile(features_output_path):
        feature_dict = np.load(features_output_path, allow_pickle=True) #this is used for all predictions, so this assumes you are predicting a single sequence 
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=4,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        feature_dict = data_processor.process_fasta(
            fasta_path=fasta_file, alignment_dir=alignment_dir_w_file_id
        )
        features_output_path = os.path.join(alignment_dir_w_file_id, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)
        logger.info('SAVED %s' % features_output_path)

    #update template features with that derived from custom_template_pdb_id
    if args.custom_template_pdb_id:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=4,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
        template_features = make_template_features(seq, None, template_featurizer, args.custom_template_pdb_id)
        for key in feature_dict:
            if key in template_features:
                feature_dict[key] = template_features[key] 

    logger.debug("FEATURE DICTIONARY:")
    logger.debug(feature_dict)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    intrinsic_param_zero = np.zeros(intrinsic_dim)
    
    if not(args.use_conformation_vectorfield_module):
        model = load_model_w_intrinsic_param(config, module_config_data, args.model_device, args.openfold_checkpoint_path, args.jax_param_path, intrinsic_param_zero)
    else:
        model = load_model_w_cvf_and_intrinsic_param(config, module_config_data, args.model_device, args.openfold_checkpoint_path, args.conformation_vectorfield_checkpoint_path, intrinsic_param_zero)
  

    #####################################################
    logger.info(asterisk_line)

    initial_pred_output_dir = '%s/initial_pred' %  l0_output_dir 
    bootstrap_output_dir = '%s/bootstrap' % output_dir  

    #process features after updating seed 
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )
    processed_feature_dict = {
        k:torch.as_tensor(v, device=args.model_device)
        for k,v in processed_feature_dict.items()
    } 


    if not(args.skip_bootstrap_phase):      
        logger.info('BEGINNING BOOTSTRAP PHASE:') 
        t0 = time.perf_counter()

        logger.info('PREDICTING INITIAL STRUCTURE FROM ORIGINAL MODEL') 
        mean_plddt_initial, disordered_percentage_initial, _, _, initial_pred_path = eval_model(model, config, intrinsic_param_zero, feature_processor, feature_dict, processed_feature_dict, 'initial_pred', initial_pred_output_dir, 'initial', args)    
        logger.info('pLDDT: %.3f, disordered percentage: %.3f, ORIGINAL MODEL' % (mean_plddt_initial, disordered_percentage_initial)) 

        if scaling_factor_bootstrap is None:
            scaling_factor_bootstrap = get_scaling_factor_bootstrap(intrinsic_dim, rw_hp_config_data, model, config, feature_processor, feature_dict, processed_feature_dict, l1_output_dir, args)       
        logger.info('SCALING FACTOR TO BE USED FOR BOOTSTRAPPING: %s' % rw_helper_functions.remove_trailing_zeros(scaling_factor_bootstrap))
        logger.info('RUNNING RW TO GENERATE CONFORMATIONS FOR BOOTSTRAP PHASE') 
        rw_hp_dict = {}
        rw_hp_dict['epsilon_scaling_factor'] = scaling_factor_bootstrap 

        if bootstrap_candidate_conformations is not None:
            #to avoid repeated conformations, we are only going to evaluate models for 
            #random vectors where template_X was accepted and significantly perturbed (relatively speaking)
            relevant_bootstrap_idx = boostrap_candidate_conformations_all[conformation_num][-2]
        else:
            relevant_bootstrap_idx = None 

        if args.use_local_context_manager:
            with local_np_seed(random_seed):
                state_history, conformation_info = run_rw_monomer(initial_pred_path, intrinsic_dim, 'vanila', rw_hp_dict, args.num_bootstrap_steps, None, 'spherical', model, config, feature_processor, feature_dict, processed_feature_dict, bootstrap_output_dir, 'bootstrap', relevant_bootstrap_idx, args, save_intrinsic_param=False, early_stop=False)
        else:
                state_history, conformation_info = run_rw_monomer(initial_pred_path, intrinsic_dim, 'vanila', rw_hp_dict, args.num_bootstrap_steps, None, 'spherical', model, config, feature_processor, feature_dict, processed_feature_dict, bootstrap_output_dir, 'bootstrap', relevant_bootstrap_idx, args, save_intrinsic_param=False, early_stop=False)
 
        bootstrap_acceptance_rate = sum(state_history)/len(state_history)
        logger.info('BOOTSTRAP ACCEPTANCE RATE: %.3f' % bootstrap_acceptance_rate)

        conformation_info = sorted(conformation_info, key=lambda x: x[0], reverse=True)
        rw_helper_functions.dump_pkl(conformation_info, 'conformation_info', bootstrap_output_dir)

        run_time = time.perf_counter() - t0
        timing_dict = {'bootstrap': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'bootstrap')

        seed_fname = '%s/seed.txt' % bootstrap_output_dir
        np.savetxt(seed_fname, [random_seed], fmt='%d')

        rmsd_all = np.array([conformation_info[i][1] for i in range(0,len(conformation_info))])
        max_rmsd = np.max(rmsd_all)
        logger.info('MAX RMSD: %.3f' % max_rmsd)

 
    bootstrap_candidate_conformations = get_bootstrap_candidate_conformations(conformation_info, args) 
    
    return scaling_factor_bootstrap, bootstrap_candidate_conformations





           
if __name__ == "__main__":
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
        "--alignment_dir", type=str, required=True,
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
        "--conformation_vectorfield_checkpoint_path", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--num_bootstrap_steps", type=int, default=50
    )
    parser.add_argument(
        "--num_bootstrap_hp_tuning_steps", type=int, default=10
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
        "--module_config", type=str, default=None,
        help=(
            "module_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--rw_hp_config", type=str, default=None,
        help=(
            "hp_config_x where x is a number"
        )
    )
    parser.add_argument(
        "--train_hp_config", type=str, default=None,
        help=(
            "train_hp_config_x wheire x is a number"
        )
    )
    parser.add_argument(
        "--use_conformation_vectorfield_module", action="store_true", default=False,
        help=(
            """whether to run use_conformation_vectorfield_module after
               generating structure prediction."""
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
        "--bootstrap_phase_only", action="store_true", default=False
    )
    parser.add_argument(
        "--skip_bootstrap_phase", action="store_true", default=False
    )
    parser.add_argument(
        "--skip_gd_phase", action="store_true", default=False
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

    run_rw_pipeline(args)

