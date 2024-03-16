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
import contextlib 

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
from pdb_utils.pdb_utils import align_and_get_rmsd
import rw_helper_functions

from run_openfold_rw_monomer import (
    eval_model, 
    propose_new_intrinsic_param_vanila_rw, 
    propose_new_intrinsic_param_discrete_ou, 
    propose_new_intrinsic_param_rw_w_momentum,
    run_rw_monomer, 
    get_scaling_factor_bootstrap, 
    get_bootstrap_candidate_conformations, 
    get_new_scaling_factor_candidates, 
    construct_grid_search_combinations, 
    run_grid_search_monomer, 
    summarize_rw 
)

FeatureDict = MutableMapping[str, np.ndarray]

logger = logging.getLogger('run_openfold_rw_monomer_batch')
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./rw_monomer_batch.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

finetune_openfold_path = './finetune_openfold.py'

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def gen_args(alignment_dir, output_dir_base):

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
        "--alignment_dir", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--conformation_dir", type=str, default=None,
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

    args.template_mmcif_dir = '/dev/shm/pdb_mmcif/mmcif_files'
    args.alignment_dir = alignment_dir
    args.output_dir_base = output_dir_base 
    args.config_preset = 'model_1_ptm'
    args.openfold_checkpoint_path = '/opt/databases/openfold/openfold_params/finetuning_ptm_2.pt'
    args.module_config = 'module_config_0'
    args.rw_hp_config = 'hp_config_0-0'
    args.skip_relaxation = True
    args.model_device = 'cuda:0'
    args.bootstrap_phase_only = True
        
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


@contextlib.contextmanager
def local_np_seed(seed):
    """ 
    The current state of the numpy random number 
    generator is saved using get_state(). 
    The yield statement is used to define the context
    block where the code within the context manager 
    will be executed. After the code within the context 
    block is executed, the original state of the numpy 
    random number generator is restored.
    
    Effectively, this enables a the same result
    to be outputted each time code using numpy 
    rng functions is executed within this context manager. 
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run_rw_pipeline(args, uniprot_id, random_seed):

    if args.log_level.lower() == 'debug':
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)

    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    if args.bootstrap_phase_only:
        output_dir = '%s/%s/%s/rw-%s' % (args.output_dir_base, 'rw', args.module_config, args.rw_hp_config)
        l1_output_dir = '%s/%s/%s' % (args.output_dir_base, 'rw', args.module_config)
        l2_output_dir = None
        output_dir = os.path.abspath(output_dir)
        l1_output_dir = os.path.abspath(l1_output_dir)
    else: 
        output_dir = '%s/%s/%s/train-%s/rw-%s' % (args.output_dir_base, 'rw', args.module_config, args.train_hp_config, args.rw_hp_config)
        l1_output_dir = '%s/%s/%s' % (args.output_dir_base, 'rw', args.module_config)
        l2_output_dir = '%s/%s/%s/train-%s' % (args.output_dir_base, 'rw', args.module_config, args.train_hp_config) 
        output_dir = os.path.abspath(output_dir)
        l1_output_dir = os.path.abspath(l1_output_dir)
        l2_output_dir = os.path.abspath(l2_output_dir) 

    logger.info("OUTPUT DIRECTORY: %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    alignment_dir = args.alignment_dir
    file_id = os.listdir(alignment_dir)
    if len(file_id) > 1:
        raise ValueError("should only be a single directory under %s" % alignment_dir)
    else:
        file_id = file_id[0] #e.g 1xyz_A
        file_id_wo_chain = file_id.split('_')[0]
    alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
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
    logger.info("PROTEIN SEQUENCE:")
    logger.info(seq)

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

    logger.debug("FEATURE DICTIONARY:")
    logger.debug(feature_dict)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    intrinsic_param_zero = np.zeros(intrinsic_dim)
    model = load_model_w_intrinsic_param(config, module_config_data, args.model_device, args.openfold_checkpoint_path, args.jax_param_path, intrinsic_param_zero)

    #for m_name, module in dict(model.named_modules()).items():
    #    logger.info(m_name)
    #    logger.info('****')
    #    for c_name, layer in dict(module.named_children()).items():
    #        logger.info(c_name)
    

    #####################################################
    logger.info(asterisk_line)

    initial_pred_output_dir = '%s/initial_pred' %  l1_output_dir
    if not(args.bootstrap_phase_only): 
        bootstrap_output_dir = '%s/bootstrap' % l1_output_dir
        bootstrap_training_conformations_dir = '%s/bootstrap_training_conformations' % l2_output_dir
    else:
        bootstrap_output_dir = '%s/bootstrap' % output_dir #note we are saving boostrap_conformations in output_dir as opposed to l1_output_dir  

    if args.skip_bootstrap_phase:
        conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
        pdb_path_initial = '%s/initial_pred_unrelaxed.pdb' % initial_pred_output_dir   
        seed_fname = '%s/seed.txt' % bootstrap_output_dir
        if os.path.exists(conformation_info_fname) and os.path.exists(pdb_path_initial) and os.path.exists(seed_fname):
            with open(conformation_info_fname, 'rb') as f:
                conformation_info = pickle.load(f)
            random_seed = int(np.loadtxt(seed_fname))
            np.random.seed(random_seed)
            torch.manual_seed(random_seed + 1)
            logger.info('SKIPPING BOOTSTRAP PHASE')
        else:
            raise FileNotFoundError('either %s or %s not found' % (conformation_info_fname,initial_pred_info_fname))

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
        mean_plddt_initial, disordered_percentage_initial, _, _, pdb_path_initial = eval_model(model, config, intrinsic_param_zero, feature_processor, feature_dict, processed_feature_dict, 'initial_pred', initial_pred_output_dir, 'initial', args)    
        logger.info('pLDDT: %.3f, disordered percentage: %.3f, ORIGINAL MODEL' % (mean_plddt_initial, disordered_percentage_initial)) 

        scaling_factor_bootstrap = get_scaling_factor_bootstrap(intrinsic_dim, rw_hp_config_data, model, config, feature_processor, feature_dict, processed_feature_dict, l1_output_dir, args)       
        logger.info(asterisk_line)  
        logger.info('SCALING FACTOR TO BE USED FOR BOOTSTRAPPING: %s' % rw_helper_functions.remove_trailing_zeros(scaling_factor_bootstrap))
        logger.info(asterisk_line)  
        logger.info('RUNNING RW TO GENERATE CONFORMATIONS FOR BOOTSTRAP PHASE') 
        rw_hp_dict = {}
        rw_hp_dict['epsilon_scaling_factor'] = scaling_factor_bootstrap 
        if args.use_local_context_manager:
            with local_np_seed(random_seed):   
                state_history, conformation_info = run_rw_monomer(pdb_path_initial, intrinsic_dim, 'vanila', rw_hp_dict, args.num_bootstrap_steps, None, 'spherical', model, config, feature_processor, feature_dict, processed_feature_dict, bootstrap_output_dir, 'bootstrap', args, save_intrinsic_param=False, early_stop=False)
        else:
                state_history, conformation_info = run_rw_monomer(pdb_path_initial, intrinsic_dim, 'vanila', rw_hp_dict, args.num_bootstrap_steps, None, 'spherical', model, config, feature_processor, feature_dict, processed_feature_dict, bootstrap_output_dir, 'bootstrap', args, save_intrinsic_param=False, early_stop=False)
 
        bootstrap_acceptance_rate = sum(state_history)/len(state_history)
        logger.info('BOOTSTRAP ACCEPTANCE RATE: %.3f' % bootstrap_acceptance_rate)

        conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
        conformation_info = sorted(conformation_info, key=lambda x: x[0], reverse=True)
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info, f)

        run_time = time.perf_counter() - t0
        timing_dict = {'bootstrap': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'bootstrap')

        seed_fname = '%s/seed.txt' % bootstrap_output_dir
        np.savetxt(seed_fname, [random_seed], fmt='%d')

        rmsd_all = np.array([conformation_info[i][1] for i in range(0,len(conformation_info))])
        max_rmsd = np.max(rmsd_all)
        logger.info('MAX RMSD: %.3f' % max_rmsd)

 
    bootstrap_candidate_conformations = get_bootstrap_candidate_conformations(conformation_info, args) 

    if args.bootstrap_phase_only:
        return 
        
    #####################################################
    logger.info(asterisk_line)

    if not(args.skip_gd_phase):
        
        logger.info('BEGINNING GD PHASE:') 
        t0 = time.perf_counter()

        for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

            logger.info('ON CONFORMATION %d/%d:' % (iter_num+1,args.num_training_conformations))
            logger.info(conformation_info_i)
            bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)
            os.makedirs(bootstrap_training_conformations_dir_conformation_i, exist_ok=True)
            rw_helper_functions.remove_files_in_dir(bootstrap_training_conformations_dir_conformation_i) #this directory should contain a single conformation so we can run train_openfold on it 
            curr_pdb_path = conformation_info_i[2]
            curr_pdb_fname = '%s.pdb' % file_id_wo_chain #during training, expected filename is without chain_id (i.e 1xyz)
            dst_path = '%s/%s' % (bootstrap_training_conformations_dir_conformation_i,curr_pdb_fname)
            shutil.copy(curr_pdb_path,dst_path)

            target_str = 'target=conformation%d' % iter_num
            fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)

            arg1 = '--train_data_dir=%s' % bootstrap_training_conformations_dir_conformation_i
            arg2 = '--train_alignment_dir=%s' % args.alignment_dir
            arg3 = '--fine_tuning_save_dir=%s' % fine_tuning_save_dir
            arg4 = '--template_mmcif_dir=%s' % args.template_mmcif_dir
            arg5 = '--max_template_date=%s' % date.today().strftime("%Y-%m-%d")
            arg6 = '--precision=bf16'
            arg7 = '--gpus=1'
            arg8 = '--openfold_checkpoint_path=%s' % args.openfold_checkpoint_path 
            arg9 = '--resume_model_weights_only=True'
            arg10 = '--config_preset=custom_finetuning-SAID-all'
            arg11 = '--module_config=%s' % args.module_config
            arg12 = '--hp_config=%s' % args.train_hp_config
            arg13 = '--save_structure_output'
     
            if args.save_training_conformations:       
                script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13]
            else:
                script_arguments = [arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12]

            cmd_to_run = ["python", finetune_openfold_path] + script_arguments
            cmd_to_run_str = s = ' '.join(cmd_to_run)
            logger.info("RUNNING GRADIENT DESCENT WRT TO: %s" % curr_pdb_fname)
            logger.info(asterisk_line)
            logger.info("RUNNING THE FOLLOWING COMMAND:")
            logger.info(cmd_to_run_str)
            subprocess.run(cmd_to_run)

        run_time = time.perf_counter() - t0
        timing_dict = {'gradient_descent': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'gradient_descent')
        

    #################################################
    logger.info(asterisk_line)
 
    if rw_hp_config_data['cov_type'] == 'full': 
        logger.info('GENERATING RANDOM CORRELATION MATRIX')
        random_corr = gen_randcorr_sap.randcorr(intrinsic_dim)
    
    rw_hp_dict = {}
    rw_hp_parent_dir = '%s/rw_hp_tuning' % output_dir
    hp_acceptance_rate_dict = {}  
    hp_acceptance_rate_fname = '%s/hp_acceptance_rate_info.pkl' % rw_hp_parent_dir

    skip_auto_calc = False
    if os.path.exists(hp_acceptance_rate_fname):
        logger.info('LOADING %s' % hp_acceptance_rate_fname)
        with open(hp_acceptance_rate_fname, 'rb') as f:
            hp_acceptance_rate_dict = pickle.load(f)
        logger.info(hp_acceptance_rate_dict)
        rw_hp_dict = rw_helper_functions.get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data, is_multimer=False)
        if rw_hp_dict != {}:
            skip_auto_calc = True

    if not(skip_auto_calc):

        logger.info('BEGINNING RW HYPERPARAMETER TUNING PHASE')
        t0 = time.perf_counter()

        upper_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']+rw_hp_config_data['rw_tuning_acceptance_threshold_ub_tolerance'],2)
        lower_bound_acceptance_threshold = round(rw_hp_config_data['rw_tuning_acceptance_threshold']-rw_hp_config_data['rw_tuning_acceptance_threshold_lb_tolerance'],2)

        scaling_factor_candidates = train_hp_config_data['initial_scaling_factor_candidate']
        scaling_factor_candidates = list(map(float, scaling_factor_candidates))  
        alpha_candidates = [10e-4,10e-3,10e-2]
        gamma_candidates = [.9,.99,.999]
        
        while rw_hp_dict == {}:

            grid_search_combinations = construct_grid_search_combinations(rw_hp_config_data['rw_type'], scaling_factor_candidates, alpha_candidates, gamma_candidates)
            logger.info('INITIAL GRID SEARCH PARAMETERS:')
            logger.info(grid_search_combinations)

            state_history_dict = {} 

            for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

                logger.info('ON CONFORMATION %d/%d:' % (iter_num+1,args.num_training_conformations))
                bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)

                pdb_files = glob.glob('%s/*.pdb' % bootstrap_training_conformations_dir_conformation_i)
                if len(pdb_files) == 0: #if files do not exist in this directory, gradient descent was not run with respect to this target structure
                    continue 

                target_str = 'target=conformation%d' % iter_num
                fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)
                latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)

                model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
                sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir)
        
                if rw_hp_config_data['cov_type'] == 'full':
                    random_cov = rw_helper_functions.get_random_cov(sigma, random_corr)
                    logger.info('calculating cholesky')
                    L = np.linalg.cholesky(random_cov)
                else:
                    L = None 

                state_history_dict = run_grid_search_monomer(grid_search_combinations, state_history_dict, None, target_str, pdb_path_initial, intrinsic_dim, rw_hp_config_data['rw_type'], args.num_rw_hp_tuning_steps_per_round, L, rw_hp_config_data['cov_type'], model, config, feature_processor, feature_dict, processed_feature_dict, rw_hp_config_data, rw_hp_parent_dir, 'rw', args)
                hp_acceptance_rate_dict, grid_search_combinations, exit_status = rw_helper_functions.get_rw_hp_tuning_info(state_history_dict, hp_acceptance_rate_dict, grid_search_combinations, rw_hp_config_data, iter_num, args)
                
                if exit_status == 1:
                    break
      
            rw_hp_dict = rw_helper_functions.get_optimal_hp(hp_acceptance_rate_dict, rw_hp_config_data, is_multimer=False)
            if rw_hp_dict == {}:
                logger.info('NO SCALING FACTOR CANDIDATES FOUND THAT MATCHED ACCEPTANCE CRITERIA')
                scaling_factor_candidates = get_new_scaling_factor_candidates(hp_acceptance_rate_dict, rw_hp_config_data)
                logger.info('HYPERPARAMETER TUNING WITH NEW SCALING FACTOR CANDIDATES')
                logger.info(scaling_factor_candidates)
            else:
                hp_acceptance_rate_fname = '%s/hp_acceptance_rate_info.pkl' % rw_hp_parent_dir
                with open(hp_acceptance_rate_fname, 'wb') as f:
                    pickle.dump(hp_acceptance_rate_dict, f)
                logger.info(hp_acceptance_rate_dict)

        run_time = time.perf_counter() - t0
        timing_dict = {'hp_tuning': run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, 'hp_tuning')
     
                

    #################################################
    logger.info(asterisk_line)

    logger.info('BEGINNING RW PHASE')
    logger.info('HYPERPARAMETERS BEING USED:')
    logger.info(rw_hp_dict)

    for iter_num, conformation_info_i in enumerate(bootstrap_candidate_conformations):

        t0 = time.perf_counter()

        conformation_info_dict = {} #maps bootstrap_key to (pdb_path,plddt,disordered_percentage,rmsd)

        logger.info('ON CONFORMATION %d/%d:' % (iter_num+1,args.num_training_conformations))
        logger.info(conformation_info_i)
        bootstrap_training_conformations_dir_conformation_i = '%s/conformation%d' % (bootstrap_training_conformations_dir,iter_num)

        pdb_files = glob.glob('%s/*.pdb' % bootstrap_training_conformations_dir_conformation_i)
        if len(pdb_files) == 0: #if files do not exist in this directory, gradient descent was not run with respect to this target structure
            continue 

        target_str = 'target=conformation%d' % iter_num
        fine_tuning_save_dir = '%s/training/%s' % (l2_output_dir, target_str)
        latest_version_num = rw_helper_functions.get_latest_version_num(fine_tuning_save_dir)

        model_train_out_dir = '%s/version_%d' % (fine_tuning_save_dir, latest_version_num)
        sigma = rw_helper_functions.get_sigma(intrinsic_dim, model_train_out_dir)

        if rw_hp_config_data['cov_type'] == 'full':
            if args.use_local_context_manager:
                with local_np_seed(random_seed):
                    random_cov = rw_helper_functions.get_random_cov(sigma, random_corr)
            else:
                random_cov = rw_helper_functions.get_random_cov(sigma, random_corr)
            logger.info('calculating cholesky')
            L = np.linalg.cholesky(random_cov)
        else:
            L = None 
 
        rw_output_dir = '%s/rw/%s' % (output_dir,target_str)

        pdb_files = glob.glob('%s/**/*.pdb' % rw_output_dir)
        if len(pdb_files) >= args.num_rw_steps:
            if args.overwrite_pred:
                logger.info('removing pdb files in %s' % rw_output_dir)
                rw_helper_functions.remove_files(pdb_files)
            else:
                logger.info('SKIPPING RW FOR: %s --%d files already exist--' % (rw_output_dir, len(pdb_files)))
                continue 
        elif len(pdb_files) > 0: #incomplete job
            logger.info('removing pdb files in %s' % rw_output_dir)
            rw_helper_functions.remove_files(pdb_files)

        logger.info('BEGINNING RW FOR: %s' % rw_output_dir)

        if args.use_local_context_manager:
            with local_np_seed(random_seed):  
                state_history, conformation_info = run_rw_monomer(pdb_path_initial, intrinsic_dim, rw_hp_config_data['rw_type'], rw_hp_dict, args.num_rw_steps, L, rw_hp_config_data['cov_type'], model, config, feature_processor, feature_dict, processed_feature_dict, rw_output_dir, 'rw', args, save_intrinsic_param=False, early_stop=False)
        else:
            state_history, conformation_info = run_rw_monomer(pdb_path_initial, intrinsic_dim, rw_hp_config_data['rw_type'], rw_hp_dict, args.num_rw_steps, L, rw_hp_config_data['cov_type'], model, config, feature_processor, feature_dict, processed_feature_dict, rw_output_dir, 'rw', args, save_intrinsic_param=False, early_stop=False)
        conformation_info_dict[iter_num] = conformation_info

        acceptance_rate = sum(state_history)/len(state_history)
        logger.info('ACCEPTANCE RATE: %.3f' % acceptance_rate)

        conformation_info_output_dir = rw_output_dir
        conformation_info_fname = '%s/conformation_info.pkl' % conformation_info_output_dir
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info_dict, f)

        summarize_rw(iter_num, conformation_info)

        inference_key = 'inference_%d' % iter_num
        run_time = time.perf_counter() - t0
        timing_dict = {inference_key: run_time} 
        rw_helper_functions.write_timings(timing_dict, output_dir, inference_key)


def run_rw_all():
 
    conformational_states_df = pd.read_csv('./conformational_states_dataset/data/conformational_states_filtered_adjudicated.csv')
    conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

    for index,row in conformational_states_df.iterrows():

        logger.info('On row %d of %d' % (index, len(conformational_states_df)))   
        print(row)
     
        uniprot_id = str(row['uniprot_id'])
        pdb_id_ref = str(row['pdb_id_ref'])
        seg_len = int(row['seg_len'])

        alignment_dir = './conformational_states_dataset/alignment_data/%s' % uniprot_id
        output_dir_base = './conformational_states_dataset/predictions/%s' % uniprot_id 

        args = gen_args(alignment_dir, output_dir_base)
        output_dir = '%s/%s/%s/rw-%s' % (output_dir_base, 'rw', args.module_config, args.rw_hp_config)
        l1_output_dir = '%s/%s/%s' % (output_dir_base, 'rw', args.module_config)
        initial_pred_output_dir = '%s/initial_pred' %  l1_output_dir
        bootstrap_output_dir = '%s/bootstrap' % output_dir

        conformation_info_fname = '%s/conformation_info.pkl' % bootstrap_output_dir
        pdb_path_initial = '%s/initial_pred_unrelaxed.pdb' % initial_pred_output_dir  

        if os.path.exists(conformation_info_fname) and os.path.exists(pdb_path_initial):
            logger.info("SKIPPING %s BECAUSE ALREADY EVALUATED" % uniprot_id)
            continue 
            
        run_rw_pipeline(args, uniprot_id, index)


run_rw_all() 
