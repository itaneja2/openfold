import argparse
import logging
import math
import numpy as np
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

from openfold.utils.script_utils import load_model, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

import subprocess 
import pickle

import random
import time
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
import pandas as pd 

from pdb_utils.pdb_utils import align_and_get_rmsd
from rw_helper_functions import write_timings, remove_files, calc_disordered_percentage

logging.basicConfig(filename='benchmark_monomer.log', filemode='w')
logger = logging.getLogger(__file__)

TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def eval_model(model, args, config, feature_processor, feature_dict, processed_feature_dict, tag, output_dir):

    logging.info('Tag: %s' % tag)
    os.makedirs(output_dir, exist_ok=True)

    out, inference_time = run_model(model, processed_feature_dict, tag, output_dir, return_inference_time=True)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
    mean_plddt = np.mean(out["plddt"])

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        args.config_preset,
        args.multimer_ri_gap,
        args.subtract_plddt
    )

    output_name = 'temp'
    model_output_dir_temp = '%s/temp' % output_dir
    os.makedirs(model_output_dir_temp, exist_ok=True)

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir_temp, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    disordered_percentage = calc_disordered_percentage(unrelaxed_output_path)
    shutil.rmtree(model_output_dir_temp)

    output_name = tag
    model_output_dir = output_dir

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if args.cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        model_output_dir, f'{output_name}{unrelaxed_file_suffix}'
    )
 
    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))


    logger.info(f"Output written to {unrelaxed_output_path}...")

    return mean_plddt, disordered_percentage, inference_time, unrelaxed_output_path 


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    msa_sampling_params = []
    for max_extra_msa in [5120, 1024, 512, 256, 128, 64, 32, 16]:
        if max_extra_msa == 5120:
            max_msa_clusters = 512
        else:
            max_msa_clusters  = int(max_extra_msa/4)
        combo = (max_extra_msa, max_msa_clusters)
        msa_sampling_params.append(combo)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)
    
    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )


    #if skipping i=0
    pdb_path_initial = '%s/%s/max_extra_msa=%d/max_msa_clusters=%d/pred_1_%d-%d_unrelaxed.pdb' % (args.output_dir_base, 'benchmark', msa_sampling_params[0][0], msa_sampling_params[0][1], msa_sampling_params[0][0], msa_sampling_params[0][1])

    for i, items in enumerate(msa_sampling_params):

        t0 = time.perf_counter()

        max_extra_msa = items[0]
        max_msa_clusters = items[1]

        config.data.predict.max_extra_msa = max_extra_msa
        config.data.predict.max_msa_clusters = max_msa_clusters 

        output_dir = '%s/%s/max_extra_msa=%d/max_msa_clusters=%d' % (args.output_dir_base, 'benchmark', max_extra_msa, max_msa_clusters)
        model_name = 'max_extra_msa=%d_max_msa_clusters=%d' % (max_extra_msa, max_msa_clusters) 

        pdb_files = glob.glob('%s/*.pdb' % output_dir)
        if len(pdb_files) >= args.num_predictions_per_model:
            if args.overwrite_pred:
                logging.info('removing pdb files in %s' % output_dir)
                remove_files(pdb_files)
            else:
                logging.info('SKIPPING PREDICTION FOR: %s --%d files already exist--' % (output_dir, len(pdb_files)))
                continue 
        elif len(pdb_files) > 0: #incomplete job
            logging.info('removing pdb files in %s' % output_dir)
            remove_files(pdb_files)
          
        output_dir = os.path.abspath(output_dir)
        logging.info('Output Directory: %s' % output_dir)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed + 1)

        os.makedirs(output_dir, exist_ok=True)
        alignment_dir = args.alignment_dir
        file_id = os.listdir(alignment_dir)
        if len(file_id) > 1:
            raise ValueError("should only be a single directory under %s" % alignment_dir)
        else:
            file_id = file_id[0] #e.g 1xyz_A
            file_id_wo_chain = file_id.split('_')[0]
        alignment_dir_w_file_id = '%s/%s' % (alignment_dir, file_id)
        logging.info("alignment directory with file_id: %s" % alignment_dir_w_file_id)

        if args.fasta_file is None:
            pattern = "%s/*.fasta" % alignment_dir_w_file_id
            files = glob.glob(pattern, recursive=True)
            if len(files) == 1:
                fasta_file = files[0]
            else: 
                raise FileNotFoundError("Multiple .fasta files found in alignment_dir -- should only be one")
        else:
            fasta_file = args.fasta_file

        with open(fasta_file, "r") as fp:
            fasta_data = fp.read()
        _, seq = parse_fasta(fasta_data)
        logger.info("PROTEIN SEQUENCE:")
        logger.info(seq)
 
        pattern = "%s/features.pkl" % alignment_dir_w_file_id
        files = glob.glob(pattern, recursive=True)
        if len(files) == 1:
            features_output_path = files[0]
            logging.info('features.pkl path: %s' % features_output_path)
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
                obsolete_pdbs_path=args.obsolete_pdbs_file_path
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
            logging.info('SAVED %s' % features_output_path)

        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        model = load_model(config, args.model_device, args.openfold_checkpoint_path, args.jax_param_path)

        conformation_info_dict = {}
        conformation_info = [] 

        for j in range(0,args.num_predictions_per_model):

            #process features after updating seed 
            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict',
            )
            processed_feature_dict = {
                k:torch.as_tensor(v, device=args.model_device)
                for k,v in processed_feature_dict.items()
            } 

            tag = 'pred_%d_%d-%d' % (j+1,max_extra_msa,max_msa_clusters)
            logging.info('RUNNING model %s, pred %d' % (model_name,j))

            mean_plddt, disordered_percentage, inference_time, pdb_path = eval_model(model, args, config, feature_processor, feature_dict, processed_feature_dict, tag, output_dir)    
            logging.info('pLDDT: %.3f, disordered percentage: %.3f' % (mean_plddt, disordered_percentage)) 

            if i == 0 and j == 0:
                rmsd = 0 
                pdb_path_initial = pdb_path
            else: 
                rmsd = align_and_get_rmsd(pdb_path_initial, pdb_path)            

            conformation_info.append((pdb_path, rmsd, mean_plddt, disordered_percentage, inference_time))

        conformation_info_dict[model_name] = conformation_info
        conformation_info_output_dir = output_dir
        conformation_info_fname = '%s/conformation_info.pkl' % conformation_info_output_dir
        with open(conformation_info_fname, 'wb') as f:
            pickle.dump(conformation_info_dict, f)

        inference_key = 'inference_%d' % i
        run_time = time.perf_counter() - t0
        timing_dict = {inference_key: run_time} 
        write_timings(timing_dict, output_dir, inference_key)

           
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
        "--alignment_dir", type=str, required=True,
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

    main(args)

