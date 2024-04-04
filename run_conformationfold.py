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

from openfold.utils.script_utils import load_conformationfold, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

import subprocess 
import pickle

import random
import time
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

from pdb_utils.pdb_utils import get_rmsd, align_and_get_rmsd
from rw_helper_functions import write_timings, remove_files, calc_disordered_percentage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./conformationfold.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


TRACING_INTERVAL = 50
asterisk_line = '******************************************************************************'


def eval_model(model, args, config, feature_processor, feature_dict, processed_feature_dict, tag, nearest_gtc_path, nearest_gtc_rmsd, output_dir):

    logging.info('Tag: %s' % tag)
    os.makedirs(output_dir, exist_ok=True)

    out, inference_time = run_model(model, processed_feature_dict, tag, output_dir, return_inference_time=True)

    #print(out)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        processed_feature_dict
    )

    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        args.config_preset,
        args.multimer_ri_gap,
        args.subtract_plddt
    )

    output_name = '%s-CF' % tag
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

    rmsd = align_and_get_rmsd(nearest_gtc_path, unrelaxed_output_path) #aligns unrelaxed_output_path to nearest_gtc_path, we don't want to overwrite gtc

    logger.info(f"Output written to {unrelaxed_output_path}...")
    logger.info("Output aligned to %s. RMSD = %.3f" % (nearest_gtc_path, rmsd))
    logger.info("Initial RMSD: %.3f --- Post ConformationFold RMSD: %.3f" % (nearest_gtc_rmsd , rmsd))

    return inference_time, unrelaxed_output_path 


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir_base, exist_ok=True)
    output_dir_name = args.output_dir_base.split('/')[-1]

    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)
    config.data.common.use_templates = False
    config.data.common.use_template_torsion_angles = False
    config.model.template.enabled = False
    config.model.heads.tm.enabled = False

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    model = load_conformationfold(config, args.model_device, args.conformationfold_checkpoint_path)

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

    for i,rw_conformation_path in enumerate(rw_conformations):

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

        msa_features = data_pipeline.make_dummy_msa_feats(seq)
        seq_features = data_pipeline.make_sequence_features(seq, template_pdb_id, len(seq)) 
        feature_dict = {**msa_features, **seq_features}

        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict'
        )
        for key in processed_feature_dict:
            processed_feature_dict[key] = processed_feature_dict[key].to(args.model_device)


        rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
        rw_conformation_name = rw_conformation_name.replace('_unrelaxed','')
        rw_conformation_name = rw_conformation_name.replace('_relaxed','')
        rw_conformation_parent_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]
        rw_conformation_input = '%s/structure_module_intermediates/%s_sm_output_dict.pkl' % (rw_conformation_parent_dir, rw_conformation_name)

        with open(rw_conformation_input, 'rb') as f:
            conformation_module_input = pickle.load(f) 

        ground_truth_conformations_path = os.path.join(args.ground_truth_data_dir, uniprot_id)
        ground_truth_conformations = sorted(glob.glob('%s/*.cif' % ground_truth_conformations_path)) 
        
        rmsd_dict = {} 
        for gtc in ground_truth_conformations:
            rmsd = get_rmsd(rw_conformation_path, gtc) 
            rmsd_dict[gtc] = rmsd
        print(rmsd_dict)
        nearest_gtc_path = min(rmsd_dict, key=rmsd_dict.get) #outputs the path to the ground_truth_conformation with the min RMSD w.r.t to random_rw_conformtion 
        nearest_gtc_rmsd = rmsd_dict[nearest_gtc_path]

        with open(rw_conformation_input, 'rb') as f:
            conformation_module_input = pickle.load(f) 

        conformation_module_input['single'] = torch.unsqueeze(conformation_module_input['single'], dim=-1)
        conformation_module_input['rigid_rotation'] = torch.unsqueeze(conformation_module_input['rigid_rotation'], dim=-1) #add recycling dimension
        conformation_module_input['rigid_translation'] = torch.unsqueeze(conformation_module_input['rigid_translation'], dim=-1) #add recycling dimension

        processed_feature_dict = {**processed_feature_dict, **conformation_module_input}
        #print(processed_feature_dict)

        gtc_name = nearest_gtc_path.split('/')[-1].split('.')[0]
        output_dir = '%s/conformationfold_predictions/gtc=%s' % (rw_conformation_parent_dir, gtc_name)
        eval_model(model, args, config, feature_processor, feature_dict, processed_feature_dict, rw_conformation_name, nearest_gtc_path, nearest_gtc_rmsd, output_dir)


           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_data_dir", type=str, default = None, 
        help="Directory containing training mmCIF files -- corresponds two conformations 1 and 2."
    )
    parser.add_argument(
        "--rw_data_dir", type=str, default = None, 
        help="Directory containing mmCIF files generated via random walk"
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
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--conformationfold_checkpoint_path", type=str, default=None,
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

