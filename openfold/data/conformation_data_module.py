import copy
from functools import partial
import json
import logging
import os
import pickle
from typing import Optional, Sequence, Any, Union
import numpy as np
import glob
import sys
import re 

import ml_collections as mlc
import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler
from openfold.np.residue_constants import restypes
from openfold.np.protein import from_pdb_string, to_modelcif
from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

from openfold.utils.script_utils import parse_fasta

from openfold.data.data_modules import OpenFoldDataLoader, OpenFoldBatchCollator 

sys.path.insert(0, '../../')
from pdb_utils.pdb_utils import get_rmsd


class ConformationFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ground_truth_data_dir: str,
                 rw_data_dir: str,
                 alignment_dir: str,
                 template_mmcif_dir: str,
                 max_template_date: str,
                 config: mlc.ConfigDict,
                 chain_data_cache_path: Optional[str] = None,
                 kalign_binary_path: str = '/usr/bin/kalign',
                 max_template_hits: int = 4,
                 obsolete_pdbs_file_path: Optional[str] = None,
                 template_release_dates_cache_path: Optional[str] = None,
                 shuffle_top_k_prefiltered: Optional[int] = None,
                 treat_pdb_as_distillation: bool = True,
                 filter_path: Optional[str] = None,
                 mode: str = "train",
                 alignment_index: Optional[Any] = None,
                 _output_raw: bool = False,
                 _structure_index: Optional[Any] = None,
                 ):
        """
            Args:
                ground_truth_data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                rw_data_dir:
                    A path to a directory containing PDB files derived from rw procedure
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(ConformationFoldSingleDataset, self).__init__()
        self.ground_truth_data_dir = ground_truth_data_dir
        self.rw_data_dir = rw_data_dir 
        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self.alignment_index = alignment_index
        self._output_raw = _output_raw
        self._structure_index = _structure_index

        self.supported_exts = [".cif", ".core", ".pdb"]

        valid_modes = ["train", "eval", "predict", "custom_train"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        if template_release_dates_cache_path is None:
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        def find_nth_occurrence(input_str, ch, n):
            start_idx = 0
            while n > 0:
                nth_occurrence = input_str.find(ch, start_idx)
                start_idx = nth_occurrence+1
                n -= 1
            return nth_occurrence

        children_dirs = glob.glob('%s/*/' % alignment_dir) #UNIPROT_ID
        children_dirs = [f[0:-1] for f in children_dirs] #remove trailing forward slash
        unique_uniprot_ids = [f[f.rindex('/')+1:] for f in children_dirs] #extract UNIPROT_ID 
        rw_conformations = [] #path to rw_conformations across all uniprot_ids 
        uniprot_ids = [] #corresponding uniprot_id for each rw_conformation
        for uniprot_id in unique_uniprot_ids:
            rw_data_dir_curr_uniprot_id = os.path.join(self.rw_data_dir, uniprot_id)
            if os.path.exists(rw_data_dir_curr_uniprot_id):
                rw_conformations_curr_uniprot_id = glob.glob('%s/*/*/*/*/bootstrap/ACCEPTED/*.pdb' % rw_data_dir_curr_uniprot_id)
                rw_conformations.extend(rw_conformations_curr_uniprot_id)
                uniprot_ids.extend([uniprot_id]*len(rw_conformations_curr_uniprot_id))
        self._rw_conformations = rw_conformations
        self._uniprot_ids = uniprot_ids 

        # If it's running template search for a monomer, then use hhsearch
        # as demonstrated in AlphaFold's run_alphafold.py code
        # https://github.com/deepmind/alphafold/blob/6c4d833fbd1c6b8e7c9a21dae5d4ada2ce777e10/run_alphafold.py#L462C1-L477
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=obsolete_pdbs_file_path,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        if not self._output_raw:
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir, residues_ignore_idx=None):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string, residues_ignore_idx=residues_ignore_idx
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if mmcif_object.mmcif_object is None:
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id,
            msa_dummy=True
        )

        return data

    def __getitem__(self, idx):

        if self.mode == 'custom_train' or self.mode == 'train' or self.mode == 'eval':

            rw_conformation_path = self._rw_conformations[idx]
            uniprot_id = self._uniprot_ids[idx]
            match = re.search(r'template=(\w+)', rw_conformation_path)
            template_pdb_id = match.group(1) #this is used as the template and the MSA is derived from this PDB_ID
            alignment_dir = '%s/%s/%s' % (self.alignment_dir, uniprot_id, template_pdb_id)

            fasta_file = "%s/%s.fasta" % (alignment_dir, template_pdb_id)
            with open(fasta_file, "r") as fp:
                fasta_data = fp.read()
            _, seq = parse_fasta(fasta_data)
            seq = seq[0]

            print('rw conformation: %s' % rw_conformation_path)

            rw_conformation_name = rw_conformation_path.split('/')[-1].split('.')[0]
            rw_conformation_name = rw_conformation_name.replace('_unrelaxed','')
            rw_conformation_name = rw_conformation_name.replace('_relaxed','')
            rw_conformation_parent_dir = rw_conformation_path[0:rw_conformation_path.rindex('/')]
            rw_conformation_input = '%s/structure_module_intermediates/%s_sm_output_dict.pkl' % (rw_conformation_parent_dir, rw_conformation_name)
            
            ground_truth_conformations_path = os.path.join(self.ground_truth_data_dir, uniprot_id)
            ground_truth_conformations = sorted(glob.glob('%s/*.cif' % ground_truth_conformations_path)) 

            with open(rw_conformation_input, 'rb') as f:
                conformation_module_input = pickle.load(f) 
            
            #outputs the path to the ground_truth_conformation with the min RMSD w.r.t random_rw_conformtion 
            rmsd_dict = {} 
            for gtc in ground_truth_conformations:
                rmsd = get_rmsd(rw_conformation_path, gtc)
                rmsd_dict[gtc] = rmsd
            print(rmsd_dict)
            nearest_gtc_path = min(rmsd_dict, key=rmsd_dict.get) 

            residues_ignore_idx_fname = nearest_gtc_path.replace('.cif', '-residues_ignore_idx.pkl')
            with open(residues_ignore_idx_fname, 'rb') as f:
                residues_ignore_idx = tuple(pickle.load(f))
 
            model_name_nearest_gtc, ext = nearest_gtc_path.split('/')[-1].split('.')
            file_id_nearest_gtc, chain_id_nearest_gtc = model_name_nearest_gtc.split('_')
            ext = '.%s' % ext

            #note: parse_mmcif assumes input msa sequence is consistent with mmcif sequence 
            if ext == ".cif":
                data = self._parse_mmcif(
                    nearest_gtc_path, file_id_nearest_gtc, chain_id_nearest_gtc, alignment_dir, residues_ignore_idx 
                )
            else:
                raise ValueError("Extension must be .cif")
           
            #print(data)
            #print('*******')
            #sequence features from process_mmcif are derived from nearest_gtc_path,  
            #which does not necessarily correspond to the input MSA, so we need to correct for this
            seq_features = data_pipeline.make_sequence_features(seq, template_pdb_id, len(seq)) 
            for key in data:
                if key in seq_features:
                    data[key] = seq_features[key]
            #print(seq_features)
            #print('*******')
            #print(data)

        if self._output_raw:
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode
        )
        #print('*******')
        #print(feats)

        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])],
            dtype=torch.int64,
            device=feats["aatype"].device)

        conformation_module_input['single'] = torch.unsqueeze(conformation_module_input['single'], dim=-1)
        conformation_module_input['rigid_rotation'] = torch.unsqueeze(conformation_module_input['rigid_rotation'], dim=-1) #add recycling dimension
        conformation_module_input['rigid_translation'] = torch.unsqueeze(conformation_module_input['rigid_translation'], dim=-1) #add recycling dimension

        feats = {**feats, **conformation_module_input}

        return feats

    def __len__(self):
        return len(self._rw_conformations)



class ConformationFoldDataModule(pl.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 template_mmcif_dir: str,
                 max_template_date: str,
                 train_ground_truth_data_dir: Optional[str] = None,
                 train_rw_data_dir: Optional[str] = None,
                 train_alignment_dir: Optional[str] = None,
                 train_chain_data_cache_path: Optional[str] = None,
                 distillation_data_dir: Optional[str] = None,
                 distillation_alignment_dir: Optional[str] = None,
                 distillation_chain_data_cache_path: Optional[str] = None,
                 val_data_dir: Optional[str] = None,
                 val_alignment_dir: Optional[str] = None,
                 predict_data_dir: Optional[str] = None,
                 predict_alignment_dir: Optional[str] = None,
                 kalign_binary_path: str = '/usr/bin/kalign',
                 train_filter_path: Optional[str] = None,
                 distillation_filter_path: Optional[str] = None,
                 obsolete_pdbs_file_path: Optional[str] = None,
                 template_release_dates_cache_path: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 train_epoch_len: int = 50000,
                 _distillation_structure_index_path: Optional[str] = None,
                 alignment_index_path: Optional[str] = None,
                 distillation_alignment_index_path: Optional[str] = None,
                 **kwargs
                 ):
        super(ConformationFoldDataModule, self).__init__()

        self.config = config
        self.template_mmcif_dir = template_mmcif_dir
        self.max_template_date = max_template_date
        self.train_ground_truth_data_dir = train_ground_truth_data_dir
        self.train_rw_data_dir = train_rw_data_dir
        self.train_alignment_dir = train_alignment_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.distillation_data_dir = distillation_data_dir
        self.distillation_alignment_dir = distillation_alignment_dir
        self.distillation_chain_data_cache_path = (
            distillation_chain_data_cache_path
        )
        self.val_data_dir = val_data_dir
        self.val_alignment_dir = val_alignment_dir
        self.predict_data_dir = predict_data_dir
        self.predict_alignment_dir = predict_alignment_dir
        self.kalign_binary_path = kalign_binary_path
        self.train_filter_path = train_filter_path
        self.distillation_filter_path = distillation_filter_path
        self.template_release_dates_cache_path = (
            template_release_dates_cache_path
        )
        self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        self.config_preset = kwargs['config_preset']

        if self.train_ground_truth_data_dir is None and self.predict_data_dir is None:
            raise ValueError(
                'At least one of train_ground_truth_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_ground_truth_data_dir is not None

        if self.training_mode and train_alignment_dir is None:
            raise ValueError(
                'In training mode, train_alignment_dir must be specified'
            )
        elif not self.training_mode and predict_alignment_dir is None:
            raise ValueError(
                'In inference mode, predict_alignment_dir must be specified'
            )
        elif val_data_dir is not None and val_alignment_dir is None:
            raise ValueError(
                'If val_data_dir is specified, val_alignment_dir must '
                'be specified as well'
            )

        # An ad-hoc measure for our particular filesystem restrictions
        self._distillation_structure_index = None
        if _distillation_structure_index_path is not None:
            with open(_distillation_structure_index_path, "r") as fp:
                self._distillation_structure_index = json.load(fp)

        self.alignment_index = None
        if alignment_index_path is not None:
            with open(alignment_index_path, "r") as fp:
                self.alignment_index = json.load(fp)

        self.distillation_alignment_index = None
        if distillation_alignment_index_path is not None:
            with open(distillation_alignment_index_path, "r") as fp:
                self.distillation_alignment_index = json.load(fp)

    def setup(self, stage=None):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(ConformationFoldSingleDataset,
                              template_mmcif_dir=self.template_mmcif_dir,
                              max_template_date=self.max_template_date,
                              config=self.config,
                              kalign_binary_path=self.kalign_binary_path,
                              template_release_dates_cache_path=self.template_release_dates_cache_path,
                              obsolete_pdbs_file_path=self.obsolete_pdbs_file_path)

        if self.training_mode:
            if 'custom_finetuning' in self.config_preset: 
                training_mode_type = 'custom_train'
            else:
                training_mode_type = 'train'

            self.train_dataset = dataset_gen(
                ground_truth_data_dir=self.train_ground_truth_data_dir,
                rw_data_dir=self.train_rw_data_dir, 
                chain_data_cache_path=self.train_chain_data_cache_path,
                alignment_dir=self.train_alignment_dir,
                filter_path=self.train_filter_path,
                max_template_hits=self.config.train.max_template_hits,
                shuffle_top_k_prefiltered=self.config.train.shuffle_top_k_prefiltered,
                treat_pdb_as_distillation=False,
                mode=training_mode_type,
                alignment_index=None,
            )


    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator()

        dl = OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    '''def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict")'''
