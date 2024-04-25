import argparse
import logging
import os
import random
import sys
import time
import glob
import pickle 

from torch import nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
import torch

from openfold.config import model_config
from openfold.data.conformation_data_module import ConformationVectorFieldDataModule 
from openfold.data import feature_pipeline
from openfold.model.model import ConformationVectorField
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants, protein 
from openfold.utils.argparse_utils import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)

from openfold.utils.script_utils import prep_output, get_model_basename, parse_fasta 

from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import ConformationVectorFieldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import (
    import_angle_resnet_weights_,
    import_openfold_weights_
)
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint
)

from pytorch_lightning.loggers import CSVLogger
from openfold.utils.logger import PerformanceLoggingCallback

import json 
import re 
from intrinsic_ft import modify_with_intrinsic_model
from collections import defaultdict

from pdb_utils.pdb_utils import align_and_get_rmsd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s : %(message)s')
console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('./train_conformation_vectorfield.log', mode='w') 
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ConformationVectorFieldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(ConformationVectorFieldWrapper, self).__init__()
        self.model = ConformationVectorField(config)
        self.is_multimer = config.globals.is_multimer
        self.loss = ConformationVectorFieldLoss(config.loss)
        self.config = config 
        self.cached_weights = None
        self.last_lr_step = -1

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():

            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

    def training_step(self, batch, batch_idx):

        ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        if self.is_multimer:
            batch = multi_chain_permutation_align(out=outputs,
                                                  features=batch,
                                                  ground_truth=ground_truth)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss


    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
#        return torch.optim.Adam(
#            self.model.parameters(),
#            lr=learning_rate,
#            eps=eps
#        )
        # Ignored as long as a DeepSpeed optimizer is configured


        optimizer = torch.optim.Adam(
            params = self.model.parameters(), 
            lr=learning_rate,
            eps=eps
        )

        return {
            "optimizer": optimizer,
        }

    
    '''def get_module_params(self):
        n = [name for name, param in self.model.named_parameters() if 'conformation_module' in name and 'angle_resnet' not in name]
        p = [param for name, param in self.model.named_parameters() if 'conformation_module' in name and 'angle_resnet' not in name]
        logger.info('PARAMETERS TO TUNE:')
        logger.info(n)
        return p''' 

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step


def main(args):
    
    torch.multiprocessing.set_start_method('spawn')

    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=(str(args.precision) == "16"),
    ) 

    model_module = ConformationVectorFieldWrapper(config)

    #######


    if args.conformation_vectorfield_checkpoint_path and not(args.resume_model_weights_only):
        sd = torch.load(args.conformation_vectorfield_checkpoint_path)
        last_global_step = int(sd['global_step'])
        model_module.resume_last_lr_step(last_global_step)
        logger.info("Successfully loaded last lr step...")
    elif args.conformation_vectorfield_checkpoint_path and args.resume_model_weights_only:
        sd = torch.load(args.conformation_vectorfield_checkpoint_path)
        sd = sd["state_dict"]
        sd = {k.replace('model.',''):v for k,v in sd.items()}
        import_openfold_weights_(model=model_module.model, state_dict=sd)
        logger.info("Successfully loaded ConformationModule weights...")
    
    #######

    if args.fine_tuning_save_dir is None:
        log_parent_dir = './conformationvec_training_logs'
        log_child_dir = 'tmp'
        fine_tuning_save_dir = '%s/%s' % (log_parent_dir, log_child_dir)
    else:
        fine_tuning_save_dir = args.fine_tuning_save_dir
        log_parent_dir = fine_tuning_save_dir[0:fine_tuning_save_dir.rfind('/')]
        log_child_dir = fine_tuning_save_dir.split('/')[-1] 
    
    os.makedirs(fine_tuning_save_dir, exist_ok=True)

    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    logger.info("Loading Monomer Data")
    data_module = ConformationVectorFieldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    if(args.checkpoint_every_nth_epoch is not None):
        mc = ModelCheckpoint(
            every_n_epochs=args.checkpoint_every_nth_epoch,
            auto_insert_metric_name=False,
            save_top_k=-1,
        )
        callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val/lddt_ca",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)

    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        if(args.wandb):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None
 
    if(args.wandb):
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wdb_logger.experiment.save(f"{freeze_path}")


    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs = 500,
        log_every_n_steps = 10,
        default_root_dir='./',
        strategy=strategy,
        callbacks=callbacks,
        logger=CSVLogger(log_parent_dir, name=log_child_dir),
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.conformation_vectorfield_checkpoint_path

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_ground_truth_data_dir", type=str, default = None, 
        help="Directory containing training mmCIF files -- corresponds two conformations 1 and 2."
    )
    parser.add_argument(
        "--train_rw_data_dir", type=str, default = None, 
        help="Directory containing mmCIF files generated via random walk"
    )
    parser.add_argument(
        "--train_alignment_dir", type=str, default= None,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "--fine_tuning_save_dir", type=str, default=None,
    )
    parser.add_argument(
        "--target_name", type=str, default=None,
        help='''Name of target structure being trained against. Defaults to name of fasta file'''
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )

    parser.add_argument(
        "--max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_filter_path", type=str, default=None,
        help='''Optional path to a text file containing names of training
                examples to include, one per line. Used to filter the training 
                set'''
    )
    parser.add_argument(
        "--distillation_filter_path", type=str, default=None,
        help="""See --train_filter_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_every_nth_epoch", type=int, default=None,
        help="""Checkpoint every nth training epoch"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--conformation_vectorfield_checkpoint_path", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=10000,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
        )
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--module_config", type=str, default='model_config_0',
        help=(
            "module_config_x where x is a number"
        )
    )

    parser.add_argument(
        "--hp_config", type=str, default='hp_config_0',
        help=(
            "hp_config_x where x is a number"
        )
    )

    parser.add_argument(
        "--_distillation_structure_index_path", type=str, default=None,
    )
    parser.add_argument(
        "--alignment_index_path", type=str, default=None,
        help="Training alignment index. See the README for instructions."
    )
    parser.add_argument(
        "--distillation_alignment_index_path", type=str, default=None,
        help="Distillation alignment index. See the README for instructions."
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--accelerator", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    ) 

    args = parser.parse_args()

    if args.max_template_date is None:
        args.max_template_date = '2024-03-29'

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(str(args.precision) == "16" and args.deepspeed_config_path is not None):
        raise ValueError("DeepSpeed and FP16 training are not compatible")

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)
