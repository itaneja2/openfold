# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import weakref

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Union

from openfold.model.conformation_module import ConformationStack
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)

from openfold.utils.feats import (
    build_conformation_pair_feat
)

from openfold.model.primitives import Linear, LayerNorm 
from openfold.model.structure_module import (
    AngleResnetBlock, 
)



class ConformationEmbedder(nn.Module):
    def __init__(self, config, c_t=39, c_z=128):
        super(ConformationEmbedder, self).__init__()
        
        self.config = config

        self.linear_t = Linear(c_t, c_z, bias=False)
        self.linear_u = Linear(c_z, c_z, bias=False)

        self.layer_norm_t = LayerNorm(c_t)

        self.relu = nn.ReLU()
         
    def forward(
        self,
        batch
    ):
        #pair_embeds = []

        #n_templ = batch["template_aatype"].shape[templ_dim]

        #idx = batch["template_aatype"].new_tensor(0)
        #single_template_feats = tensor_tree_map(
        #    lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
        #    batch,
        #)

        # [*, N, N, C_t]
        t = build_conformation_pair_feat(
            batch,
            inf=self.config.inf,
            eps=self.config.eps,
            **self.config.distogram,
        ).to(batch["template_pseudo_beta"].dtype)

        #this follows AF3-like logic in Algorithm 16
        u = self.linear_t(self.layer_norm_t(t))
        u = self.linear_u(self.relu(u))
        
        return u

class VectorFieldResnet(nn.Module):

    def __init__(self, c_in, c_hidden, no_blocks, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            epsilon:
                Small constant for normalization
        """
        super(VectorFieldResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_phi = Linear(self.c_hidden, 2)
        self.linear_theta = Linear(self.c_hidden, 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
        Returns:
            [*, no_angles, 2] predicted angles
        """

        s = self.relu(s)
        s = self.linear_in(s)

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, 2]
        theta = self.linear_theta(s)
        unnormalized_theta = theta
        theta_norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(theta ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        theta = theta / theta_norm_denom

        # [*, 2]
        phi = self.linear_phi(s)
        unnormalized_phi = phi
        phi_norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(phi ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        phi = phi / phi_norm_denom

        # [*, 2, 2]
        unnormalized_phi_theta = torch.stack((unnormalized_phi, unnormalized_theta), dim=-2)
        normalized_phi_theta = torch.stack((phi, theta), dim=-2)
        
        return unnormalized_phi_theta, normalized_phi_theta


class ConformationVectorField(nn.Module):

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(ConformationVectorField, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        cvf_config = self.config["conformation_vectorfield_module"]

        self.s_embedder = Linear(cvf_config.c_esm, cvf_config.c_s, bias=False) 
        self.layer_norm_s = LayerNorm(cvf_config.c_esm)

        self.conformation_embedder = ConformationEmbedder(
            self.template_config,
        )

        self.s_conformation = ConformationStack(
            **self.config["conformation_stack"],
        )

        self.conformation_vectorfield = VectorFieldResnet(
            cvf_config.c_s,
            cvf_config.c_resnet,
            cvf_config.no_resnet_blocks,
            cvf_config.epsilon,
        )


    def forward(self, feats):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        is_grad_enabled = torch.is_grad_enabled()

        s = feats["s"]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        #template_feats 
        feats = {
            k: v for k, v in feats.items() if k.startswith("template_")
        }

        print('embedding s')
        s = self.s_embedder(self.layer_norm_s(s))


        print('embeding conformation')
        z = self.conformation_embedder(
            feats
        )

        del feats 

        print('conformation stack')
        s = self.s_conformation(s,
                                   z,
                                   chunk_size=self.globals.chunk_size,
                                   use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                                   use_lma=self.globals.use_lma,
                                   use_flash=self.globals.use_flash,
                                   inplace_safe=inplace_safe)
     
        del z 

        unnormalized_phi_theta, normalized_phi_theta = self.conformation_vectorfield(
            s,
        )
        outputs = {
            "unnormalized_phi_theta": unnormalized_phi_theta,
            "normalized_phi_theta": normalized_phi_theta
        }

        del s 
        
            
        return outputs





