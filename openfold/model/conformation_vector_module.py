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
from functools import reduce
import importlib
import math
import sys
from operator import mul

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Union

from openfold.model.primitives import Linear, LayerNorm, s_attn_point_weights_init_
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.geometry.quat_rigid import QuatRigid
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold.utils.geometry.vector import Vec3Array, square_euclidean_distance
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.precision_utils import is_fp16_enabled
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

from openfold.model.structure_module import AngleResnetBlock, StructureModuleTransition 

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class SphericalCoordsResnet(nn.Module):

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
        super(SphericalCoordsResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_theta = Linear(self.c_hidden, 2)
        self.linear_phi = Linear(self.c_hidden, 2)
        self.linear_r = Linear(self.c_hidden, 1)


        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

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
        phi[:,1] = self.relu(phi[:,1]) #so y-values are between 0 and 1 because phi is between [0,Pi]
        unnormalized_phi = phi
        phi_norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(phi ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        phi = phi / phi_norm_denom

        # [*, 1]
        r = self.linear_r(s)

        # [*, 2, 2]
        unnormalized_theta_phi = torch.stack((unnormalized_theta, unnormalized_phi), dim=-2)
        normalized_theta_phi = torch.stack((normalized_theta, normalized_phi), dim=-2)
        
        return unnormalized_theta_phi, normalized_theta_phi, r


class SAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(SAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))
        self.linear_kv = Linear(self.c_s, 2 * hc)
        self.linear_out = Linear(hc, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        mask: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H * 2 * C_hidden]
        kv = self.linear_kv(s)

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)


        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]

        # [*, H, N_res, N_res]
        if (is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        if (inplace_safe):
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        s = self.linear_out(o)

        return s



class ConformationVectorModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_s_attn,
        c_resnet,
        no_heads_s_attn,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_s_attn:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_s_attn:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            save_intermediates:
                Whether to save s (i.e first row of MSA) and backbone frames representation 
        """
        super(ConformationVectorModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_s_attn = c_s_attn
        self.c_resnet = c_resnet
        self.no_heads_s_attn = no_heads_s_attn
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.s_attn = SAttention(
            self.c_s,
            self.c_z,
            self.c_s_attn,
            self.no_heads_s_attn,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.s_attn_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_s_attn = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.spherical_coords_resnet = SphericalCoordsResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.epsilon,
        )

    def forward(
        self,
        evoformer_output_dict,
        mask=None,
        inplace_safe=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

                
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.s_attn(
                s, 
                mask, 
                inplace_safe=inplace_safe
            )
            s = self.s_attn_dropout(s)
            s = self.layer_norm_s_attn(s)
            s = self.transition(s)
           
            unnormalized_theta, theta, unnormalized_phi, phi, r = self.spherical_coords_resnet(s, s_initial)
 
            preds = {
                "unnormalized_theta": unnormalized_theta,   
                "theta": theta,
                "unnormalized_phi": unnormalized_phi,
                "phi": phi,
                "r": r
            }

            outputs.append(preds)


        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs


