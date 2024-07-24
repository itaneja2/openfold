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
import math
import sys
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from functools import partial
from abc import ABC, abstractmethod

from openfold.model.primitives import Linear, LayerNorm, softmax_no_cast
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.msa import (
    MSAAttention,
    MSAColumnAttention,
)
from openfold.utils.chunk_utils import chunk_layer, ChunkSizeTuner
from openfold.utils.tensor_utils import add
from openfold.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn

from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)

from openfold.model.triangular_attention import (
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)



class SPairWeightedAveraging(nn.Module):
    """
    Implements a modified version of MSAPairWeightedAveraging for {s_i} from Algorithm 10 of AF3 
        *specifically, no gating or dropout is applied 
    """

    def __init__(self, c_s, c_z, c_hidden, no_heads, gating=False): 
        """-
        Args:
            c_s:
                MSA channel dimension
            c_z:
                Pair channel dimension 
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
        """
        super(SPairWeightedAveraging, self).__init__()

        self.no_heads = no_heads

        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        
        self.linear_s = Linear(c_s, c_hidden * no_heads, bias=False, init="glorot")
        self.linear_z = Linear(c_z, no_heads, bias=False, init="glorot")
        if gating:
            self.linear_g = Linear(c_s, c_hidden * no_heads, init="gating")
        else:
            self.linear_g = None
        self.linear_o = Linear(c_hidden * no_heads, c_s, bias=False, init="glorot")

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor):

        s = self.layer_norm_s(s)        
        # [*, N_res, H * C_hidden]
        v = self.linear_s(s)
        # [*, N_res, N_res, H]
        b = self.linear_z(self.layer_norm_z(z))
        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        # [*, H, N_res, C_hidden]
        v = permute_final_dims(v, (1, 0, 2))

        # [*, H, N_res, N_res] 
        b = permute_final_dims(b, (2, 0, 1))
        b = softmax_no_cast(b, -1) 
        # [*, H, N_res, C_hidden]
        o = torch.einsum("...hij,...hjc->...hic",b,v)

        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(s))
            g = g.view(g.shape[:-1] + (self.no_heads, -1)) 
            g = permute_final_dims(g, (1, 0, 2)) 
            o = o*g
        
        # [*, N_res, H, C_hidden]
        o = permute_final_dims(v, (1, 0, 2))
        # [*, N_res, H*C_hidden]
        o = flatten_final_dims(o, 2)
        o = self.linear_o(o)

        return o 



class Transition(nn.Module):
    """
    Implements Algorithm 11 of AF3
    """
    def __init__(self, c, n):
        """
        Args:
            c:
                Input channel dimension
            n:
                Factor multiplied to c to obtain the hidden channel
                dimension
        """
        super(Transition, self).__init__()

        self.c = c
        self.n = n

        self.layer_norm = LayerNorm(self.c)
        self.linear_a = Linear(self.c, self.n * self.c, bias=False)
        self.linear_b = Linear(self.c, self.n * self.c, bias=False)
        self.linear_o = Linear(self.n * self.c, self.c, bias=False)
        self.swish = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.layer_norm(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        x = self.linear_o(self.swish(a)*b) 
        
        return x
        

#no transition layer or dropout# 
class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        fuse_projection_weights: bool,
        inf: float,
        eps: float
    ):
        super(PairStack, self).__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

    def forward(self,
        z: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=None,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + tmu_update
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=None,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + tmu_update
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.tri_att_start(
                    z,
                    mask=None,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
               )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.tri_att_end(
                    z,
                    mask=None,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
               )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        return z





 
class ConformationBlock(nn.Module, ABC):
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_s_att: int,
        no_heads_s: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        inf: float,
        eps: float,
    ):
        super(ConformationBlock, self).__init__()

        self.s_pwa = SPairWeightedAveraging(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_s_att,
            no_heads=no_heads_s,
        )
                       
        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            fuse_projection_weights=False,
            inf=inf,
            eps=eps
        )

        self.s_transition = Transition(
            c=c_s,
            n=transition_n,
        )

        self.pair_transition = Transition(
            c=c_z,
            n=transition_n,
        )


    def forward(self,
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = self.s_pwa(s, z)
 
        s = add(
            s,
            self.s_transition(
                s, 
            ),
            inplace=inplace_safe,
        )

        z = self.pair_stack(
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe
        )


        z = add(
            z,
            self.pair_transition(
                z, 
            ),
            inplace=inplace_safe,
        )

        return s, z






class ConformationStack(nn.Module):
    """

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_s_att: int,
        no_heads_s: int,
        c_hidden_mul: int, 
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
        """
        super(ConformationStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = ConformationBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_s_att=c_hidden_s_att,
                no_heads_s=no_heads_s,
                c_hidden_mul=c_hidden_mul,    
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair, 
                transition_n=transition_n,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)


    def _prep_blocks(self, 
        s: torch.Tensor, 
        z: torch.Tensor, 
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        inplace_safe: bool,
    ):
        blocks = [
            partial(
                b,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
            )
            for b in self.blocks
        ]


        return blocks


    def forward(self,
        s: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            pair_mask:
                [*, N_seq, N_res] MSA mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma and use_deepspeed_evo_attention.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
 
        blocks = self._prep_blocks(
            s=s,
            z=z,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            inplace_safe=inplace_safe,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None
        
        if blocks_per_ckpt is not None:
            s, z = checkpoint_blocks(
                blocks,
                args=(s, z),
                blocks_per_ckpt=blocks_per_ckpt,
            )
        else:
            for b in blocks:
                s, z = b(s, z)

        return s

