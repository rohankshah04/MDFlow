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

# some changes from above to isolate different parts.

import sys
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

import torch
import torch.nn as nn

from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    ExtraMSAEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
import openfold.np.residue_constants as residue_constants
from openfold.model.primitives import Linear

from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import add

from input_pair_stack import InputPairStack
from layers import GaussianFourierProjection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs if needed
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)  # Adjust logging level here
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

def sinusoidal_time_embedding(time_values, embed_dim):
    """
    time_values: A 1D or 2D tensor of shape [B] or [B, 1] containing 
                 the scalar time steps (e.g., temp_pos).
    embed_dim:   int, must be even for sin/cos pairing to work easily.

    Returns:
        A tensor of shape [B, embed_dim] representing the 
        non-learnable sinusoidal embedding for each time step.
    """
    if time_values.dim() == 1:
        # Make sure shape is [B, 1]
        time_values = time_values.unsqueeze(-1)  # [B, 1]

    # Convert to float
    time_values = time_values.float()

    # We'll create the "angles" used for sin/cos
    half_dim = embed_dim // 2
    # exponents: [0..half_dim-1]
    exponents = torch.arange(half_dim, device=time_values.device).float()

    # typical Transformer approach for frequencies:
    #  1 / (10000^(2*i / embed_dim))
    div_term = torch.exp(-math.log(10000.0) * (2 * exponents / embed_dim))

    # Make shape [1, half_dim] for broadcasting
    div_term = div_term.unsqueeze(0)  # [1, half_dim]

    # Now multiply the time values by the frequencies
    angles = time_values * div_term  # [B, half_dim]

    sin_part = torch.sin(angles)     # [B, half_dim]
    cos_part = torch.cos(angles)     # [B, half_dim]

    # Concatenate sin and cos to form the full embedding: [B, embed_dim]
    time_emb = torch.cat([sin_part, cos_part], dim=-1)
    return time_emb

class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config, extra_input=True):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )
        

        if(self.extra_msa_config.enabled):
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )
        
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

        ################
        self.input_pair_embedding = Linear(
            self.config.input_pair_embedder.no_bins, 
            self.config.evoformer_stack.c_z,
            init="final",
        )
        self.input_time_projection = GaussianFourierProjection(
            embedding_size=self.config.input_pair_embedder.time_emb_dim
        )
        self.input_time_embedding = Linear(
            self.config.input_pair_embedder.time_emb_dim, 
            self.config.evoformer_stack.c_z,
            init="final",
        )
        self.input_pair_stack = InputPairStack(**self.config.input_pair_stack)
        self.extra_input = extra_input
        if extra_input:
            self.extra_input_pair_embedding = Linear(
                self.config.input_pair_embedder.no_bins, 
                self.config.evoformer_stack.c_z,
                init="final",
            )   
            self.extra_input_pair_stack = InputPairStack(**self.config.input_pair_stack)
        
        ################

    def _get_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        
        lower = torch.linspace(
            self.config.input_pair_embedder.min_bin,
            self.config.input_pair_embedder.max_bin,
            self.config.input_pair_embedder.no_bins, 
        device=dists.device)
        dists = dists.unsqueeze(-1)
        inf = self.config.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z

    def _get_extra_input_pair_embeddings(self, dists, mask):

        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        
        lower = torch.linspace(
            self.config.input_pair_embedder.min_bin,
            self.config.input_pair_embedder.max_bin,
            self.config.input_pair_embedder.no_bins, 
        device=dists.device)
        dists = dists.unsqueeze(-1)
        inf = self.config.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        inp_z = self.extra_input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.extra_input_pair_stack(inp_z, mask, chunk_size=None)
        return inp_z
    
    def forward(self, batch):
        if 'prev_outputs' in batch:
            prev_outputs = batch['prev_outputs']
        else:
            prev_outputs = {}
        feats = batch

        # Primary output dictionary
        outputs = {}
        logger.info(f"batch keys: {batch.keys()}")
        logger.info(f"msa_mask shape: {batch['msa_mask'].shape}")
        logger.info(f"extra_mask_mask shape: {batch['extra_msa_mask'].shape}")

        # # This needs to be done manually for DeepSpeed's sake
        # dtype = next(self.parameters()).dtype
        # for k in feats:
        #     if(feats[k].dtype == torch.float32):
        #         feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device
        
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        ## Initialize the MSA and pair representations
        logger.info(f"")
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )
        logger.info(f"dimensions of m after input_embedder: {m.shape}")
        if prev_outputs is None or prev_outputs == {}:
            m_1_prev = m.new_zeros((*batch_dims, n, self.config.input_embedder.c_m), requires_grad=False)
            # [*, N, N, C_z]
            z_prev = z.new_zeros((*batch_dims, n, n, self.config.input_embedder.c_z), requires_grad=False)
            # [*, N, 3]
            x_prev = z.new_zeros((*batch_dims, n, residue_constants.atom_type_num, 3), requires_grad=False)
        else:
            m_1_prev, z_prev, x_prev = prev_outputs['m_1_prev'], prev_outputs['z_prev'], prev_outputs['x_prev']
        logger.info(f"dimensions of m_1_prev are {m_1_prev.shape}")
        logger.info(f"dimensions of noised_struc is {feats['noised_structure'].shape}")
        x_prev = pseudo_beta_fn(
            feats["aatype"], feats["noised_structure"], None
        ).to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            inplace_safe=inplace_safe,
        )

        # [*, S_c, N, C_m]
    
        m[..., 0, :, :] += m_1_prev_emb
        logger.info(f"m dimensions after adding m_1_prev_emb: {m.shape}")

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        temp_pos = batch['temp_pos']  # shape [B] or [B, 1]
        
        c_z = self.config.evoformer_stack.c_z
        c_m = self.config.evoformer_stack.c_m
        # compute time embedding of shape [B, c_z]
        # time_emb = sinusoidal_time_embedding(temp_pos, c_m)  # [B, c_m]

        # Expand to match z: [B, 1, 1, c_z]
        time_emb_z = sinusoidal_time_embedding(temp_pos, c_z)
        time_emb_z = time_emb_z.unsqueeze(1).unsqueeze(1)
        # z: [B, N, N, c_z]
        z = add(z, time_emb_z, inplace=inplace_safe)
    
        if self.extra_input:
            if 'extra_all_atom_positions' in batch:
                extra_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['extra_all_atom_positions'], None)
                extra_pseudo_beta_dists = torch.sum((extra_pseudo_beta.unsqueeze(-2) - extra_pseudo_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    extra_pseudo_beta_dists, 
                    batch['pseudo_beta_mask'],
                )
                
            else: # otherwise DDP complains
                B, L = batch['aatype'].shape
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    z.new_zeros(B, L, L), 
                    z.new_zeros(B, L),
                ) * 0.0
    
            z = add(z, extra_inp_z, inplace=inplace_safe)
        ########################

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))

            if(self.globals.offload_inference):
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z
    
                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )
    
                del input_tensors
            else:
                # [*, N, N, C_z]
                extra_msa_mask = feats["extra_msa_mask"].to(dtype=m.dtype)
                pair_mask = pair_mask.to(dtype=m.dtype)

                logger.info(f"Before extra_mask_stack")
                logger.info(f"a: device={a.device}, shape={a.shape}, dtype={a.dtype}")
                logger.info(f"z: device={z.device}, shape={z.shape}, dtype={z.dtype}")
                logger.info(f"msa_mask: device={msa_mask.device}, shape={msa_mask.shape}, dtype={msa_mask.dtype}")
                logger.info(f"pair_mask: device={pair_mask.device}, shape={pair_mask.shape}, dtype={pair_mask.dtype}")
                z = self.extra_msa_stack(
                    a, z,
                    msa_mask=extra_msa_mask,
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask,
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if(self.globals.offload_inference):
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )
    
            del input_tensors
        else:
            logger.info(f"Before evoformer")
            logger.info(f"m: device={m.device}, shape={m.shape}, dtype={m.dtype}")
            logger.info(f"z: device={z.device}, shape={z.shape}, dtype={z.dtype}")
            msa_mask = msa_mask.to(dtype=m.dtype)
            logger.info(f"msa_mask: device={msa_mask.device}, shape={msa_mask.shape}, dtype={msa_mask.dtype}")
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask,
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        
        outputs.update(self.aux_heads(outputs))


        # [*, N, C_m]
        outputs['m_1_prev'] = m[..., 0, :, :]

        # [*, N, N, C_z]
        outputs['z_prev'] = outputs["pair"]

        # [*, N, 3]
        outputs['x_prev'] = outputs["final_atom_positions"]

        logger.info(f"shape of outputs['final_atom_positions] are {outputs['final_atom_positions'].shape}")

        return outputs
