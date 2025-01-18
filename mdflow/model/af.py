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

from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import add

from misc import InputPairStack, GaussianFourierProjection, Linear

# -------------------------------------------------------------------
# INPUT EMBEDDER (NO RECYCLING)
# -------------------------------------------------------------------
def input_embedder(
    config,
    feats,
    inplace_safe: bool = False,
):
    """
    A minimal function that:
      - Uses InputEmbedder to produce MSA & pair embeddings (m, z).
      - Does NOT handle recycling (which we'll do in the transformer block).
    """
    # Initialize the standard AlphaFold input embedder
    input_embedder = InputEmbedder(**config["input_embedder"])

    # Generate initial MSA & pair embeddings
    m, z = input_embedder(
        feats["target_feat"], 
        feats["residue_index"], 
        feats["msa_feat"],
        inplace_safe=inplace_safe,
    )
    return m, z


# -------------------------------------------------------------------
# TRANSFORMER BLOCK W/ RECYCLING, EVOFORMER, STRUCTURE
# -------------------------------------------------------------------
def nx_transformer_block(
    config,
    globals_config,
    feats,
    m,
    z,
    prev_outputs=None,
    inplace_safe: bool = False,
):
    """
    A function that:
      1) Takes MSA+pair embeddings (m, z)
      2) Performs recycling (m_1_prev, z_prev, x_prev) if needed
      3) Optionally runs extra MSA embedder
      4) Passes everything through Evoformer
      5) Runs StructureModule => final 3D coords
      6) Returns outputs (including new recycling info)

    By having recycling here, we can let it update structure info with time
    if we do multiple passes or a time-based approach for flow matching.
    """

    # Basic shapes & masks
    seq_mask = feats["seq_mask"]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    msa_mask = feats["msa_mask"]
    n_seq = feats["msa_feat"].shape[-3]

    # 1) Recycling logic
    recycling_embedder = RecyclingEmbedder(**config["recycling_embedder"])

    if prev_outputs is None:
        # No prior pass => create zeros
        batch_dims = feats["target_feat"].shape[:-2]
        n_res = feats["target_feat"].shape[-2]
        m_1_prev = m.new_zeros(
            (*batch_dims, n_res, config["input_embedder"]["c_m"]),
            requires_grad=False
        )
        z_prev = z.new_zeros(
            (*batch_dims, n_res, n_res, config["input_embedder"]["c_z"]),
            requires_grad=False
        )
        x_prev = z.new_zeros(
            (*batch_dims, n_res, residue_constants.atom_type_num, 3),
            requires_grad=False
        )
    else:
        m_1_prev = prev_outputs["m_1_prev"]
        z_prev   = prev_outputs["z_prev"]
        x_prev   = prev_outputs["x_prev"]

    # Convert x_prev to pseudo-beta
    x_prev = pseudo_beta_fn(
        feats["aatype"], x_prev, None
    ).to(dtype=z.dtype)

    # Run recycling embedder
    m_1_prev_emb, z_prev_emb = recycling_embedder(
        m_1_prev,
        z_prev,
        x_prev,
        inplace_safe=inplace_safe,
    )

    # Add them in
    m[..., 0, :, :] += m_1_prev_emb
    z = add(z, z_prev_emb, inplace=inplace_safe)

    # 2) Extra MSA trunk (if enabled in config)
    if config["extra_msa"].enabled:
        extra_msa_embedder = ExtraMSAEmbedder(
            **config["extra_msa"]["extra_msa_embedder"]
        )
        extra_msa_stack = ExtraMSAStack(
            **config["extra_msa"]["extra_msa_stack"]
        )
        logger.info("feats from msa:", feats.keys())
        a = extra_msa_embedder(build_extra_msa_feat(feats))
        z = extra_msa_stack(
            a, z,
            msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
            chunk_size=globals_config.chunk_size,
            use_lma=globals_config.use_lma,
            pair_mask=pair_mask.to(dtype=m.dtype),
            inplace_safe=inplace_safe,
            _mask_trans=config["_mask_trans"],
        )

    # 3) Evoformer
    evoformer = EvoformerStack(**config["evoformer_stack"])
    m, z, s = evoformer(
        m,
        z,
        msa_mask=msa_mask.to(dtype=m.dtype),
        pair_mask=pair_mask.to(dtype=z.dtype),
        chunk_size=globals_config.chunk_size,
        use_lma=globals_config.use_lma,
        use_flash=globals_config.use_flash,
        inplace_safe=inplace_safe,
        _mask_trans=config["_mask_trans"],
    )

    # Prepare for StructureModule
    outputs = {
        "msa": m[..., :n_seq, :, :],
        "pair": z,
        "single": s,
    }

    # 4) Structure module => final 3D coords
    structure_module = StructureModule(**config["structure_module"])
    sm_outputs = structure_module(
        outputs,
        feats["aatype"],
        mask=seq_mask.to(dtype=s.dtype),
        inplace_safe=inplace_safe,
        _offload_inference=globals_config.offload_inference,
    )

    # Convert from atom14 -> atom37
    final_positions = atom14_to_atom37(
        sm_outputs["positions"][-1],
        feats
    )
    outputs["final_atom_positions"] = final_positions
    outputs["final_atom_mask"] = feats["atom37_atom_exists"]
    outputs["final_affine_tensor"] = sm_outputs["frames"][-1]

    # 5) (Optional) run any heads like distogram, pTM, etc.
    aux_heads = AuxiliaryHeads(config["heads"])
    outputs.update(aux_heads(outputs))

    # 6) Store recycling outputs for next iteration/time
    outputs["m_1_prev"] = m[..., 0, :, :]
    outputs["z_prev"]   = z
    outputs["x_prev"]   = outputs["final_atom_positions"]

    return outputs


# -------------------------------------------------------------------
# ALPHAFOLD CLASS
# -------------------------------------------------------------------
class AlphaFold(nn.Module):
    """
    A high-level wrapper that:
      1) Calls the "input embedder" (input_embedder) to get (m, z)
      2) Optionally inserts time/noise embeddings
      3) Calls the "blue part" (nx_transformer_block) for recycling,
         Evoformer, and structure module => final coords
      4) Returns outputs (including next recycling info)
    """

    def __init__(self, config):
        super().__init__()
        self.globals = config.globals
        self.config = config.model

        # If you want to define any time/noise embedding layers here,
        # you can do so. For instance:
        #
        # self.input_time_projection = GaussianFourierProjection(
        #     embedding_size=self.config["input_pair_embedder"]["time_emb_dim"]
        # )
        # self.input_time_embedding = Linear(
        #     self.config["input_pair_embedder"]["time_emb_dim"],
        #     self.config["evoformer_stack"]["c_z"],
        #     init="final",
        # )
        # self.input_pair_stack = InputPairStack(**self.config["input_pair_stack"])

    def forward(self, feats, prev_outputs=None):
        """
        Single forward pass. In a time-unrolled or multi-step scenario,
        you might call this repeatedly, passing outputs from the previous
        step as `prev_outputs`.
        """
        inplace_safe = not (self.training or torch.is_grad_enabled())

        m, z = input_embedder(
            config=self.config,
            feats=feats,
            inplace_safe=inplace_safe,
        )

        # to add a time/noise embedding to `z`,
        #
        # if "t" in feats:
        #     time_emb = self.input_time_embedding(
        #         self.input_time_projection(feats["t"])
        #     )  # shape [B, c_z]
        #     # you'd have to broadcast or reshape this for a [B, N, N, c_z] shape
        #     z = z + ...
        #
        # For now, we omit that.

        outputs = nx_transformer_block(
            config=self.config,
            globals_config=self.globals,
            feats=feats,
            m=m,
            z=z,
            prev_outputs=prev_outputs,
            inplace_safe=inplace_safe,
        )

        return outputs