#!/usr/bin/env python3
"""
af_test_inf.py

Demonstration script for:
 - Loading a config from `config.py`
 - Replacing placeholders (NUM_RES, NUM_MSA_SEQ, etc.) within the config
 - Creating mock features based on the newly updated config
 - Running a forward pass on the AlphaFold model
"""

import sys
import logging
import torch
import ml_collections as mlc

# Adjust these imports to match your own file structure/package names
from config import model_config
from af import AlphaFold

def set_config_shapes(
cfg: mlc.ConfigDict,
num_res: int = 10,
num_msa_seq: int = 3,
num_extra_seq: int = 2,
num_templates: int = 4,
):
    feat_shapes = cfg.data.common.feat
    logger.info(f"Original feat shapes: {feat_shapes.keys()}")

    # Update shapes for target features
    feat_shapes["target_feat"] = [num_res, None]
    feat_shapes["aatype"] = [num_res]
    feat_shapes["msa_feat"] = [num_msa_seq, num_res, cfg.model.input_embedder.msa_dim]
    feat_shapes["residue_index"] = [num_res]
    feat_shapes["seq_mask"] = [num_res]
    feat_shapes["msa_mask"] = [num_msa_seq, num_res]
    feat_shapes["extra_msa_mask"] = [num_extra_seq, num_res]

    # Template features
    feat_shapes["template_aatype"] = [num_templates, num_res]
    feat_shapes["template_all_atom_mask"] = [num_templates, num_res, 37]
    feat_shapes["template_all_atom_positions"] = [num_templates, num_res, 37, 3]

    # Atom-level features
    feat_shapes["atom37_atom_exists"] = [num_res, 37]

    logger.info(f"Updated feat shapes: {feat_shapes}")
    return cfg

def make_mock_feats_from_config(cfg: mlc.ConfigDict, batch_size=1):
    """
    Generates mock features as tensors based on the config shapes.
    """
    feat_shapes = cfg.data.common.feat
    feats = {}

    for feat_name, shape_list in feat_shapes.items():
        if not shape_list or not isinstance(shape_list, list):
            # Skip invalid or empty feature shapes
            logger.warning(f"Skipping feature {feat_name} with invalid shape: {shape_list}")
            continue

        # Ensure all dimensions in shape_list are integers, replace None with 1
        try:
            final_shape = [batch_size] + [
                int(dim) if dim is not None else 1  # Replace None with a default value
                for dim in shape_list
            ]
        except (TypeError, ValueError):
            logger.error(f"Invalid shape for feature {feat_name}: {shape_list}")
            continue

        # Determine data type for the feature
        dtype = torch.float32
        if feat_name in ["aatype", "template_aatype", "residue_index"]:
            dtype = torch.long

        # Create the mock tensor
        if dtype == torch.long:
            feats[feat_name] = torch.zeros(final_shape, dtype=dtype)
        else:
            feats[feat_name] = torch.randn(final_shape, dtype=dtype)
    return feats

def main():

    # 1) Load model config (e.g., "model_1")
    cfg_name = "initial_training"
    logger.info(f"Using config preset: {cfg_name}")
    cfg = model_config(name=cfg_name, train=False, low_prec=False)

    # 2) Replace placeholders in the config
    cfg = set_config_shapes(
        cfg,
        num_res=10,
        num_msa_seq=3,
        num_extra_seq=2,
        num_templates=4,
    )

    # 3) Instantiate the AlphaFold model
    model = AlphaFold(cfg)
    model.eval()

    # 4) Create mock features
    feats = make_mock_feats_from_config(cfg, batch_size=2)

    # 5) Forward pass (no recycling info)
    with torch.no_grad():
        outputs = model(feats, prev_outputs=None)

    logger.info(f"Outputs keys: {list(outputs.keys())}")
    if "final_atom_positions" in outputs:
        logger.info(f"final_atom_positions: {outputs['final_atom_positions'].shape}")
    if "msa" in outputs:
        logger.info(f"msa: {outputs['msa'].shape}")
    if "pair" in outputs:
        logger.info(f"pair: {outputs['pair'].shape}")
    if "single" in outputs:
        logger.info(f"single: {outputs['single'].shape}")

    # 6) Optional second pass for recycling
    with torch.no_grad():
        outputs_2 = model(feats, prev_outputs=outputs)
        logger.info(
            f"Second pass final_atom_positions: {outputs_2['final_atom_positions'].shape}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    main()
