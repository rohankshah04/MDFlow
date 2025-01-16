#!/usr/bin/env python3
"""
Demonstration script for:
 - Loading a config from `config.py`
 - Instantiating our AlphaFold model
 - Loading *pretrained weights* from a checkpoint
 - Running a forward pass on synthetic data
"""

import os
import torch
import ml_collections as mlc

from config import model_config
from af import AlphaFold


def load_pretrained_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load pretrained weights from a local checkpoint file.
    Adjust logic if your checkpoint has a different structure.
    E.g., if the checkpoint is a dictionary with keys like
    ['model', 'optimizer', 'ema'], you might do:

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    or if it's already a direct state dict, just load that.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # This is a guess: many OpenFold checkpoints store weights under "model".
    # Adjust if your checkpoint keys differ.
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # If the checkpoint is already a raw state dict, just use it.
        state_dict = checkpoint

    # Attempt to load
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained weights from {ckpt_path}")


def make_mock_feats(batch_size=2, n_res=10, msa_depth=3):
    """
    Synthetic data shaped similarly to what OpenFold expects.
    Adjust shapes as needed for your particular architecture.
    """
    c_target = 22 # typical for AF2's target features
    c_msa = 49 # typical for AF2's MSA features

    feats = {
        "target_feat": torch.randn(batch_size, n_res, c_target),
        "msa_feat": torch.randn(batch_size, msa_depth, n_res, c_msa),
        "residue_index": torch.arange(n_res).unsqueeze(0).expand(batch_size, n_res),
        "seq_mask": torch.ones(batch_size, n_res),
        "msa_mask": torch.ones(batch_size, msa_depth, n_res),
        "aatype": torch.zeros(batch_size, n_res, dtype=torch.int64),
        "atom37_atom_exists": torch.ones(batch_size, n_res, 37),
    }

    return feats


def main():
    # 1) Create config via model_config
    #    For example, pick "model_1". Adjust as needed.
    cfg_name = "model_1"
    c = model_config(name=cfg_name, train=False, low_prec=False)

    # 2) Instantiate the AlphaFold model
    model = AlphaFold(c)
    model.eval() # inference mode

    # 3) Load pretrained weights
    #    If you have a checkpoint: "openfold_v1.pt" or similar
    checkpoint_path = "openfold_v1.pt" # <--- Adjust to your checkpoint filename
    load_pretrained_weights(model, checkpoint_path)

    # 4) Create synthetic features for a test forward pass
    feats = make_mock_feats(batch_size=2, n_res=10, msa_depth=3)

    # 5) Forward pass (no previous recycling info)
    with torch.no_grad():
        outputs = model(feats, prev_outputs=None)

    # 6) Print output shapes
    print("Outputs keys:", list(outputs.keys()))
    if "final_atom_positions" in outputs:
        print("final_atom_positions:", outputs["final_atom_positions"].shape)
    if "msa" in outputs:
        print("msa:", outputs["msa"].shape)
    if "pair" in outputs:
        print("pair:", outputs["pair"].shape)
    if "single" in outputs:
        print("single:", outputs["single"].shape)

    # 7) Optional: second pass for recycling
    with torch.no_grad():
        outputs_2 = model(feats, prev_outputs=outputs)
    print("Second pass final_atom_positions:", outputs_2["final_atom_positions"].shape)


if __name__ == "__main__":
    main()