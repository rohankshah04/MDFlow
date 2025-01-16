"""
model.py

Houses a wrapper around alphafold.py's AlphaFold
(e.g. MDFlow) if you need:
 - Additional velocity or score heads
 - Cross-attention to external data
 - Etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your core AF model
from af import AlphaFold
from .layers import GaussianFourierProjection
from openfold.model.primitives import Linear


class MDFlow(nn.Module):
    """
    A simple wrapper that:
      - Holds a MovieGenAlphaFold instance
      - Optionally adds a velocity or direction head
      - Incorporates time embeddings if you prefer to do that here
    """
    def __init__(self, config, use_velocity_head=False):
        super().__init__()
        self.config = config
        self.af_model = MovieGenAlphaFold(config)

        # Example: If you want a separate velocity or direction head
        # that directly predicts velocity from the single embedding (s)
        self.use_velocity_head = use_velocity_head
        if use_velocity_head:
            c_s = config.model["evoformer_stack"]["c_s"]
            # Predict 3D velocity for each residue
            self.velocity_head = nn.Linear(c_s, 3)

        # If you want to embed time here instead of alphafold.py:
        self.use_time_embed = True
        if self.use_time_embed:
            time_dim = self.config.model["input_pair_embedder"]["time_emb_dim"]
            c_z      = self.config.model["evoformer_stack"]["c_z"]
            self.time_proj = GaussianFourierProjection(embedding_size=time_dim)
            self.time_linear = Linear(time_dim, c_z, init="final")

    def forward(self, batch, prev_outputs=None):
        """
        1) If we embed time here, we modify batch["t"] -> time embedding
        2) We call self.af_model with the updated feats
        3) Optionally apply velocity head
        """
        # Suppose we do something like:
        if self.use_time_embed and "t" in batch:
            # batch["t"] shape: [B], i.e. one time per item
            t_fourier = self.time_proj(batch["t"])       # [B, time_dim]
            t_emb     = self.time_linear(t_fourier)      # [B, c_z]
            # We have to broadcast or incorporate t_emb into pair embedding
            # Typically done inside alphafold.py, but you can do it here:
            # e.g., store it in batch to be used by alphafold
            batch["time_pair_emb"] = t_emb  # your alphafold code can look for it

        # 2) Standard AF model call
        outputs = self.af_model(batch, prev_outputs=prev_outputs)

        # 3) If we have a velocity head, apply it to single embedding
        if self.use_velocity_head:
            # single embedding shape: [B, N, c_s]
            s = outputs["single"]
            velocity = self.velocity_head(s)  # [B, N, 3]
            outputs["velocity"] = velocity

        return outputs