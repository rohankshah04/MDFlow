"""
flow_matching.py

Logic for flow-/score-matching objectives on ATLAS MD snapshots.
We:
 - define a flow_matching_loss that compares predicted coords/velocity
   to ground truth next-frame coords
 - optionally add noise or time embeddings
 - implement a train_step_flow_matching() for a single training step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def flow_matching_loss(
    model_outputs,
    batch,
    config=None,
    loss_mode="next_frame",
):
    """
    Compute a flow-matching or next-frame loss given:
      - model_outputs: dict from MovieGenAlphaFold forward()
        containing final coords, pair/single embeddings, etc.
      - batch: your input from data_pipeline, e.g.:
        {
          "aatype": ...,
          "coords_t": shape [B, N, 37, 3],
          "coords_next": shape [B, N, 37, 3],
          "t": shape [B],  # time steps
          ...
        }
      - config: optional reference to your ml_collections config
      - loss_mode: "next_frame" or "velocity" or something else
    Returns:
      A scalar loss
    """

    # 1) Predicted coords from the model
    #    In alphafold.py, we typically store them at
    #    model_outputs["final_atom_positions"]
    pred_coords = model_outputs["final_atom_positions"]
    # shape might be [B, N, 37, 3] (AlphaFold style)

    # 2) Ground truth coords from the batch
    true_coords_next = batch["coords_next"]  # [B, N, 37, 3]
    coords_t         = batch["coords_t"]     # [B, N, 37, 3]

    # 3) Decide how you compute the loss
    #    a) Next-frame matching (directly match predicted coords to the next coords)
    if loss_mode == "next_frame":
        loss = F.mse_loss(pred_coords, true_coords_next)

    #    b) Velocity matching (predict direction from coords_t to coords_next)
    elif loss_mode == "velocity":
        # For instance, the model's 'velocity' might be (pred_coords - coords_t)
        pred_vel = pred_coords - coords_t
        true_vel = true_coords_next - coords_t
        loss = F.mse_loss(pred_vel, true_vel)

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    return loss


def train_step_flow_matching(
    model,
    batch,
    optimizer,
    config=None,
    device="cuda",
    loss_mode="next_frame",
    add_noise=False
):
    """
    A single training step for flow matching with the
    MovieGenAlphaFold or FlowMatchingModel. This:
      1) Moves data to device
      2) (Optionally) adds noise to coords_t
      3) Runs model forward pass
      4) Computes flow-matching loss
      5) Backprops + step
    """

    # 1) Move to device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    # 2) (Optionally) add noise
    if add_noise:
        # e.g., Gaussian noise with small std
        noise_level = 0.5  # or from config
        batch["coords_t"] = batch["coords_t"] + noise_level * torch.randn_like(batch["coords_t"])

    # 3) Forward pass
    #    If this is single-step, we do prev_outputs=None.
    #    If you have multi-frame unrolling, you might pass prior outputs from a loop.
    outputs = model(batch, prev_outputs=None)

    # 4) Compute flow matching or next-frame loss
    loss = flow_matching_loss(outputs, batch, config=config, loss_mode=loss_mode)

    # 5) Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()