import torch
import torch.nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class FlowMatchingLoss(nn.Module):
    """
    Implements Flow Matching loss for molecular dynamics trajectories.
    Similar to MovieGen but adapted for protein structures.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                predicted_velocity: torch.Tensor,
                target_velocity: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted_velocity: Predicted velocity field from the model
            target_velocity: Target velocity computed from MD trajectories
            mask: Optional mask for padding
        Returns:
            Flow matching loss
        """
        if mask is not None:
            loss = F.mse_loss(
                predicted_velocity * mask,
                target_velocity * mask,
                reduction='sum'
            ) / (mask.sum() + 1e-8)
        else:
            loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss

    def compute_target_velocity(x0: torch.Tensor, 
                            x1: torch.Tensor, 
                            t: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocity field between two protein conformations.
        Args:
            x0: Starting protein structure (B, L, 3) or (B, L, 14, 3) for atom14
            x1: Ending protein structure
            t: Timestep between 0 and 1
        Returns:
            Target velocity field
        """
        # Linear interpolation between structures
        return x1 - x0

    def sample_timesteps(batch_size: int,
                        num_timesteps: int,
                        device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps for training
        """
        return torch.rand(batch_size, device=device)

    def interpolate_structures(x0: torch.Tensor,
                            x1: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between protein structures at time t
        Args:
            x0: Starting structure
            x1: Ending structure  
            t: Timestep between 0 and 1
        Returns:
            Interpolated structure at time t
        """
        t = t.view(-1, 1, 1) if x0.dim() == 3 else t.view(-1, 1, 1, 1)
        return x0 + t * (x1 - x0)
