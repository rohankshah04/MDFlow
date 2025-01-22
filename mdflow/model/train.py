import pdb

import os
import wandb
import torch
import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
from typing import Dict, List

from model import Model
from mdflow.data.s3_dataloader import S3DataLoader
from mdflow.model.config import model_config
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.validation_metrics import drmsd, gdt_ts, gdt_ha
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import residue_constants
from torch.utils.data import Dataset, DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs if needed
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)  # Adjust logging level here
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

config = model_config(
    "initial_training",
    train=True,
    low_prec=True
)

loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 3



def visualize_protein_transition(current: torch.Tensor, predicted: torch.Tensor, 
                               target: torch.Tensor, protein_idx: int, timestep: float):
    """
    Create visualization of protein transition prediction vs actual
    Returns: Dictionary of metrics and plots for wandb logging
    """
    # Calculate structural similarity metrics
    pred_drmsd = drmsd(predicted, target).item()
    pred_gdt_ts = gdt_ts(predicted, target).item()
    pred_gdt_ha = gdt_ha(predicted, target).item()
    
    # Create plots/visualizations (you may want to customize this based on your needs)
    # For example, you could create:
    # - RMSD plots between structures
    # - Contact map differences
    # - 3D structure visualizations
    
    return {
        f"protein_{protein_idx}/drmsd_t{timestep:.2f}": pred_drmsd,
        f"protein_{protein_idx}/gdt_ts_t{timestep:.2f}": pred_gdt_ts,
        f"protein_{protein_idx}/gdt_ha_t{timestep:.2f}": pred_gdt_ha,
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Training arguments for MDFlow')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bucket_name', type=str, default='mdflow.atlas')
    parser.add_argument('--wandb_project', type=str, default='mdflow')
    parser.add_argument('--wandb_entity', type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument('--run_name', type=str, default='mdflow_overfit')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt_freq', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--noise_prob', type=float, default=0.8)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--overfit_samples', type=int, default=2, 
                      help='Number of samples to overfit on')
    parser.add_argument('--msa_dir', type=str, default='/cbica/home/shahroha/projects/AF-DIT/atlas/MSA')
    parser.add_argument('--val_samples', type=int, default=2)
    return parser.parse_args()

def load_msa_data(pt_file_path):
    msa_data = torch.load(pt_file_path)
    return msa_data['msa'], msa_data['deletion_matrix']

def calculate_b_factors(atom_positions):
    if isinstance(atom_positions, torch.Tensor):
        atom_positions = atom_positions.numpy()
    coordinate_variance = np.var(atom_positions, axis=-1)
    b_factors = np.array(8 * np.pi ** 2 / 3) * coordinate_variance
    return b_factors
    
class ProteinTrajectoryDataset(Dataset):
    def __init__(self, trajectory_folders, alignment_dir="/cbica/home/shahroha/projects/AF-DIT/atlas/alignment_dir/"):
        """
        Args:
            trajectory_folders (list): List of folder paths containing NPZ files
            alignment_dir (str): Directory containing MSA feature files
        """
        self.snapshot_pairs = []
        self.alignment_dir = alignment_dir
        self._load_trajectories(trajectory_folders)
        
    def _load_trajectories(self, folders):
        logger.info(f"Loading {len(folders)} trajectory folders")
        
        for folder_idx, folder in enumerate(folders):
            logger.info(f"Processing folder {folder_idx + 1}/{len(folders)}: {folder}")
            
            # Find NPZ file
            npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]
            if not npz_files:
                logger.error(f"No .npz files found in {folder}")
                continue
                
            data_path = os.path.join(folder, npz_files[0])
            protein_id = os.path.splitext(os.path.basename(data_path))[0]
            
            # Load trajectory data
            with np.load(data_path, allow_pickle=True) as data:
                # Load MSA data
                msa_pt_file = os.path.join(self.alignment_dir, f"{protein_id}_msa_features.pt")
                msa, deletion_matrix = load_msa_data(msa_pt_file)
                logger.info(f"data shsape: {data.files}")
                # Store trajectory metadata

                trajectory_data = {
                    'name': data['domain_name'][0].decode('utf-8'),
                    'aatype': data['aatype'],
                    'positions': data['all_atom_positions'],
                    'masks': data['all_atom_mask'],
                    'residue_index': data['residue_index'],
                    'seq_length': data['seq_length'],
                    'msa': msa,
                    'deletion_matrix': deletion_matrix,
                    'protein_idx': folder_idx
                }
                
                self.snapshot_pairs.append(trajectory_data)

    def __len__(self):
        total_pairs = 0
        for traj in self.snapshot_pairs:
            num_frames = traj['positions'].shape[0]
            total_pairs += (num_frames - 1)  # pairs of consecutive frames
        return total_pairs

    def __getitem__(self, idx):
        # Find which trajectory and which frame pair
        current_idx = idx
        for traj_idx, traj in enumerate(self.snapshot_pairs):
            num_frames = traj['positions'].shape[0]
            num_pairs = num_frames - 1
            if current_idx < num_pairs:
                # Found the right trajectory
                t = current_idx
                break
            current_idx -= num_pairs
        
        traj = self.snapshot_pairs[traj_idx]
        
        # Create input features (batch[0])
        b_factors = calculate_b_factors(traj['positions'][t])
        b_factors_target = calculate_b_factors(traj['positions'][t + 1])


        input_data = {
            'name': traj['name'],
            'all_atom_positions': torch.from_numpy(traj['positions'][t]).float(),
            'all_atom_mask': torch.from_numpy(traj['masks'][t]).float(),
            'aatype': torch.from_numpy(traj['aatype']).long(),
            'residue_index': torch.from_numpy(traj['residue_index']).long(),
            'seq_length': torch.from_numpy(traj['seq_length']).long(),
            't': t,
            'protein_idx': torch.tensor(traj['protein_idx']).long(),
            'msa': traj['msa'],
            'deletion_matrix': traj['deletion_matrix'],
            'plddt': torch.from_numpy(b_factors).float(),
            'prev_outputs': None
        }
        
        # Optional fields
        # if traj['b_factors'] is not None:
        #     input_data['b_factors'] = torch.from_numpy(traj['b_factors']).float()
            
        # Create target data (batch[1])
        target_data = {
            'name': traj['name'],
            'aatype': torch.from_numpy(traj['aatype']).long(),
            'residue_index': torch.from_numpy(traj['residue_index']).long(),
            'all_atom_positions': torch.from_numpy(traj['positions'][t + 1]).float(),
            'all_atom_mask': torch.from_numpy(traj['masks'][t + 1]).float(),
            'seq_length': torch.from_numpy(traj['seq_length']).long(),
            'plddt': torch.from_numpy(b_factors_target).float(),
            'prev_outputs': None
        }
        
        return input_data, target_data

def get_protein_dataloader(trajectory_folders, batch_size=1, num_workers=0):
    dataset = ProteinTrajectoryDataset(trajectory_folders)
    
    def collate_fn(batch):
        """Custom collate function to handle dictionary data"""
        input_batch = {}
        target_batch = {}
        
        # Combine all input features
        for key in batch[0][0].keys():
            if key in ['msa', 'deletion_matrix']:  # Special handling for MSA data if needed
                input_batch[key] = [item[0][key] for item in batch]
            else:
                elements = [item[0][key] for item in batch]
                if isinstance(elements[0], int):
                    elements = [torch.tensor(elem) for elem in elements]
                    input_batch[key] = torch.stack(elements)
                elif isinstance(elements[0], str):
                    input_batch[key] = elements
                elif elements[0] is None:
                    input_batch[key] = elements
                else:
                    input_batch[key] = torch.stack(elements)
        
        # Combine all target features
        for key in batch[0][1].keys():
            elements = [item[1][key] for item in batch]
            if isinstance(elements[0], int):
                elements = [torch.tensor(elem) for elem in elements]
                target_batch[key] = torch.stack(elements)
            elif isinstance(elements[0], str):
                target_batch[key] = elements
            elif elements[0] is None:
                target_batch[key] = elements
            else:
                target_batch[key] = torch.stack(elements)
        
        return input_batch, target_batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.login(key=os.environ['WANDB_API_KEY'])
    if args.wandb_entity:
        wandb.init(
            project=args.wandb_project,
            config=args,
            reinit=True,
        )
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    # Initialize S3 data loader
    s3_loader = S3DataLoader(
        bucket_name=args.bucket_name,
        batch_size=args.overfit_samples
    )

    train_trajectory_folders = s3_loader.get_next_batch()
    logger.info(f"paths to folders are: {train_trajectory_folders}")
    train_loader = get_protein_dataloader(train_trajectory_folders)
    val_loader = train_loader
    
    model = Model(config, args)
    
    # Load initial AlphaFold weights
    if os.path.exists('initial_training.pt'):
        logger.info("Loading initial AlphaFold weights")
        checkpoint = torch.load('/cbica/home/shahroha/projects/AF-DIT/mdflow/model/openfold/resources/openfold_params/initial_training.pt')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    # Add custom logging callback
    class ProteinVisualizationCallback(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % config['visualization_frequency'] == 0:
                with torch.no_grad():
                    current_struct = batch['current']
                    predicted_struct = outputs['predicted_structure']
                    target_struct = batch['next']
                    
                    # Log metrics and visualizations
                    metrics = visualize_protein_transition(
                        current_struct,
                        predicted_struct,
                        target_struct,
                        batch['protein_idx'],
                        batch['timestep']
                    )
                    
                    trainer.logger.log_metrics(metrics)
                    
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Similar visualization for validation batches
            pass
    
    # Initialize EMA if needed
    if not args.no_ema:
        model.ema = ExponentialMovingAverage(
            model=model.model,
            decay=0.999
        )

    # Set up PyTorch Lightning trainer with visualization callback
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=-1,
                every_n_epochs=args.ckpt_freq,
            ),
            ProteinVisualizationCallback()
        ],
        gradient_clip_val=args.grad_clip,
        check_val_every_n_epoch=args.val_freq,
        # Enable overfit check
        overfit_batches=args.overfit_samples,
    )

    # Start training
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main()