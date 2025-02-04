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
from typing import Mapping, Optional, Sequence, Any, Dict, List
import subprocess


from model import Model
from mdflow.data.s3_dataloader import S3DataLoader
from mdflow.model.config import model_config
from mdflow.data import data_pipeline
from mdflow.data.data_pipeline import _aatype_to_str_sequence
from mdflow.data import feature_pipeline
from mdflow.utils import protein
from mdflow.utils.protein import Protein
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.validation_metrics import drmsd, gdt_ts, gdt_ha
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import residue_constants
from openfold.data import parsers
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

def tensorize_features(features) -> Dict[str, torch.Tensor]:
    """Convert all numpy arrays in the FeatureDict to PyTorch tensors, 
       casting object arrays to float if necessary."""
    tensor_dict = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            # If the dtype is object, try converting to float
            if value.dtype == np.object or value.dtype == np.dtype('O'):
                value = value.astype(np.float32)
            # Convert the (now numeric) numpy array to a torch tensor
            tensor_dict[key] = torch.tensor(value)
        else:
            # If it’s not a numpy array, just copy it as-is
            tensor_dict[key] = value
    return tensor_dict

def load_msa_data(pt_file_path):
    msa_data = torch.load(pt_file_path)
    logger.info(f"msa_data info: {msa_data.keys()}")
    return msa_data['msa'], msa_data['deletion_matrix']

def calculate_b_factors(atom_positions):
    if isinstance(atom_positions, torch.Tensor):
        atom_positions = atom_positions.numpy()
    coordinate_variance = np.var(atom_positions, axis=-1)
    b_factors = np.array(8 * np.pi ** 2 / 3) * coordinate_variance
    return b_factors
    
class ProteinTrajectoryDataset(Dataset):
    def __init__(self, trajectory_folders, alignment_dir="/cbica/home/shahroha/projects/AF-DIT/atlas/MSA", mode: str = "train"):
        """
        Args:
            trajectory_folders (list): List of folder paths containing NPZ files
            alignment_dir (str): Directory containing MSA feature files
        """
        self.snapshot_pairs = []
        self.mode = mode

        valid_modes = ["train", "eval", "predict"]
        if(self.mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        self.alignment_dir = alignment_dir
        self._load_trajectories(trajectory_folders)
        self.data_cfg = data_cfg
        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer = None
        )
        self.feature_pipeline = feature_pipeline.FeaturePipeline(data_cfg)
        
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
                
                # Store trajectory metadata
                # aatype_dim = np.argmax(data['aatype'], axis=1)
                # logger.info(f"aatype_dim: {aatype_dim}")

                # aatype = _aatype_to_str_sequence(aatype_dim)

                trajectory_data = {
                    'name': data['domain_name'][0].decode('utf-8'),
                    'aatype': data['aatype'],
                    'positions': data['all_atom_positions'],
                    'masks': data['all_atom_mask'],
                    'residue_index': data['residue_index'],
                    'seq_length': data['seq_length'],
                    'msa': msa_pt_file,
                    'sequence': data['sequence'][0].decode('utf-8'),
                    'between_segment_residues': data['between_segment_residues']
                }
                
                self.snapshot_pairs.append(trajectory_data)

    def __len__(self):
        total_pairs = 0
        for traj in self.snapshot_pairs:
            num_frames = traj['positions'].shape[0]
            total_pairs += (num_frames - 1)  # pairs of consecutive frames
        return total_pairs

    def __getitem__(self, idx):
        # Find trajectory and frame pair (keep this part)
        current_idx = idx
        for traj_idx, traj in enumerate(self.snapshot_pairs):
            num_frames = traj['positions'].shape[0]
            num_pairs = num_frames - 1
            if current_idx < num_pairs:
                t = current_idx
                break
            current_idx -= num_pairs
        
        traj = self.snapshot_pairs[traj_idx]
        
        # Create raw features dictionary that combines structure and MSA data
        mmcif_feats = {
            'all_atom_positions': traj['positions'][t],  # Current frame
            'all_atom_mask': traj['masks'][t],
            'aatype': traj['aatype'],
            'residue_index': traj['residue_index'],
            'seq_length': traj['seq_length'],
            'between_segment_residues': traj['between_segment_residues'],
            'is_distillation': np.array(0., dtype=np.float32)
        }
        
        # Process MSA features using the pipeline
        msa_features = self.data_pipeline._process_msa_feats(
            f'{self.alignment_dir}/{traj["name"]}', 
            traj['sequence'],  # Assuming this is your seqres equivalent
            alignment_index=None
        )
        
        # Combine structure and MSA features
        data = {**mmcif_feats, **msa_features}
            
        # Process features through the pipeline
        input_feats = self.feature_pipeline.process_features(data, self.mode)
        input_feats['name'] = traj['name']
        input_feats['seqres'] = traj['sequence']
        input_feats['temp_pos'] = t
        ref_prot_inp = protein.from_dict(input_feats)
        input_feats['ref_prot'] = ref_prot_inp
        
        # Do the same for target frame (t+1)
        mmcif_feats_target = {
            'all_atom_positions': traj['positions'][t + 1],  # Next frame
            'all_atom_mask': traj['masks'][t + 1],
            'aatype': traj['aatype'],
            'residue_index': traj['residue_index'],
            'seq_length': traj['seq_length'],
            'between_segment_residues': traj['between_segment_residues'],
            'is_distillation': np.array(0., dtype=np.float32)
        }
        
        data_target = {**mmcif_feats_target, **msa_features}  # Use same MSA features
        target_feats = self.feature_pipeline.process_features(data_target, self.mode)
        target_feats['name'] = traj['name']
        target_feats['seqres'] = traj['sequence']
        ref_prot_tar = protein.from_dict(target_feats)
        target_feats['ref_prot'] = ref_prot_tar
        target_feats['temp_pos'] = t + 1

        logger.info(f"keys in target_feats are: {target_feats.keys()}")
        
        return input_feats, target_feats

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
                elif isinstance(elements[0], dict):
                    input_batch[key] = elements
                elif isinstance(elements[0], Protein):
                    input_batch[key] = elements
                else:
                    input_batch[key] = torch.stack(elements)
        
        # Combine all target features
        for key in batch[0][1].keys():
            if key in ['msa', 'deletion_matrix']:  # Special handling for MSA data if needed
                input_batch[key] = [item[0][key] for item in batch]
            else:
                elements = [item[1][key] for item in batch]
                if isinstance(elements[0], int):
                    elements = [torch.tensor(elem) for elem in elements]
                    target_batch[key] = torch.stack(elements)
                elif isinstance(elements[0], str):
                    target_batch[key] = elements
                elif elements[0] is None:
                    target_batch[key] = elements
                elif isinstance(elements[0], dict):
                    target_batch[key] = elements
                elif isinstance(elements[0], Protein):
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
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"current device: {torch.cuda.current_device()}")
    logger.info(f"device name: {torch.cuda.get_device_name()}")

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(config, args)
    
    # Load initial AlphaFold weights
    if os.path.exists('/cbica/home/shahroha/projects/AF-DIT/mdflow/model/openfold/resources/openfold_params/initial_training.pt'):
        logger.info("Loading initial AlphaFold weights")
        checkpoint = torch.load('/cbica/home/shahroha/projects/AF-DIT/mdflow/model/openfold/resources/openfold_params/initial_training.pt')
        model.load_state_dict(checkpoint, strict=False)
    
    logger.info(f"device: {device}")
    model = model.to(device)

    # Initialize EMA if needed
    if not args.no_ema:
        model.ema = ExponentialMovingAverage(
            model=model.model,
            decay=0.999
        )
    
    logger.info(f"model device before training: {next(model.parameters()).device}")
    
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
    logger.info(f"torch.version.cuda: {torch.version.cuda}")
    main()