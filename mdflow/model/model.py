import logging
import sys
logger = get_logger(__name__)

import torch, os, wandb, time
import pandas as pd

from openfold.model.model import AlphaFold

from mdflow.utils.loss import AlphaFoldLoss
from mdflow.utils.diffusion import HarmonicPrior
from mdflow.utils import protein

from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.feats import pseudo_beta_fn
from openfold.data import data_transforms
from openfold.utils.exponential_moving_average import ExponentialMovingAverage

import pytorch_lightning as pl
import numpy as np
from openfold.np import residue_constants
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from collections import defaultdict
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler

class Model(pl.LightningModule):
    def __init__(config, args):
        self.model = AlphaFold(config)
        pass

    def noise(self, batch):
        device = batch['aatype'].device
        batch_dims = batch['all_atom_positions'].shape
        ny = self.harmonic_prior.sample(batch_dims)
        t = torch.rand(batch_dims, device=device)
        noisy = (1 - t[:,None,None]) * batch['all_atom_position'] + t[:,None,None] * ny
        
        # dists = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
        batch['noised_coords'] = noisy
        batch['t'] = t
    
    def training_step(self, batch, batch_idx):
        self.iter_step += 1
        device = batch[0]["aatype"].device
        batch_size = batch[0]['aatype'].shape[0]
        self.harmonic_prior.to(device)
        self.stage = 'train'
        if not self.args.no_ema:
            if(self.ema.device != device):
                self.ema.to(device)
        
        if torch.rand(1, generator=self.generator).item() < self.args.noise_prob:
            self.noise(batch[0])
            self.log('time', [batch[0]['t'].mean().item()])
        else:
            self.log('time', [1])

        if self.args.extra_input:
            if torch.rand(1, generator=self.generator).item() < self.args.extra_input_prob:
                pass
            else:
                del batch['extra_all_atom_positions']
        
        outputs = None
        if torch.rand(1, generator=self.generator).item() < self.args.self_cond_prob:  
            with torch.no_grad():
                outputs = self.model(batch[0])
        
        outputs = self.model(batch[0], prev_outputs=outputs)

        loss, loss_breakdown = self.loss(outputs, batch[1], _return_breakdown=True)

        with torch.no_grad():
            metrics = self._compute_validation_metrics(batch[1], outputs, superimposition_metrics=False)
        
        
        for k, v in loss_breakdown.items():
            self.log(k, [v.item()])
        for k, v in metrics.items():
            self.log(k, [v.item()])

        self.log('dur', [time.time() - self.last_log_time])
        self.last_log_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.args.no_ema:
            if(self.cached_weights is None):
                self.load_ema_weights()
            
        if self.args.normal_validate:
            self.training_step(batch, batch_idx, 'val')
            if self.args.validate:
                self.try_print_log()
            return 
            
        self.iter_step += 1
        self.stage = 'val'
        # At the start of validation, load the EMA weights
            
        ref_prot = batch['ref_prot'][0]
        
        pred_prots = []
        for _ in range(self.args.val_samples):
            if self.args.distillation:
                prots = self.inference(batch, no_diffusion=True, noisy_first=True, as_protein=True)
            else:
                prots = self.inference(batch, as_protein=True)
            pred_prots.append(prots[-1])

        first_metrics = protein.global_metrics(ref_prot, prots[0])
        for key in first_metrics:
            self.log('first_ref_'+key, [first_metrics[key]])

        ref_metrics = []
        for pred_prot in pred_prots:
            ref_metrics.append(protein.global_metrics(ref_prot, pred_prot, lddt=True))

        self_metrics = []
        for i, pred_prot1 in enumerate(pred_prots):
            pred_prot2 = pred_prots[(i+1) % len(pred_prots)]
            self_metrics.append(protein.global_metrics(pred_prot1, pred_prot2, lddt=True))
        
        self.log('name', batch['name'])
        
        ref_metrics = pd.DataFrame(ref_metrics)
        for key in ref_metrics:
            self.log('mean_ref_'+key, [ref_metrics[key].mean()])
            self.log('max_ref_'+key, [ref_metrics[key].max()]) 
            self.log('min_ref_'+key, [ref_metrics[key].min()]) 
        
        self_metrics = pd.DataFrame(self_metrics)
        for key in self_metrics:
            self.log('self_'+key, [self_metrics[key].mean()])
        
        if self.args.validate:
            self.try_print_log()







