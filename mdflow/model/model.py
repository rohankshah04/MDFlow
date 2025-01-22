### file @parsa for the model wrapper stuff
import torch, os, wandb, time, sys
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

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs if needed
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)  # Adjust logging level here
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

class Model(pl.LightningModule):
    def __init__(self, config, args):
        super().__init__()
        self.model = AlphaFold(config)
        self.args = args
        self.cached_weights = None
        self.iter_step = 0
        self.HarmonicPrior = HarmonicPrior(config.data.train.crop_size)
        self.generator = torch.Generator().manual_seed(137)
        self.last_log_time = time.time()
        pass

    def noise(self, batch):
        device = batch['aatype'].device
        batch_dims = batch[''].shape
        ny = self.harmonic_prior.sample(batch_dims)
        t = torch.rand(batch_dims, device=device)
        noised_structure = (1 - t[:,None,None]) * batch['all_atom_position'] + t[:,None,None] * ny
        
        batch['noised_structre'] = noised_structure
        batch['t'] = t
    
    def training_step(self, batch, batch_idx):
        self.iter_step += 1
        device = batch[0]["aatype"].device
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

        # batch['temp_pos'] = batch_idx
        batch[0]['prev_outputs'] = outputs
        outputs = self.model(batch[0])

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
            
        
        self.iter_step += 1
        self.stage = 'val'
        batch[1]['aatype'] = torch.argmax(batch[1]['aatype'], dim=-1)
        logger.info(f"shape of aatype: {batch[1]['aatype'].shape}")

        ref_prot = protein.output_to_protein(batch[1])
        pred_prots = []

        for _ in range(self.args.val_samples):
            prots = self.inference(batch[0], as_protein=True)
            proteins = protein.output_to_protein(prots)
            pred_prots.append(proteins[-1])

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

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling 
        # load_state_dict().
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])
        logger.info('Finished loading EMA weights')
    
    def inference(self, batch, as_protein=False, no_diffusion=False, self_cond=True, noisy_first=False, schedule=None):
        device = batch['aatype'].device
        batch_dims = batch['all_atom_positions'].shape[0]
        ny = HarmonicPrior(batch_dims)
        # t = torch.rand(batch_dims, device=device)
        # noised_structure = (1 - t[:,None,None]) * batch['all_atom_position'] + t[:,None,None] * ny
        
        if noisy_first:
            batch['noised_structure'] = ny
            batch['t'] = torch.ones(1, device=noisy.device)
            
        if no_diffusion:
            logger.info(f"making call to model for inference")
            output = self.model(batch)
            logger.info(f"received output from model for inference")
            if as_protein:
                return protein.output_to_protein({**output, **batch})
            else:
                return [{**output, **batch}]

        if schedule is None:
            schedule = np.array([1.0, 0.75, 0.5, 0.25, 0.1, 0]) 
        outputs = []
        for t, s in zip(schedule[:-1], schedule[1:]):
            logger.info(f"making call to model for inference")
            logger.info(f"original batch keys: {batch.keys()}")
            output = self.model(batch)
            logger.info(f"received output from model for inference")
            outputs.append({**output, **batch})
            
            ny = (s / t) * ny + (1 - s / t) * batch['all_atom_position']
            batch['noised_struture'] = ny
            batch['t'] = torch.ones(1, device=ny.device) * s # first one doesn't get the time embedding, last one is ignored :)
            if self_cond:
                batch['prev_outputs'] = outputs

        del batch['noised_structure'], batch['t']
        if as_protein:
            prots = []
            for output in outputs:
                prots.extend(protein.output_to_protein(output))
            return prots
        else:
            return outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        
        lr_scheduler = AlphaFoldLRScheduler(optimizer, max_lr=self.args.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }








