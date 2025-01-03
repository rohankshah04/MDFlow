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
from .input_stack import InputPairStack
from .layers import GaussianFourierProjection
from openfold.model.primitives import Linear


class AlphaFold(nn.Module):
    '''
    Implements AF2 modules with training.

    two main funcs:
    1) input sequence -> embeddings
    2) evoformer, recycling module
    '''

    def __init__(self, config, extra_input=False):
        pass
