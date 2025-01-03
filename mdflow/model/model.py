import torch
import torch.nn as nn

class MDFlow(nn.Module):
    '''
    Implements MDFlow model parts.
    '''

    def __init__(self, config, extra_input=False):
        pass
    
    '''
    function for patching the ensemble of structures.

    convert ensemble of structures to 1D sequence.
    '''
    def patchify(self, ensemble_of_structures):
        pass

    '''
    function for cross attention: TBD IF WE NEED

    inputs: sequence embeddings + positional embeddings
    outputs: ______
    '''
    def cross_attention(self, seq_embs, pos_embs):
        pass

    '''
    function for structure -> structure attention.

    input: sequence embeddings, structure at n timestep
    output: structure at n + 1 timestep

    implementation of structure module from AF + evoformer
    '''
    def structure_to_structure_attention(self, seq_embs, curr_structure, timestep):
        pass
    

    