import torch.nn as nn
import torch
# -------------------------------------------------------------------
# CUSTOM PRIMITIVES, LAYERS, AND HELPERS
# -------------------------------------------------------------------
class InputPairStack(nn.Module):
    """
    Example class that processes pairwise data (e.g., distances) into an embedding.
    In real code, this might have multiple layers (MLPs, attention, etc.).
    Here, it's just a placeholder that returns the data or lightly transforms it.
    """
    def __init__(self, c_in=1, c_out=32, num_layers=1):
        super().__init__()
        # For demonstration, store a list of linear transforms
        self.layers = nn.ModuleList([
            nn.Linear(c_in if i == 0 else c_out, c_out)
            for i in range(num_layers)
        ])
    def forward(self, pairwise_data, pairwise_mask=None):
        """
        pairwise_data: Tensor shaped like [B, L, L] or [B, L, L, c_in]
        pairwise_mask: Tensor shaped like [B, L] or [B, L, L], etc.
        Returns: [B, L, L, c_out]
        """
        # Reshape if input is [B, L, L], treat as [B, L, L, 1] for linear ops
        if pairwise_data.dim() == 3:
            pairwise_data = pairwise_data.unsqueeze(-1)  # [B, L, L, 1]
        x = pairwise_data
        # A trivial pass through a few linear layers
        for layer in self.layers:
            # Flatten the last dimension to apply a Linear
            B, L, L2, C = x.shape
            x = x.view(B*L*L2, C)  # Flatten
            x = layer(x)
            x = torch.relu(x)      # Some nonlinearity
            x = x.view(B, L, L2, -1)
        return x
class GaussianFourierProjection(nn.Module):
    """
    Used to encode continuous variables (like time t) into a sinusoidal embedding
    using random frequencies. Common in diffusion/flow-based models.
    """
    def __init__(self, embedding_size=128, scale=1.0):
        super().__init__()
        self.embedding_size = embedding_size
        # For simplicity, create a fixed random frequency matrix:
        # half of embedding_size is sine, half is cosine
        self.B = nn.Parameter(torch.randn(embedding_size // 2) * scale,
                              requires_grad=False)
    def forward(self, t):
        """
        t: [B] or [B, 1] — a scalar for each batch element (time or noise level).
        Returns: [B, embedding_size], a sinusoidal embedding.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        # Multiply t by the random frequencies
        proj = t * self.B  # shape [B, embedding_size//2]
        # Sine and cosine
        sin_part = torch.sin(proj)
        cos_part = torch.cos(proj)
        # Concatenate
        fourier = torch.cat([sin_part, cos_part], dim=-1)
        return fourier
class Linear(nn.Module):
    """
    A custom Linear layer (similar to openfold.model.primitives.Linear)
    that might have specialized initialization.
    """
    def __init__(self, in_dim, out_dim, init="default", bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.init = init
        self._init_weights()
    def _init_weights(self):
        if self.init == "final":
            # Possibly a special scaled initialization
            nn.init.xavier_uniform_(self.weight, gain=0.1)
        else:
            # Default or some other approach
            nn.init.xavier_uniform_(self.weight)
    def forward(self, x):
        # x shape: [..., in_dim]
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y