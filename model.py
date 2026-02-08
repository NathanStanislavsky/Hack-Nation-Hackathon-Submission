import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiResEmbedding(nn.Module):
    def __init__(self, n_channels, resolutions, dim):
        super().__init__()
        self.n_channels = n_channels
        # Ensure resolutions is a list for iteration
        if isinstance(resolutions, int): resolutions = [resolutions]
        self.resolutions = torch.tensor(resolutions)
        
        # Calculate total vocab size based on bins per channel
        # (res + 1) bins because N cuts create N+1 regions
        res_sizes = n_channels * (self.resolutions + 1)
        total_vocab_size = res_sizes.sum().item()
        
        # MODE='SUM' is usually better for learning distinct features than 'MEAN'
        self.bag = nn.EmbeddingBag(total_vocab_size, dim, mode='sum')
        
        # Calculate offsets so every channel/resolution gets unique embedding IDs
        global_offsets = torch.cat([torch.tensor([0]), res_sizes.cumsum(0)[:-1]])
        self.register_buffer("global_offsets", global_offsets)
        
        for i, res in enumerate(resolutions):
            # Bin boundaries: 0.0 to 1.0
            self.register_buffer(f"b_{i}", torch.linspace(0, 1, res))
            # Channel offsets: shift by (res+1) for each channel
            self.register_buffer(f"o_{i}", torch.arange(n_channels) * (res + 1))

    def forward(self, features):
        # features shape: (Batch, n_channels)
        indices = []
        for i in range(len(self.resolutions)):
            # 1. Bucketize: maps float values to bin indices
            # 2. Add o_{i}: offsets bin index by channel ID
            idx = torch.bucketize(features, getattr(self, f"b_{i}")) + getattr(self, f"o_{i}")
            
            # 3. Add global_offsets: offsets by resolution block
            indices.append(idx + self.global_offsets[i])
        
        # Flatten for the bag: (Batch, Total_Indices)
        indices = torch.cat(indices, dim=-1).reshape(features.size(0), -1)
        return self.bag(indices)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout), # Added Dropout
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)  # Added Dropout
        )

    def forward(self, x):
        return self.net(x)

class SDPA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # --- MISSING PIECE 1: Projections ---
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = dropout

        # Precompute RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def apply_rope(self, x):
        # x shape: (Batch, Seq, Heads, Head_Dim)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq) # (Seq, Head_Dim/2)
        
        # Reshape freqs for broadcasting: (1, Seq, 1, Head_Dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.unsqueeze(0).unsqueeze(2) 
        
        # Rotate
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x * emb.cos()) + (x_rotated * emb.sin())

    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 1. Project
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. Split Heads (Batch, Seq, Heads, Head_Dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        # 3. Apply RoPE
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        # 4. Attention (Transpose for PyTorch API: Batch, Heads, Seq, Dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Using built-in SDPA (Flash Attention compatible)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        
        # 5. Reassemble and Output Project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = SDPA(dim, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BrainWaveIntentModel(nn.Module):
    def __init__(self, num_layers=4, dim=32, mlp_ratio=4, num_classes=5):
        super().__init__()
        # EEG: 6 Channels -> 1 Token
        self.eeg_table = MultiResEmbedding(n_channels=6, resolutions=[8], dim=dim)
        
        # Moment: 720 Channels -> 1 Token
        # NOTE: 720 channels is A LOT for one embedding bag. 
        # Ideally, use nn.Linear(720, dim) if inputs are continuous.
        self.moment_table = MultiResEmbedding(n_channels=720, resolutions=[8], dim=dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, dim * mlp_ratio) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # Init weights (Important for Transformers!)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.EmbeddingBag):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, eeg_features, moment_features):
        # Ensure inputs are [0, 1] for MultiResEmbedding bucketize!
        
        eeg_emb = self.eeg_table(eeg_features)       # (Batch, Dim)
        moment_emb = self.moment_table(moment_features) # (Batch, Dim)
        
        # Create Sequence: (Batch, 2, Dim)
        x = torch.stack([eeg_emb, moment_emb], dim=1)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        # Average over sequence length (2)
        return self.head(x.mean(dim=1))