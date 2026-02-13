import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block with explicit hook points for the residual stream.
    """
    def __init__(self, hidden_dim: int, resblock_width: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.resblock_width = resblock_width
        self.dropout = dropout
        
        # 1. Define Hook Points (Identity layers)
        # These are no-ops computationally, but allow you to attach hooks 
        # specifically to the stream "x" before and after processing.
        self.hook_resid_pre = nn.Identity()
        self.hook_resid_post = nn.Identity()
        self.hook_mlp_out = nn.Identity()
        
        # 2. Block Layers
        self.ln1 = nn.LayerNorm(hidden_dim)        
        self.fc1 = nn.Linear(hidden_dim, resblock_width)

        self.ln2 = nn.LayerNorm(resblock_width)
        self.fc2 = nn.Linear(resblock_width, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        
        out = self.ln1(x)
        out = F.relu(out)
        out = self.fc1(out)
        
        out = self.dropout(out)
        
        out = self.ln2(out)
        out = F.relu(out)
        out = self.fc2(out)

        out = self.hook_mlp_out(out)
        
        x = x + out
        
        x = self.hook_resid_post(x)
        
        return x

class ResNetMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, resblock_width: int, num_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.resblock_width = resblock_width
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.hook_input = nn.Identity()
                
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hook_embed = nn.Identity()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, resblock_width, dropout) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_input(x)
        x = self.input_layer(x)
        x = self.hook_embed(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        x = F.relu(x)
        x = self.output_layer(x)
        # if self.output_dim == 1:
        #     x = x.squeeze(1)
        return x



class ResNetMLPSkeleton(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, resblock_width: int, num_blocks: int, dropout: float = 0.0):
        """input dimension is the hidden dimension: R^n --> R^n mapping"""
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.resblock_width = resblock_width
        self.num_blocks = num_blocks
        self.dropout = dropout
        
        self.hook_input = nn.Identity()
        self.hook_embed = nn.Identity()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim=input_dim,
                resblock_width=resblock_width,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_input(x)
        x = self.hook_embed(x)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x