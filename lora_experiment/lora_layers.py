"""
LoRA (Low-Rank Adaptation) layers for Open-Sora STDiT3 model.

LoRA adds trainable low-rank matrices to frozen model weights:
    h = W0*x + (B @ A) * x * scaling
    
Where:
    - W0: Original frozen weights
    - A: Low-rank down-projection (d x r)
    - B: Low-rank up-projection (r x d)  
    - r: Rank (much smaller than d)
    - scaling: alpha / r

Reference: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math


class LoRALinear(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.
    
    Args:
        original_layer: The original nn.Linear layer to wrap
        rank: Rank of the low-rank matrices (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout probability for LoRA layers (default: 0.0)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA matrices
        # A: down projection (in_features -> rank)
        # B: up projection (rank -> out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights using Kaiming uniform for A and zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        result = self.original_layer(x)
        
        # LoRA forward pass
        # x @ A^T @ B^T * scaling
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (batch, seq, rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (batch, seq, out_features)
        lora_out = lora_out * self.scaling
        
        return result + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into the original layer for inference."""
        with torch.no_grad():
            # W_merged = W0 + B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_w
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the original layer."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data -= delta_w


class LoRAQKV(nn.Module):
    """
    LoRA wrapper for the combined QKV projection layer.
    
    Applies separate LoRA adapters to Q, K, V projections.
    """
    
    def __init__(
        self,
        original_qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        enable_lora: List[bool] = [True, True, True],  # [Q, K, V]
    ):
        super().__init__()
        
        self.original_qkv = original_qkv
        self.in_features = original_qkv.in_features
        self.out_features = original_qkv.out_features
        assert self.out_features == self.in_features * 3, "QKV layer should output 3x input dim"
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enable_lora = enable_lora
        
        # Freeze the original layer
        for param in self.original_qkv.parameters():
            param.requires_grad = False
        
        # Create LoRA matrices for Q, K, V separately
        self.lora_A_q = nn.Parameter(torch.zeros(rank, self.in_features)) if enable_lora[0] else None
        self.lora_B_q = nn.Parameter(torch.zeros(self.in_features, rank)) if enable_lora[0] else None
        
        self.lora_A_k = nn.Parameter(torch.zeros(rank, self.in_features)) if enable_lora[1] else None
        self.lora_B_k = nn.Parameter(torch.zeros(self.in_features, rank)) if enable_lora[1] else None
        
        self.lora_A_v = nn.Parameter(torch.zeros(rank, self.in_features)) if enable_lora[2] else None
        self.lora_B_v = nn.Parameter(torch.zeros(self.in_features, rank)) if enable_lora[2] else None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights."""
        for A, B in [(self.lora_A_q, self.lora_B_q), 
                     (self.lora_A_k, self.lora_B_k),
                     (self.lora_A_v, self.lora_B_v)]:
            if A is not None:
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                nn.init.zeros_(B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original QKV projection
        qkv = self.original_qkv(x)
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply LoRA to each
        x_dropped = self.dropout(x)
        device = x.device
        dtype = x.dtype
        
        if self.lora_A_q is not None:
            lora_A_q = self.lora_A_q.to(device=device, dtype=dtype)
            lora_B_q = self.lora_B_q.to(device=device, dtype=dtype)
            q = q + F.linear(F.linear(x_dropped, lora_A_q), lora_B_q) * self.scaling
        
        if self.lora_A_k is not None:
            lora_A_k = self.lora_A_k.to(device=device, dtype=dtype)
            lora_B_k = self.lora_B_k.to(device=device, dtype=dtype)
            k = k + F.linear(F.linear(x_dropped, lora_A_k), lora_B_k) * self.scaling
        
        if self.lora_A_v is not None:
            lora_A_v = self.lora_A_v.to(device=device, dtype=dtype)
            lora_B_v = self.lora_B_v.to(device=device, dtype=dtype)
            v = v + F.linear(F.linear(x_dropped, lora_A_v), lora_B_v) * self.scaling
        
        # Concatenate back
        return torch.cat([q, k, v], dim=-1)


def inject_lora_into_attention(
    attn_module: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: List[str] = ["qkv", "proj"],
) -> Dict[str, nn.Module]:
    """
    Inject LoRA layers into an attention module.
    
    Args:
        attn_module: The attention module (e.g., Attention from blocks.py)
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to
    
    Returns:
        Dict of original modules replaced by LoRA versions
    """
    lora_modules = {}
    
    if "qkv" in target_modules and hasattr(attn_module, "qkv"):
        original_qkv = attn_module.qkv
        lora_qkv = LoRAQKV(original_qkv, rank=rank, alpha=alpha, dropout=dropout)
        attn_module.qkv = lora_qkv
        lora_modules["qkv"] = lora_qkv
    
    if "proj" in target_modules and hasattr(attn_module, "proj"):
        original_proj = attn_module.proj
        lora_proj = LoRALinear(original_proj, rank=rank, alpha=alpha, dropout=dropout)
        attn_module.proj = lora_proj
        lora_modules["proj"] = lora_proj
    
    return lora_modules


def inject_lora_into_stdit3(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: List[str] = ["qkv", "proj"],
    target_blocks: str = "all",  # "all", "spatial", "temporal"
) -> Dict[str, Dict[str, nn.Module]]:
    """
    Inject LoRA layers into STDiT3 model.
    
    Args:
        model: STDiT3 model
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout for LoRA layers
        target_modules: Which attention modules to apply LoRA to
        target_blocks: Which blocks to apply LoRA to ("all", "spatial", "temporal")
    
    Returns:
        Dict mapping block names to their LoRA modules
    """
    all_lora_modules = {}
    
    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Inject LoRA into spatial blocks
    if target_blocks in ["all", "spatial"]:
        if hasattr(model, "spatial_blocks"):
            for i, block in enumerate(model.spatial_blocks):
                block_name = f"spatial_block_{i}"
                all_lora_modules[block_name] = {}
                
                # Self-attention
                if hasattr(block, "attn"):
                    lora_mods = inject_lora_into_attention(
                        block.attn, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
                    )
                    all_lora_modules[block_name]["attn"] = lora_mods
                
                # Cross-attention
                if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "proj"):
                    lora_proj = LoRALinear(block.cross_attn.proj, rank=rank, alpha=alpha, dropout=dropout)
                    block.cross_attn.proj = lora_proj
                    all_lora_modules[block_name]["cross_attn_proj"] = lora_proj
    
    # Inject LoRA into temporal blocks
    if target_blocks in ["all", "temporal"]:
        if hasattr(model, "temporal_blocks") and model.temporal_blocks is not None:
            for i, block in enumerate(model.temporal_blocks):
                block_name = f"temporal_block_{i}"
                all_lora_modules[block_name] = {}
                
                # Self-attention
                if hasattr(block, "attn"):
                    lora_mods = inject_lora_into_attention(
                        block.attn, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
                    )
                    all_lora_modules[block_name]["attn"] = lora_mods
                
                # Cross-attention
                if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "proj"):
                    lora_proj = LoRALinear(block.cross_attn.proj, rank=rank, alpha=alpha, dropout=dropout)
                    block.cross_attn.proj = lora_proj
                    all_lora_modules[block_name]["cross_attn_proj"] = lora_proj
    
    return all_lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count LoRA and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100 * trainable_params / total_params,
    }


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights to a file."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data.clone()
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """Load LoRA weights from a file."""
    lora_state_dict = torch.load(path, map_location="cpu")
    
    model_state = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        elif strict:
            raise KeyError(f"LoRA weight {name} not found in model")
    
    model.load_state_dict(model_state)

