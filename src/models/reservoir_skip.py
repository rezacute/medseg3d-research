"""
Classical ESN Reservoir Skip Module for Medical Image Segmentation

Self-contained module implementing:
- ReservoirSkip: Echo State Network (ESN) based reservoir computing
- FMQEReservoirSkip: Frequency Modulation Quantum Encoding (quantum-inspired)
- HybridReservoirSkip: Combined ESN + FMQE in parallel
- SkipWrapper: Helper to inject reservoir processing into U-Net decoders

No external dependencies beyond PyTorch. CPU-compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ReservoirSkip(nn.Module):
    """
    Echo State Network (ESN) Reservoir for processing skip connections.
    
    The reservoir is applied per-spatial-position, compressing spatial dims
    into a fixed-size reservoir state, then expanding back. The residual
    connection preserves original spatial information.
    
    Architecture:
    - Input: (B, C, D, H, W) or (B, C, H, W)
    - Pool W axis -> (B, C, D, H) or (B, C, H)
    - Project C -> reservoir_size, apply ESN dynamics, project back to C
    - Add residual: output = input + processed
    
    Args:
        channels: Number of input/output channels
        reservoir_size: Size of reservoir hidden state
        spectral_radius: Spectral radius of W_res (controls chaos, default 0.9)
        input_scaling: Scaling of input weights (default 1.0)
        leak_rate: Leak rate for reservoir dynamics (default 1.0)
    """
    
    def __init__(
        self,
        channels: int,
        reservoir_size: int = 256,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
    ):
        super().__init__()
        
        self.channels = channels
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        
        # Fixed reservoir weights (not trainable)
        self._init_reservoir_weights()
        
        # Fixed input weights (not trainable): project channels -> reservoir_size
        self._init_input_weights(input_scaling)
        
        # Trainable readout: reservoir_size -> channels
        self.readout = nn.Linear(reservoir_size, channels)
        
        # Ensure fixed weights stay fixed
        self.W_res.requires_grad = False
        self.W_in.requires_grad = False
    
    def _init_reservoir_weights(self):
        W = torch.randn(self.reservoir_size, self.reservoir_size)
        try:
            _, S, _ = torch.svd(W)
            spectral_norm = S[0]
        except Exception:
            spectral_norm = 1.0
        self.W_res = nn.Parameter(
            (W / (spectral_norm + 1e-8)) * self.spectral_radius,
            requires_grad=False
        )
    
    def _init_input_weights(self, input_scaling: float):
        # Project from channels to reservoir_size
        W_in = torch.randn(self.reservoir_size, self.channels) * input_scaling
        self.W_in = nn.Parameter(W_in, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) or (B, C, H, W)
        Returns:
            (B, C, D, H, W) or (B, C, H, W) with residual added
        """
        is_5d = len(x.shape) == 5
        
        if is_5d:
            B, C, D, H, W = x.shape
            # Pool W: (B, C, D, H, W) -> (B, C, D, H)
            x_pooled = x.mean(dim=-1)  # (B, C, D, H)
            spatial_prod = D * H
            x_2d = x_pooled.view(B, C, spatial_prod)  # (B, C, D*H)
        else:
            B, C, H, W = x.shape
            x_pooled = x.mean(dim=-1)  # (B, C, H)
            spatial_prod = H
            x_2d = x_pooled  # (B, C, H)
        
        # Transpose to (B, spatial, C) for matrix multiply
        x_2d = x_2d.transpose(1, 2)  # (B, spatial, C)
        
        # Project: (B, spatial, C) @ (reservoir_size, C).T -> (B, spatial, reservoir_size)
        x_proj = torch.matmul(x_2d, self.W_in.T)  # (B, spatial, reservoir_size)
        
        # ESN dynamics: apply per sample in batch
        batch_size, spatial, reservoir_dim = x_proj.shape
        state = torch.zeros(batch_size, spatial, self.reservoir_size, device=x.device, dtype=x.dtype)
        
        for _ in range(3):
            state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(
                x_proj + torch.matmul(state, self.W_res)
            )
        
        # Readout: (B, spatial, reservoir_size) -> (B, spatial, channels)
        readout_out = self.readout(state)  # (B, spatial, channels)
        
        # Transpose back: (B, spatial, channels) -> (B, channels, spatial)
        readout_out = readout_out.transpose(1, 2)  # (B, C, spatial)
        
        # Reshape to spatial dimensions and broadcast residual
        if is_5d:
            out = readout_out.view(B, C, D, H).unsqueeze(-1)  # (B, C, D, H, 1)
        else:
            out = readout_out.view(B, C, H).unsqueeze(-1)  # (B, C, H, 1)
        
        return x + out


class FMQEReservoirSkip(nn.Module):
    """
    Frequency Modulation Quantum Encoding (FM-QE) Reservoir.
    
    Uses sinusoidal encoding with fixed random frequencies. Each spatial
    position gets independently encoded via FM-QE, then projected back.
    
    Architecture:
    - Pool W axis -> (B, C, D, H) or (B, C, H)
    - Apply per-channel FM encoding: sin(freq * x), cos(freq * x)
    - Project C*2*n_freq -> C
    - Add residual
    
    Args:
        channels: Number of input/output channels
        n_frequencies: Number of frequency components per channel (default 12)
    """
    
    def __init__(
        self,
        channels: int,
        n_frequencies: int = 4,
        bottleneck_dim: int = 16,
    ):
        super().__init__()
        
        self.channels = channels
        self.n_frequencies = n_frequencies
        
        # Fixed frequency matrix: (channels, n_frequencies)
        self.frequencies = nn.Parameter(
            torch.randn(channels, n_frequencies) * math.sqrt(2),
            requires_grad=False
        )
        
        # Trainable readout with bottleneck: (C * 2 * n_freq) -> bottleneck_dim -> C
        # For n_freq=4, bottleneck_dim=16: input is C * 8, gives ~110K total params
        self.readout = nn.Sequential(
            nn.Linear(channels * 2 * n_frequencies, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) [5D], (B, C, H, W) [4D], or (B, spatial, C) [3D pooled]
        Returns:
            Same shape as input with residual added
        """
        ndims = len(x.shape)
        
        if ndims == 3:
            # Already pooled: (B, spatial, C)
            B, spatial_prod, C = x.shape
            x_spatial = x
        elif ndims == 5:
            # 5D: (B, C, D, H, W)
            B, C, D, H, W = x.shape
            x_pooled = x.mean(dim=-1)  # (B, C, D, H)
            spatial_prod = D * H
            x_spatial = x_pooled.view(B, C, spatial_prod).transpose(1, 2)  # (B, spatial, C)
        else:
            # 4D: (B, C, H, W)
            B, C, H, W = x.shape
            x_pooled = x.mean(dim=-1)  # (B, C, H)
            spatial_prod = H
            x_spatial = x_pooled.transpose(1, 2)  # (B, H, C)
        
        # FM encoding: (B, spatial, C) * (C, n_freq) -> (B, spatial, C, n_freq)
        x_expanded = x_spatial.unsqueeze(-1)  # (B, spatial, C, 1)
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(0)  # (1, 1, C, n_freq)
        phase = x_expanded * freq_expanded  # (B, spatial, C, n_freq)
        sin_enc = torch.sin(phase)
        cos_enc = torch.cos(phase)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, spatial, C, 2*n_freq)
        encoded = encoded.reshape(B, spatial_prod, C * 2 * self.n_frequencies)
        readout_out = self.readout(encoded)  # (B, spatial, C)
        readout_out = readout_out.transpose(1, 2)  # (B, C, spatial)
        
        # Expand back to original spatial dimensions
        if ndims == 3:
            residual = readout_out.squeeze(0)  # (C, spatial)
            return x + residual
        elif ndims == 5:
            residual = readout_out.view(B, C, D, H).unsqueeze(-1)  # (B, C, D, H, 1)
            residual = residual.expand(B, C, D, H, W)  # (B, C, D, H, W)
        else:
            residual = readout_out.view(B, C, H).unsqueeze(-1)  # (B, C, H, 1)
            residual = residual.expand(B, C, H, W)  # (B, C, H, W)
        
        return x + residual


class HybridReservoirSkip(nn.Module):
    """
    Hybrid combining ESN + FMQE in parallel.
    
    Both branches process the skip tensor independently, producing
    residuals that are summed. A shared readout combines their
    pooled representations for the final residual.
    
    Args:
        channels: Number of input/output channels
        reservoir_size: Size of ESN reservoir (default 128)
        n_frequencies: Number of FMQE frequencies (default 4)
    """
    
    def __init__(
        self,
        channels: int,
        reservoir_size: int = 128,
        n_frequencies: int = 4,
    ):
        super().__init__()
        
        self.esn = ReservoirSkip(channels, reservoir_size)
        self.fmqe = FMQEReservoirSkip(channels, n_frequencies)
        
        # Shared readout with bottleneck: (2C) -> 16 -> C
        # ~16*C*2 + 16*C params ≈ 5K per level (vs 378K without bottleneck)
        self.readout = nn.Sequential(
            nn.Linear(channels * 2, 16),
            nn.GELU(),
            nn.Linear(16, channels),
        )
    
    @property
    def n_frequencies(self):
        return self.fmqe.n_frequencies
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) or (B, C, H, W)
        Returns:
            (B, C, D, H, W) or (B, C, H, W) with residual added
        """
        is_5d = len(x.shape) == 5
        
        # Both branches add residuals internally
        esn_out = self.esn(x)
        fmqe_out = self.fmqe(x)
        
        # Pool both outputs and combine through readout
        if is_5d:
            B, C, D, H, W = x.shape
            esn_pooled = esn_out.mean(dim=-1).mean(dim=-1).mean(dim=-1)  # (B, C)
            fmqe_pooled = fmqe_out.mean(dim=-1).mean(dim=-1).mean(dim=-1)  # (B, C)
        else:
            B, C, H, W = x.shape
            esn_pooled = esn_out.mean(dim=-1).mean(dim=-1)  # (B, C)
            fmqe_pooled = fmqe_out.mean(dim=-1).mean(dim=-1)  # (B, C)
        
        combined = torch.cat([esn_pooled, fmqe_pooled], dim=-1)  # (B, 2*C)
        readout_out = self.readout(combined)  # (B, C)
        
        # Expand back to spatial shape
        if is_5d:
            D_H = D * H
            out = readout_out.view(B, C, 1, 1).expand(B, C, D, H).unsqueeze(-1)
        else:
            out = readout_out.view(B, C, 1).expand(B, C, H).unsqueeze(-1)
        
        return x + out


class SkipWrapper(nn.Module):
    """
    Wraps an existing decoder block to add reservoir processing on skip input.
    
    Args:
        original_decoder_block: forward(x, skip) -> output
        skip_processor: ReservoirSkip, FMQEReservoirSkip, or HybridReservoirSkip
    """
    
    def __init__(
        self,
        original_decoder_block: nn.Module,
        skip_processor: nn.Module,
    ):
        super().__init__()
        self.original_decoder_block = original_decoder_block
        self.skip_processor = skip_processor
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        enriched_skip = self.skip_processor(skip)
        return self.original_decoder_block(x, enriched_skip)
    
    def extra_repr(self) -> str:
        return f"skip_processor={self.skip_processor.__class__.__name__}"


class ReservoirDecoder(nn.Module):
    """
    Applies reservoir processing to a list of skip connection tensors.
    """
    
    def __init__(
        self,
        decoder_channels: list,
        n_reservoir: int = 256,
        n_frequencies: int = 12,
        reservoir_type: str = 'esn',
    ):
        super().__init__()
        self.skip_processors = nn.ModuleList()
        for channels in decoder_channels:
            if reservoir_type == 'esn':
                processor = ReservoirSkip(channels, n_reservoir)
            elif reservoir_type == 'fmqe':
                processor = FMQEReservoirSkip(channels, n_frequencies)
            elif reservoir_type == 'hybrid':
                processor = HybridReservoirSkip(channels, n_reservoir, n_frequencies)
            else:
                raise ValueError(f"Unknown reservoir_type: {reservoir_type}")
            self.skip_processors.append(processor)
    
    def forward(self, skip_features: list) -> list:
        return [processor(sf) for sf, processor in zip(skip_features, self.skip_processors)]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def get_trainable_params(module: nn.Module) -> dict:
    return {name: p.shape for name, p in module.named_parameters() if p.requires_grad}


def get_fixed_params(module: nn.Module) -> dict:
    return {name: p.shape for name, p in module.named_parameters() if not p.requires_grad}
