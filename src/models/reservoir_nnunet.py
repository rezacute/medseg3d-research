"""
reservoir_nnunet.py - Integrate Reservoir Skip Connections into nnU-Net

Properly hooks into encoder-decoder boundary to add reservoir processing
at skip connection points.

Architecture:
    Input -> Encoder (frozen) -> [Skip outputs at levels 1-4]
                                   |
                                   v
                         [Reservoir modules process skips]
                                   |
                                   v
                         Decoder (frozen) <- [enriched skip outputs]

Usage:
    from reservoir_nnunet import ReservoirNNUNet, load_reservoir_nnunet
    
    # Load trained model
    model, trainer = load_nnunet_model(checkpoint_path)
    
    # Wrap with reservoir skips
    reservoir_model = ReservoirNNUNet(
        base_model=model,
        skip_type='esn',
        skip_levels=[1, 2, 3, 4],
    )
    
    # Train only reservoir params
    optimizer = Adam(reservoir_model.reservoir_params, lr=1e-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from src.models.reservoir_skip import ReservoirSkip, FMQEReservoirSkip, HybridReservoirSkip


class ReservoirNNUNet(nn.Module):
    """
    nnU-Net with Reservoir Skip Connections.
    
    The reservoir modules are inserted at the encoder-decoder boundary,
    processing skip connections before they enter the decoder.
    
    Only reservoir readout layers are trainable (~60-120K params).
    Base nnU-Net weights remain frozen.
    """
    
    # Channel counts at each encoder level
    CHANNELS = [32, 64, 128, 256, 320, 320]
    
    # Default skip levels (skip 0 = too high-res, skip 5 = bottleneck)
    DEFAULT_SKIP_LEVELS = [1, 2, 3, 4]
    
    def __init__(
        self,
        base_model: nn.Module,
        skip_type: str = 'esn',
        skip_levels: List[int] = None,
        reservoir_size: int = 128,
        n_frequencies: int = 12,
        alpha: float = 0.3,  # Residual blend ratio
    ):
        super().__init__()
        
        self.base_model = base_model
        self.skip_type = skip_type
        self.skip_levels = skip_levels or self.DEFAULT_SKIP_LEVELS
        self.reservoir_size = reservoir_size
        self.alpha = alpha
        
        # Freeze entire base model
        for p in base_model.parameters():
            p.requires_grad = False
        
        # Build reservoir modules for each skip level
        self.reservoirs = nn.ModuleDict()
        
        if skip_type != 'none':
            for lvl in self.skip_levels:
                ch = self.CHANNELS[lvl]
                
                if skip_type == 'esn':
                    self.reservoirs[str(lvl)] = ReservoirSkip(
                        channels=ch,
                        reservoir_size=reservoir_size,
                    )
                elif skip_type == 'fmqe':
                    self.reservoirs[str(lvl)] = FMQEReservoirSkip(
                        channels=ch,
                        n_frequencies=n_frequencies,
                    )
                elif skip_type == 'hybrid':
                    self.reservoirs[str(lvl)] = HybridReservoirSkip(
                        channels=ch,
                        reservoir_size=reservoir_size,
                        n_frequencies=n_frequencies,
                    )
                else:
                    raise ValueError(f"Unknown skip_type: {skip_type}")
        
        # Track reservoir parameters
        self._reservoir_params = None
        
        # Report architecture
        self._report_architecture()
    
    @property
    def reservoir_params(self):
        """Return generator of trainable reservoir parameters."""
        if self._reservoir_params is None:
            self._reservoir_params = [p for p in self.parameters() if p.requires_grad]
        return self._reservoir_params
    
    def _report_architecture(self):
        """Print architecture summary."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        print("\n" + "=" * 60)
        print(f"ReservoirNNUNet ({self.skip_type})")
        print("=" * 60)
        print(f"Base model: {type(self.base_model).__name__}")
        print(f"Skip levels: {self.skip_levels}")
        print(f"Alpha (residual blend): {self.alpha}")
        print(f"Total parameters: {total:,}")
        print(f"  Trainable (reservoir): {trainable:,}")
        print(f"  Frozen (nnU-Net): {frozen:,}")
        
        if self.skip_type != 'none':
            print(f"\nReservoir modules:")
            total_res = 0
            for lvl in self.skip_levels:
                ch = self.CHANNELS[lvl]
                mod = self.reservoirs[str(lvl)]
                n = sum(p.numel() for p in mod.parameters())
                total_res += n
                if hasattr(mod, 'reservoir_size'):
                    dim_str = f"{mod.reservoir_size:3d}n"
                elif hasattr(mod, 'n_frequencies'):
                    dim_str = f"{mod.n_frequencies:2d}f"
                else:
                    dim_str = "???"
                print(f"  Level {lvl}: {ch:3d}ch -> {dim_str} | {n:,} params")
            print(f"  Total reservoir: {total_res:,}")
        print("=" * 60 + "\n")
    
    def _process_skips(self, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process encoder skip outputs through reservoir modules.
        
        Args:
            skips: List of 6 tensors [skip0, skip1, ..., skip5]
                   skip0-5 from encoder stages 0-5
                   
        Returns:
            Modified skips list with selected levels processed through reservoirs
        """
        processed = []
        
        for lvl, skip in enumerate(skips):
            if lvl in self.skip_levels and self.skip_type != 'none':
                # Process through reservoir
                mod = self.reservoirs[str(lvl)]
                skip_out = mod(skip)
                
                # Residual blend: (1-alpha)*original + alpha*processed
                skip_out = (1 - self.alpha) * skip + self.alpha * skip_out
                
                processed.append(skip_out)
            else:
                # Pass through unchanged
                processed.append(skip)
        
        return processed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with reservoir-enriched skip connections.
        
        Flow:
            1. Encoder generates 6 skip outputs
            2. Selected skips [1-4] processed through reservoirs
            3. Modified skips fed to decoder
        """
        # Move reservoirs to same device as input
        for lvl in self.skip_levels:
            if str(lvl) in self.reservoirs:
                self.reservoirs[str(lvl)].to(x.device)
        
        # Run encoder
        encoder = self.base_model.encoder
        skips = encoder(x)
        
        # Process skips through reservoirs
        modified_skips = self._process_skips(skips)
        
        # Run decoder with modified skips
        decoder = self.base_model.decoder
        output = decoder(modified_skips)
        
        return output
    
    def train(self, mode: bool = True):
        """Override train to ensure base model stays in eval mode."""
        super().train(mode)
        # Keep base model in eval mode
        self.base_model.eval()
        return self
    
    def eval(self):
        """Override eval."""
        super().eval()
        self.base_model.eval()
        return self


def load_nnunet_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load trained nnU-Net model from checkpoint.
    
    Uses the same approach as nnUNetPredictor (no torch.compile).
    
    Args:
        checkpoint_path: Path to checkpoint_best.pth
        device: 'cpu' or 'cuda'
        
    Returns:
        Tuple of (model, trainer)
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from pathlib import Path
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Get the model training output directory
    checkpoint_path = Path(checkpoint_path)
    model_dir = checkpoint_path.parent.parent  # Go up from fold_X to dataset folder
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device(device),
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=[int(checkpoint_path.parent.name.split('_')[-1])],
        checkpoint_name=checkpoint_path.name,
    )
    
    model = predictor.network
    print(f"Loaded model: {type(model).__name__}")
    
    return model, predictor


def create_reservoir_nnunet(
    checkpoint_path: str,
    skip_type: str = 'esn',
    skip_levels: List[int] = None,
    reservoir_size: int = 128,
    n_frequencies: int = 12,
    alpha: float = 0.3,
    device: str = 'cpu',
) -> ReservoirNNUNet:
    """
    Create ReservoirNNUNet from checkpoint.
    
    Args:
        checkpoint_path: Path to trained nnU-Net checkpoint
        skip_type: 'esn', 'fmqe', 'hybrid', or 'none'
        skip_levels: Which encoder levels get reservoir processing
        reservoir_size: Hidden dimension of ESN reservoir
        n_frequencies: Number of FMQE frequencies per channel
        alpha: Residual blend ratio
        device: 'cpu' or 'cuda'
        
    Returns:
        ReservoirNNUNet with frozen base + trainable reservoir modules
    """
    # Load base model
    base_model, trainer = load_nnunet_model(checkpoint_path, device)
    
    # Create wrapper
    reservoir_model = ReservoirNNUNet(
        base_model=base_model,
        skip_type=skip_type,
        skip_levels=skip_levels or [1, 2, 3, 4],
        reservoir_size=reservoir_size,
        n_frequencies=n_frequencies,
        alpha=alpha,
    )
    
    return reservoir_model, trainer


# Quick test
if __name__ == '__main__':
    import os
    
    # Set nnUNet paths
    os.environ['nnUNet_preprocessed'] = '/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed'
    os.environ['nnUNet_results'] = '/opt/dlami/nvme/medseg3d_data/results'
    
    checkpoint = '/opt/dlami/nvme/medseg3d_data/results/Dataset003_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
    
    print("Testing ReservoirNNUNet creation...")
    
    # Load base model
    base_model, trainer = load_nnunet_model(checkpoint)
    
    # Test with dummy input
    x = torch.randn(1, 1, 64, 64, 64)
    out = base_model(x)
    print(f"\nBase model output: {out.shape}")
    
    # Wrap with ESN reservoir
    print("\nCreating ESN reservoir model...")
    esn_model = ReservoirNNUNet(
        base_model=base_model,
        skip_type='esn',
        skip_levels=[1, 2, 3, 4],
        reservoir_size=128,
        alpha=0.3,
    )
    
    # Test forward
    out_esn = esn_model(x)
    print(f"ESN model output: {out_esn.shape}")
    
    # Verify trainable params
    trainable = sum(p.numel() for p in esn_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in esn_model.parameters() if not p.requires_grad)
    print(f"\nTrainable params: {trainable:,}")
    print(f"Frozen params: {frozen:,}")
    
    print("\n✓ All tests passed!")