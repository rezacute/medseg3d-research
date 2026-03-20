#!/usr/bin/env python3
"""
train_reservoir_v2.py - Train Reservoir Skip Connections on nnU-Net
Proper implementation with MONAI sliding window evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import blosc2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.reservoir_nnunet import ReservoirNNUNet
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from monai.inferers import sliding_window_inference

# Paths
DATA_DIR = '/opt/dlami/nvme/medseg3d_data'
PREPROCESSED_DIR = f'{DATA_DIR}/nnunet_preprocessed'
RESULTS_DIR = f'{DATA_DIR}/results'
DATA_FOLDER = f'{PREPROCESSED_DIR}/Dataset003_ACDC/nnUNetPlans_3d_fullres'

PATCH_SIZE = (32, 224, 224)
BATCH_SIZE = 2
LR = 1e-3
EPOCHS = 100


class Blosc2Dataset(torch.utils.data.Dataset):
    """Dataset that loads preprocessed blosc2 files."""
    
    def __init__(self, case_ids, data_folder=DATA_FOLDER):
        self.data_folder = Path(data_folder)
        self.case_ids = case_ids
        
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        
        # Load data and segmentation
        data = blosc2.asarray(blosc2.open(self.data_folder / f'{case_id}.b2nd'))
        seg = blosc2.asarray(blosc2.open(self.data_folder / f'{case_id}_seg.b2nd'))
        
        data = np.ascontiguousarray(data)
        seg = np.ascontiguousarray(seg)
        
        D, H, W = data.shape[1:]
        pD, pH, pW = PATCH_SIZE
        
        # Random crop
        d = np.random.randint(0, max(1, D - pD + 1))
        h = np.random.randint(0, max(1, H - pH + 1))
        w = np.random.randint(0, max(1, W - pW + 1))
        
        img = data[0, d:d+pD, h:h+pH, w:w+pW].astype(np.float32)
        lbl = seg[0, d:d+pD, h:h+pH, w:w+pW].astype(np.int64)
        lbl[lbl < 0] = 0
        
        # Add channel dim
        img = torch.from_numpy(img).unsqueeze(0)
        lbl = torch.from_numpy(lbl)
        
        return img, lbl


def dice_ce_loss(pred, target, num_classes=17):
    """Dice + CE Loss. Use out[0] only (full resolution)."""
    # pred: (B, C, D, H, W), target: (B, D, H, W)
    pred = F.softmax(pred, dim=1)
    
    # One-hot encode target: (B, D, H, W) -> (B, C, D, H, W)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # (B, D, H, W, C)
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
    
    dims = tuple(range(2, pred.dim()))
    intersection = (pred * target_one_hot).sum(dims)
    cardinality = pred.sum(dims) + target_one_hot.sum(dims)
    dice = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
    dice_loss = 1 - dice[:, 1:].mean()  # Exclude background
    
    ce_loss = F.cross_entropy(pred, target, ignore_index=0)
    
    return 0.5 * dice_loss + 0.5 * ce_loss


def evaluate(model, val_cases, data_folder=DATA_FOLDER):
    """Evaluate using MONAI sliding window inference."""
    model.eval()
    
    all_dice = {c: [] for c in range(1, 16)}
    
    organ_names = ['spleen', 'rkidney', 'lkidney', 'gallbladder', 'liver', 
                   'stomach', 'pancreas', 'bladder', 'prostate', 'R_adventitia',
                   'L_adventitia', 'heart', 'aorta', 'IVC', 'trachea']
    
    def pred_fn(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out
    
    with torch.no_grad():
        for case_id in val_cases:
            # Load data
            data = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}.b2nd'))
            seg = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}_seg.b2nd'))
            
            volume = torch.from_numpy(np.ascontiguousarray(data)[0]).float()
            label = np.ascontiguousarray(seg)[0].copy()
            label[label < 0] = 0
            
            # Sliding window inference
            img = volume.unsqueeze(0).unsqueeze(0).cuda()
            output = sliding_window_inference(
                img, roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.25, predictor=pred_fn
            )
            pred = output.argmax(dim=1).cpu().squeeze().numpy()
            
            # Per-organ Dice
            for c in range(1, 16):
                inter = ((pred == c) & (label == c)).sum()
                union = (pred == c).sum() + (label == c).sum()
                dice = 2 * inter / (union + 1e-8)
                all_dice[c].append(dice.item())
    
    # Print results
    print("\n  Per-organ Dice:")
    mean_all = []
    for c in range(1, 16):
        m = np.mean(all_dice[c]) if all_dice[c] else 0
        mean_all.append(m)
        if m > 0.01:
            print(f"    {organ_names[c-1]:15s}: {m:.3f}")
    
    mean_dice = np.mean(mean_all)
    print(f"  Mean Dice: {mean_dice:.3f}")
    
    return mean_dice


def train_reservoir(skip_type, train_cases, val_cases, epochs=EPOCHS, lr=LR):
    """Train reservoir model."""
    
    # Load base model using nnUNetPredictor (no torch.compile)
    print("\n" + "="*60)
    print(f"Loading base model for {skip_type}...")
    print("="*60)
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device('cuda'),
    )
    predictor.initialize_from_trained_model_folder(
        f'{RESULTS_DIR}/Dataset003_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres',
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    base_model = predictor.network
    base_model = base_model.cuda()
    base_model.eval()
    
    # Create reservoir wrapper
    model = ReservoirNNUNet(
        base_model=base_model,
        skip_type=skip_type,
        skip_levels=[1, 2, 3, 4],
        n_frequencies=4,  # n_freq=4, bottleneck=16 gives ~110K trainable
    )
    model = model.cuda()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable: {trainable:,}, Frozen: {frozen:,}")
    
    # Datasets
    train_ds = Blosc2Dataset(train_cases)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5
    )
    
    # AMP scaler
    scaler = torch.amp.GradScaler()
    
    best_dice = 0
    best_epoch = 0
    
    print(f"\nTraining {skip_type} for {epochs} epochs...")
    print(f"Train: {len(train_ds)} cases, Val: {len(val_cases)} cases")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for img, lbl in train_dl:
            img = img.cuda()
            lbl = lbl.cuda()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                out = model(img)
                # Use full resolution output only
                if isinstance(out, (list, tuple)):
                    out = out[0]
                loss = dice_ce_loss(out, lbl)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= max(n_batches, 1)
        
        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            val_dice = evaluate(model, val_cases)
            
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                # Save best
                torch.save(model.state_dict(), f'{RESULTS_DIR}/reservoir_{skip_type}_best.pth')
        
        if (epoch + 1) % 20 == 0:
            print(f"\n  [Checkpoint saved at epoch {epoch+1}]")
    
    print(f"\n✓ {skip_type} training complete! Best Dice: {best_dice:.3f} at epoch {best_epoch}")
    
    return model, best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_type', type=str, default='esn', choices=['esn', 'fmqe', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()
    
    # Load splits
    with open(f'{DATA_DIR}/nnunet_preprocessed/Dataset003_ACDC/splits_final.json') as f:
        splits = json.load(f)
    
    train_cases = splits[0]['train']
    val_cases = splits[0]['val']
    
    print(f"\n{'='*60}")
    print(f"Experiment: Reservoir Skip ({args.skip_type})")
    print(f"{'='*60}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    
    # Train
    model, best_dice = train_reservoir(
        args.skip_type,
        train_cases,
        val_cases,
        epochs=args.epochs,
        lr=args.lr,
    )
    
    # Save final
    torch.save(model.state_dict(), f'{RESULTS_DIR}/reservoir_{args.skip_type}_final.pth')
    print(f"\nModel saved: {RESULTS_DIR}/reservoir_{args.skip_type}_final.pth")


if __name__ == '__main__':
    os.environ['nnUNet_results'] = RESULTS_DIR
    os.environ['nnUNet_preprocessed'] = PREPROCESSED_DIR
    main()