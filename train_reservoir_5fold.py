#!/usr/bin/env python3
"""
5-Fold Cross-Validation for Reservoir Skip Evaluation.
Uses nnU-Net raw data for proper preprocessing + MONAI sliding window.

Usage:
    export nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
    export nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
    export nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
    tmux new -s fivefold
    PYTHONUNBUFFERED=1 python train_reservoir_5fold.py > fivefold.log 2>&1
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, '.')
from src.models.reservoir_nnunet import ReservoirNNUNet
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from monai.inferers import sliding_window_inference

# ============================================================
# Config
# ============================================================
RESULTS_DIR = os.environ.get('nnUNet_results', '/opt/dlami/nvme/medseg3d_data/results')
PREPROCESSED_DIR = os.environ.get('nnUNet_preprocessed', '/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed')
MODEL_FOLDER = f'{RESULTS_DIR}/Dataset003_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres'
SPLITS_FILE = f'{PREPROCESSED_DIR}/Dataset003_ACDC/splits_final.json'

PATCH_SIZE = (32, 224, 224)
BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-3
NUM_WORKERS = 0
DEVICE = 'cuda'

ORGAN_NAMES = [
    'spleen', 'rkidney', 'lkidney', 'gallbladder', 'liver',
    'stomach', 'pancreas', 'bladder', 'prostate',
    'R_adventitia', 'L_adventitia', 'heart', 'aorta', 'IVC', 'trachea'
]


# ============================================================
# Dice Loss
# ============================================================
class DiceCE(nn.Module):
    def __init__(self, num_classes=17, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes  # 17 = background(1) + 16 foreground organs
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, 17, D, H, W) — 17 = background(1) + 16 foreground organs
        # target: (B, D, H, W) with values 0-16
        pred_softmax = F.softmax(pred, dim=1)
        if target.dim() == 3:
            target = target.unsqueeze(0)  # (D, H, W) -> (1, D, H, W)
        # One-hot target: (B, 17, D, H, W)
        target_onehot = F.one_hot(target.long(), num_classes=17).permute(0, 4, 1, 2, 3).float()
        # Dice on all 17 classes (including background)
        dims = tuple(range(2, pred_softmax.dim()))
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        union = pred_softmax.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ============================================================
# Dataset (uses raw data, preprocessed on the fly by nnU-Net predictor)
# ============================================================
class RawDataset(torch.utils.data.Dataset):
    """Loads raw NIfTI data and applies nnU-Net preprocessing."""
    def __init__(self, case_ids, dataset_name='Dataset003_ACDC'):
        self.case_ids = case_ids
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            device=torch.device(DEVICE)
        )
        self.predictor.initialize_from_trained_model_folder(
            MODEL_FOLDER, use_folds=(0,), checkpoint_name='checkpoint_best.pth'
        )
        # Get properties needed for preprocessing
        self.plans_manager = self.predictor.plans_manager
        self.configuration_manager = self.predictor.configuration_manager
        
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        # nnU-Net predictor handles preprocessing internally
        # We return the preprocessed tensor for training
        img, seg = self.predictor.predict_from_raw_data(
            raw_nifti_folder=os.path.join(os.environ.get('nnUNet_raw', '/opt/dlami/nvme/medseg3d_data/nnunet_raw'), 'Dataset003_ACDC', case_id),
            output_filename=None,
            return_probabilities=False,
            predict_without_mirroring=False,
        )
        return img, seg


# ============================================================
# Evaluation with MONAI sliding window
# ============================================================
def evaluate(model, case_ids, fold, verbose=True):
    """Evaluate model on cases using MONAI sliding window on RESAMPLED data."""
    model.eval()
    
    # Load preprocessed data directly (avoids raw data I/O overhead)
    # The preprocessed data is at target spacing and can be used with sliding window
    data_folder = f'{PREPROCESSED_DIR}/Dataset003_ACDC/nnUNetPlans_3d_fullres'
    
    all_dice = {c: [] for c in range(1, 16)}
    case_results = []
    
    for case_id in case_ids:
        # Load preprocessed data
        import blosc2
        data = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}.b2nd'))
        seg = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}_seg.b2nd'))
        volume = np.ascontiguousarray(data)[0]
        label = np.ascontiguousarray(seg)[0].copy()
        label[label < 0] = 0

        # Resample to patch_size if needed (MONAI's sliding_window_inference handles it)
        # But we need to match the model's expected input size
        # The nnU-Net model expects input at target spacing [32, 224, 224]
        # Since our preprocessed data is already at target spacing, use it directly
        img = torch.from_numpy(volume).float().cuda()
        
        # For sliding window, we need 5D: (B, C, D, H, W)
        if img.ndim == 3:
            img = img.unsqueeze(0).unsqueeze(0)  # (D, H, W) -> (1, 1, D, H, W)
        
        with torch.no_grad():
            output = sliding_window_inference(
                img, roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.25,
                predictor=lambda x: model(x)[0] if isinstance(model(x), list) else model(x)
            )
        pred = output.argmax(dim=1).cpu().squeeze().numpy()
        
        organ_dice = {}
        for c in range(1, 16):
            inter = ((pred == c) & (label == c)).sum()
            union = (pred == c).sum() + (label == c).sum()
            dice = (2 * inter / (union + 1e-8)).item()
            organ_dice[ORGAN_NAMES[c-1]] = dice
            all_dice[c].append(dice)
        
        organ_dice['Mean'] = np.mean([organ_dice.get(ORGAN_NAMES[c-1], 0) for c in range(1, 16)])
        organ_dice['case_id'] = case_id
        case_results.append(organ_dice)
    
    # Per-organ mean
    results = {}
    for c in range(1, 16):
        if all_dice[c]:
            results[ORGAN_NAMES[c-1]] = float(np.mean(all_dice[c]))
    results['Mean'] = float(np.mean([results.get(ORGAN_NAMES[c-1], 0) for c in range(1, 16)]))
    
    if verbose:
        for org, dice in results.items():
            if org != 'Mean':
                print(f"    {org:15s}: {dice:.3f}")
        print(f"  Mean Dice: {results['Mean']:.3f}")
    
    return results, case_results


# ============================================================
# Training
# ============================================================
def train_one_fold(fold, epochs=EPOCHS):
    """Train one fold and return results dict."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    # Load splits
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_cases = splits[fold]['train']
    val_cases = splits[fold]['val']
    print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)} cases")
    print(f"Val cases: {val_cases}")

    # Load base model for this fold
    print(f"Loading fold {fold} base model...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        device=torch.device(DEVICE)
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_FOLDER, use_folds=(fold,), checkpoint_name='checkpoint_best.pth'
    )
    base_model = predictor.network.cuda().eval()
    
    # ---- Baseline evaluation ----
    print(f"Evaluating baseline (skip_type='none')...")
    model_base = ReservoirNNUNet(base_model, skip_type='none', skip_levels=[1, 2, 3, 4]).cuda().eval()
    baseline_results, baseline_cases = evaluate(model_base, val_cases, fold)
    print(f"  Baseline Mean: {baseline_results['Mean']:.3f}")

    # ---- Train FMQE reservoir ----
    print(f"\nTraining FMQE reservoir for {epochs} epochs...")
    model = ReservoirNNUNet(
        base_model=base_model,
        skip_type='fmqe',
        skip_levels=[1, 2, 3, 4],
        n_frequencies=4,
    ).cuda()
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FMQE trainable params: {trainable:,}")

    # Load training data from preprocessed (already at target spacing)
    data_folder = f'{PREPROCESSED_DIR}/Dataset003_ACDC/nnUNetPlans_3d_fullres'
    import blosc2
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, case_ids):
            self.case_ids = case_ids
        def __len__(self):
            return len(self.case_ids)
        def __getitem__(self, idx):
            case_id = self.case_ids[idx]
            data = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}.b2nd'))
            seg = blosc2.asarray(blosc2.open(f'{data_folder}/{case_id}_seg.b2nd'))
            volume = np.ascontiguousarray(data)[0]
            label = np.ascontiguousarray(seg)[0].copy()
            label[label < 0] = 0
            
            # Use sliding window inference for training (like nnU-Net does)
            # Random crop to PATCH_SIZE for training variety
            D, H, W = volume.shape
            # Random starting positions
            d = np.random.randint(0, max(1, D - PATCH_SIZE[0]))
            h = np.random.randint(0, max(1, H - PATCH_SIZE[1]))
            w = np.random.randint(0, max(1, W - PATCH_SIZE[2]))
            
            # Crop
            crop_vol = volume[d:d+PATCH_SIZE[0], h:h+PATCH_SIZE[1], w:w+PATCH_SIZE[2]]
            crop_lbl = label[d:d+PATCH_SIZE[0], h:h+PATCH_SIZE[1], w:w+PATCH_SIZE[2]]
            
            return torch.from_numpy(crop_vol).float(), torch.from_numpy(crop_lbl).long()
    
    train_ds = SimpleDataset(train_cases)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = DiceCE(num_classes=15)

    best_dice = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for volume, label in train_dl:
            volume = volume.unsqueeze(1).cuda()  # (B, D, H, W) -> (B, 1, D, H, W)
            label = label.cuda()

            optimizer.zero_grad()
            with autocast():
                output = model(volume)
                if isinstance(output, list):
                    output = output[0]
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)

        if epoch % 10 == 0 or epoch == epochs:
            val_results, _ = evaluate(model, val_cases, fold, verbose=False)
            mean_dice = val_results['Mean']
            print(f"Epoch {epoch:3d}/{epochs} - Loss: {avg_loss:.4f} | Val Mean: {mean_dice:.3f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch
                torch.save(model.state_dict(), f'{RESULTS_DIR}/reservoir_fmqe_fold{fold}_best.pth')
                print(f"  ✓ New best: {best_dice:.3f} at epoch {best_epoch}")

    elapsed = time.time() - start_time
    print(f"\nFold {fold} FMQE: Best={best_dice:.3f} at epoch {best_epoch} ({elapsed/60:.1f} min)")

    # Load best and get per-case results
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/reservoir_fmqe_fold{fold}_best.pth'))
    fmqe_results, fmqe_cases = evaluate(model, val_cases, fold)
    
    return {
        'fold': fold,
        'train_cases': train_cases,
        'val_cases': val_cases,
        'baseline_mean': baseline_results['Mean'],
        'fmqe_mean': fmqe_results['Mean'],
        'best_epoch': best_epoch,
        'elapsed_min': elapsed / 60,
        'baseline_results': baseline_results,
        'fmqe_results': fmqe_results,
        'baseline_cases': baseline_cases,
        'fmqe_cases': fmqe_cases,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("5-FOLD RESERVOIR EVALUATION (FMQE)")
    print("=" * 70)
    print(f"nnUNet_results: {RESULTS_DIR}")
    print(f"nnUNet_preprocessed: {PREPROCESSED_DIR}")
    print(f"nnUNet_raw: {os.environ.get('nnUNet_raw', 'not set')}")
    total_start = time.time()

    fold_results = []

    for fold in range(5):
        result = train_one_fold(fold, epochs=EPOCHS)
        fold_results.append(result)

        # Save intermediate
        safe_results = []
        for r in fold_results:
            safe_r = {k: v for k, v in r.items() if k not in ['baseline_cases', 'fmqe_cases']}
            safe_results.append(safe_r)
        with open(f'{RESULTS_DIR}/fivefold_intermediate.json', 'w') as f:
            json.dump(safe_results, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    print(f"\n\n{'='*70}")
    print("ALL 5 FOLDS COMPLETE")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"{'='*70}")

    # Per-organ summary
    from scipy.stats import wilcoxon
    
    all_baseline = {org: [] for org in ORGAN_NAMES}
    all_fmqe = {org: [] for org in ORGAN_NAMES}
    total_val = 0
    
    for r in fold_results:
        for c, org in enumerate(ORGAN_NAMES, 1):
            all_baseline[org].append(r['baseline_results'].get(org, 0))
            all_fmqe[org].append(r['fmqe_results'].get(org, 0))
        total_val += len(r['val_cases'])

    print(f"\n{'Organ':<15} {'Baseline':>12} {'FMQE':>12} {'Delta':>10} {'p-value':>10}")
    print("-" * 62)

    deltas = []
    for org in ORGAN_NAMES:
        b = np.array(all_baseline[org])
        f = np.array(all_fmqe[org])
        b_mean, b_std = b.mean(), b.std()
        f_mean, f_std = f.mean(), f.std()
        delta = f_mean - b_mean
        deltas.append(delta)
        
        try:
            _, p = wilcoxon(b, f)
            p_str = f"{p:.4f}"
        except:
            p_str = "N/A"
        
        print(f"{org:<15} {b_mean:.3f}±{b_std:.3f}  {f_mean:.3f}±{f_std:.3f}  {delta:>+9.3f}  {p_str:>10}")

    overall_b = np.mean([r['baseline_mean'] for r in fold_results])
    overall_f = np.mean([r['fmqe_mean'] for r in fold_results])
    print(f"\n{'OVERALL':<15} {overall_b:.3f}          {overall_f:.3f}          {overall_f-overall_b:>+.3f}")
    print(f"\nTotal val cases: {total_val}")

    # Per-fold summary
    print(f"\n--- PER-FOLD SUMMARY ---")
    print(f"{'Fold':>5} {'Train':>6} {'Val':>5} {'Baseline':>10} {'FMQE':>10} {'Delta':>8} {'BestEp':>7}")
    for r in fold_results:
        print(f"{r['fold']:>5} {len(r['train_cases']):>6} {len(r['val_cases']):>5} {r['baseline_mean']:>10.3f} {r['fmqe_mean']:>10.3f} {r['fmqe_mean']-r['baseline_mean']:>+8.3f} {r['best_epoch']:>7}")

    # Save final results
    safe_results = []
    for r in fold_results:
        safe_r = {k: v for k, v in r.items() if k not in ['baseline_cases', 'fmqe_cases']}
        safe_results.append(safe_r)

    final_results = {
        'fold_results': safe_results,
        'organ_summary': {
            org: {
                'baseline': {'mean': float(np.mean(all_baseline[org])), 'std': float(np.std(all_baseline[org]))},
                'fmqe': {'mean': float(np.mean(all_fmqe[org])), 'std': float(np.std(all_fmqe[org]))},
                'delta': float(np.mean(all_fmqe[org]) - np.mean(all_baseline[org])),
            }
            for org in ORGAN_NAMES
        },
        'total_val_cases': total_val,
        'total_time_hours': total_elapsed / 3600,
    }

    with open(f'{RESULTS_DIR}/fivefold_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/fivefold_results.json")
    print("DONE!")
