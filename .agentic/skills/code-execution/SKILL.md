# SKILL.md — Code Execution

Execute research code, run experiments, manage environments for the medseg3d-research project.

## Working Directory

```
~/.openclaw/workspace/medseg3d-research/
```

## Environment Setup

### Conda Environment
```bash
conda create -n medseg python=3.10 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate medseg
pip install nnunetv2 monai blosc2 gdown pennylane
```

### Required Env Vars
```bash
export nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
export nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
export nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
```

### nnU-Net Data Preparation
```bash
nnUNetv2_plan_and_preprocess -d Dataset220_AMOS22 --verify_dataset_integrity
```

## Running Experiments

### QRC Pipeline (PennyLane — works)
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_qrc_pipeline.py
```

### CUDA-Q Backend Test
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_cudaq_backend.py
python test_cudaq_simple.py
```

### 5-Fold MedSeg3D Training
```bash
cd ~/.openclaw/workspace/medseg3d-research
tmux new -s fivefold
PYTHONUNBUFFERED=1 python train_reservoir_5fold.py > fivefold.log 2>&1
```

### Palo Alto EV Charging Training
```bash
python train_paloalto.py
```

## Git Workflow

```bash
cd ~/.openclaw/workspace/medseg3d-research

# Check status
git status

# Commit changes
git add .
git commit -m "feat: description"

# Push to branch
git push -u origin feature/agentic-research-workflow
```

## Python Path

The repo root is added to `sys.path` in scripts. For ad-hoc execution:
```bash
cd ~/.openclaw/workspace/medseg3d-research
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## GPU Checking

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
