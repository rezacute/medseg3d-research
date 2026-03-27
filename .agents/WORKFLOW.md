# Research Agent Workflow

Main workflow for AI-assisted quantum ML research in the medseg3d-research project.

## Overview

This workflow orchestrates three research threads:
1. **MedSeg3D** — Reservoir skip connections (NEGATIVE result — publishable)
2. **QRC-EV** — Quantum Reservoir Computing for EV charging (Phase 2 ongoing)
3. **Cross-cutting** — CUDA-Q + Braket integration, paper writing

## Project State

| Thread | Status | Next Action |
|--------|--------|-------------|
| MedSeg3D (negative result) | Results complete | Paper writing |
| QRC-EV Phase 1 | 95% complete | Resolve CUDA-Q API, stationarity testing |
| QRC-EV Phase 2 | In progress | Run A2-A6 benchmarks vs B1-B3 baselines |
| CUDA-Q + Braket | Integration exists | Test with Python 3.10/3.11 |

## Daily Workflow

### Morning Research Sync
1. Read today's notes: `memory/YYYY-MM-DD.md`
2. Check git status of medseg3d-research
3. Review experiment results in `results/`
4. Check for new literature on QRC / reservoir computing

### Experiment Pipeline

```
[Literature Review] → [Hypothesis] → [Code/Run] → [Analyze] → [Write]
                           ↑_______________|________________|
```

### Code → Run → Analyze Loop

**QRC Pipeline Test (5 min):**
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_qrc_pipeline.py
```

**CUDA-Q Backend Test (5 min):**
```bash
cd ~/.openclaw/workspace/medseg3d-research
python test_cudaq_simple.py  # Test basic functionality
python test_cudaq_backend.py # Test full backend
```

**5-Fold Training (long run):**
```bash
export nnUNet_raw=/opt/dlami/nvme/medseg3d_data/nnunet_raw
export nnUNet_preprocessed=/opt/dlami/nvme/medseg3d_data/nnunet_preprocessed
export nnUNet_results=/opt/dlami/nvme/medseg3d_data/results
tmux new -s fivefold
PYTHONUNBUFFERED=1 python train_reservoir_5fold.py > fivefold.log 2>&1
```

### Analysis Steps

1. **Run experiment** → results saved to `results/`
2. **Load results** → JSON with per-fold, per-organ metrics
3. **Statistical test** → Wilcoxon signed-rank (paired)
4. **Update literature review** → note findings
5. **Log to daily notes** → what worked, what didn't

## Decision Tree

```
New experiment idea?
├── Check literature first (avoid duplicate)
├── Does it align with MedSeg3D or QRC-EV?
│   ├── MedSeg3D → Is it already addressed by negative result?
│   └── QRC-EV → Which phase does it belong to?
└── Get Riza's input before significant compute time

Experiment failed?
├── Check env vars are set
├── Check GPU availability: nvidia-smi
├── Check Python path: echo $PYTHONPATH
└── Try minimal reproduction first

Paper writing needed?
├── Start with MedSeg3D (results are ready)
├── Structure: Abstract → Intro → Methods → Results → Discussion
└── Use paper-writing skill for details
```

## Skill Routing

| Task | Skill |
|------|-------|
| QRC experiments, benchmarks | `research-qrc` |
| MedSeg3D, nnU-Net, AMOS22 | `research-medseg` |
| Running code, git, environment | `code-execution` |
| Paper structure, figures, citations | `paper-writing` |

## File Locations

```
~/.openclaw/workspace/medseg3d-research/
├── results/                    # Experiment outputs
├── src/qrc_ev/               # QRC-EV code
├── src/models/               # MedSeg3D reservoir models
├── train_reservoir_5fold.py  # 5-fold training
├── train_paloalto.py         # Palo Alto EV training
└── docs/                     # Reports, roadmaps, figures
```

## Git Protocol

```bash
# Branch for agentic work
git checkout feature/agentic-research-workflow

# After significant changes
git add .
git commit -m "feat|fix|docs|test: description"
git push origin feature/agentic-research-workflow
```

## Memory Management

After each session:
1. Save significant findings to `memory/YYYY-MM-DD.md`
2. Update `MEMORY.md` with key decisions
3. Log any blocking issues to `HEARTBEAT.md`

## Emergency Contacts

If stuck on:
- **CUDA-Q API issues** → try Python 3.10/3.11 env, check NVIDIA docs
- **nnU-Net preprocessing** → check Dataset220_AMOS22 integrity
- **Paper writing** → flag for Riza to review structure
