# QRC-EV: Quantum Reservoir Computing for EV Charging Load Forecasting

**Paper directory.** Contains LaTeX source for the manuscript.

## Files

```
paper/
├── main.tex                    # Root document
├── chapter1_introduction.tex   # Chapter 1: Introduction
├── chapter2_background.tex     # Chapter 2: Literature & Math Foundations
├── appendix_hyperparameters.tex # Appendix A: Hyperparameters
├── appendix_proofs.tex          # Appendix B: Proofs
├── refs.bib                     # Bibliography
├── build_paper.py              # Build script
└── README.md                   # This file
```

## Building Locally

Requires TeX Live. On Ubuntu/Debian:

```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-bibtex-extra
```

Then:

```bash
cd paper/
python build_paper.py           # Full build (pdflatex × 2 + bibtex)
python build_paper.py --clean   # Remove build artifacts
```

Or use `latexmk` for watch mode:

```bash
latexmk -pvc -view=pdf main.tex
```

## CI Build

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically
builds the paper on every push to `main` and feature branches.
The compiled PDF is available as a workflow artifact.

## Citation

```bibtex
@article{qrc_ev2026,
  title  = {Quantum Reservoir Computing for EV Charging Load Forecasting
            with Quantum Hidden Markov Models and Eligibility Traces},
  author = {Syah, Riza},
  year   = {2026},
}
```
