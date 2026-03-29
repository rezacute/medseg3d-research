#!/usr/bin/env python3
"""Build script for QRC-EV paper.

Usage:
    python build_paper.py          # Full build (pdflatex + bibtex + pdflatex x2)
    python build_paper.py --watch  # Watch mode with latexmk
    python build_paper.py --clean  # Remove build artifacts
"""
import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

PAPER_DIR = Path(__file__).parent
MAIN_TEX = PAPER_DIR / "main.tex"
BUILD_DIR = PAPER_DIR / "_build"
TEXLIVE_IMAGE = "registry.access.rhat/redhat/texlive"  # Fallback if local TeX missing


def run(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"  {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    if result.stdout:
        print(result.stdout)
    return result


def pdflatex_main(tex_file, out_dir):
    """Run pdflatex once."""
    run([
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_file),
    ])


def build_full():
    """Run pdflatex + bibtex + pdflatex x2."""
    BUILD_DIR.mkdir(exist_ok=True)

    print("\n=== Pass 1: pdflatex ===")
    pdflatex_main(MAIN_TEX, BUILD_DIR)

    print("\n=== bibtex ===")
    run(["bibtex", str(BUILD_DIR / "main")], check=False)

    print("\n=== Pass 2: pdflatex ===")
    pdflatex_main(MAIN_TEX, BUILD_DIR)

    print("\n=== Pass 3: pdflatex ===")
    pdflatex_main(MAIN_TEX, BUILD_DIR)

    pdf = BUILD_DIR / "main.pdf"
    if pdf.exists():
        size_kb = pdf.stat().st_size // 1024
        print(f"\n✅  Built {pdf}  ({size_kb} KB)")
    else:
        print("\n❌  PDF not found — check .log for errors", file=sys.stderr)
        sys.exit(1)


def clean():
    """Remove build artifacts."""
    patterns = ["*.aux", "*.bbl", "*.blg", "*.log", "*.out", "*.toc", "*.lof", "*.lot"]
    for p in patterns:
        for f in BUILD_DIR.glob(p):
            f.unlink()
    print("Cleaned build directory.")


def watch():
    """Watch mode using latexmk."""
    try:
        run(["latexmk", "-pvc", "-view=pdf", str(MAIN_TEX)], cwd=PAPER_DIR)
    except FileNotFoundError:
        print("latexmk not found. Install TeX Live or run: pip install latexmk", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Watch mode")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    args = parser.parse_args()

    if args.watch:
        watch()
    elif args.clean:
        clean()
    else:
        build_full()
