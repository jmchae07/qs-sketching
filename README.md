# Sketching Experiments (QS / CUR)

This repository contains a cleaned-up, GitHub-ready version of the original exploratory notebook
`main_sketching.ipynb`.

It focuses on **matrix sketching / CUR-style reconstruction** under a simple synthetic **QS model**:

\[
X = QS + E,\quad Q\in\mathbb{R}^{n\times \ell},\; S\in\mathbb{R}^{\ell\times m}.
\]

Where:
- `S` is a side-information/basis matrix (DCT-like or polynomial),
- `Q` is a latent coefficient matrix,
- `E` is structured noise (including an “orthogonal/perpendicular” component).

## Project layout

```
.
├── sketching/
│   ├── __init__.py
│   └── sketching.py      # core utilities + experiment runner
├── run_experiment.py     # CLI entrypoint
└── main_sketching.ipynb  # (original notebook, kept for reference)
```

## Install

Recommended: create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # (macOS/Linux)
# .venv\Scripts\activate   # (Windows)

pip install -U pip
pip install -r requirements.txt
```

## Quick start

Run a small synthetic QS + CUR experiment:

```bash
python run_experiment.py --n 80 --m 100 --l 12 --d 30 --s 30 --basis dct --seed 42
```

You should see:

- **NMSE** (normalized mean-squared error):  \(\|X_{\hat{}}\!-\!X\|_F^2 / \|X\|_F^2\)
- Elapsed time in seconds

## Notes on “DCT” basis

The original notebook used a *DCT-like* cosine basis plus a small column-wise perturbation.
The cleaned code keeps this behavior for reproducibility. If you want a “pure” DCT basis,
remove the perturbation loop near the end of `get_S()`.

## Reproducibility

- The CLI takes `--seed`, which seeds both NumPy and Python’s `random`.
- All randomness is drawn from a NumPy `Generator` created from that seed.

## Citation / acknowledgement

If you use or extend this code, please cite your related paper/tech report (add the bib entry here).
