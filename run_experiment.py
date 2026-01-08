from __future__ import annotations

import argparse
from sketching.sketching import run_qs_cur_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Run a small QS + CUR sketching experiment.")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--m", type=int, default=100)
    p.add_argument("--l", type=int, default=12)
    p.add_argument("--d", type=int, default=30, help="Number of sampled columns")
    p.add_argument("--s", type=int, default=30, help="Number of sampled rows (for B)")
    p.add_argument("--basis", choices=["dct", "poly"], default="dct")
    p.add_argument("--sparsity", type=float, default=0.0)
    p.add_argument("--noise-scale", type=float, default=0.0)
    p.add_argument("--row-sampling", choices=["random", "real"], default="random")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    val, sec = run_qs_cur_experiment(
        n=args.n,
        m=args.m,
        l=args.l,
        d=args.d,
        s=args.s,
        basis=args.basis,
        sparsity=args.sparsity,
        noise_scale=args.noise_scale,
        row_sampling=args.row_sampling,
        seed=args.seed,
    )

    print(f"NMSE: {val:.6e}")
    print(f"Elapsed: {sec:.3f} s")


if __name__ == "__main__":
    main()
