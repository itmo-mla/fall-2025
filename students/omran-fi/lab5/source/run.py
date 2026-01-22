from __future__ import annotations

import argparse
from pathlib import Path

from source.data import load_dataset
from source.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab5: Logistic Regression (lecture formulas) - Newton & IRLS, y in {-1,+1}"
    )
    parser.add_argument("--dataset", choices=["breast_cancer", "iris_binary"], default="breast_cancer")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--q-tol", type=float, default=1e-10)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--step-size", type=float, default=1.0)

    parser.add_argument("--out-dir", type=str, default="outputs_lecture")

    args = parser.parse_args()

    ds = load_dataset(args.dataset, test_size=args.test_size, random_state=args.seed)
    out_dir = Path(args.out_dir)

    reports = run_experiment(
        ds=ds,
        out_dir=out_dir,
        max_iter=args.max_iter,
        tol=args.tol,
        ridge=args.ridge,
        step_size=args.step_size,
        q_tol=args.q_tol,
    )

    print(f"Dataset: {args.dataset} (y in {{-1,+1}}), outputs: {out_dir.resolve()}")
    print("-" * 70)
    for name, r in reports.items():
        print(name)
        print(f"  converged={r.converged}, n_iter={r.n_iter}")
        print(f"  acc_train={r.acc_train:.4f}, acc_test={r.acc_test:.4f}")
        print(f"  cm_test=\n{r.cm_test}")
        print("-" * 70)

    print("Saved: metrics.txt, equivalence.txt, history/*.csv, and plots *.png")


if __name__ == "__main__":
    main()
