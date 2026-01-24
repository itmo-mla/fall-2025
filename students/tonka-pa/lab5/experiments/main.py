import argparse

from src.utils import check_for_alzheimers_dataset
from .experiments import run_tests

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./results', help='Path to directory where to store test results.')
    solver_names = ", ".join(["newton", "irls"])
    parser.add_argument(
        "--solvers",
        nargs="*",
        default=None,
        help=(
            "Solver(s) to test. "
            f"Available: {solver_names}. If omitted, all solvers are used."
        ),
    )

    args = parser.parse_args(argv)

    csv_path = check_for_alzheimers_dataset()
    print("Using dataset:", csv_path)

    run_tests(input_dir=csv_path, results_dir=args.output_dir, solvers=args.solvers)

if __name__ == "__main__":
    SystemExit(main())
