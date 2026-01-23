import argparse

from .utils import check_for_dataset
from .experiments import run_tests

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./results', help='Path to directory where to store test results.')
    solver_names = ", ".join(["full", "covariance_eigh"])
    parser.add_argument(
        "--solver",
        nargs="*",
        default=None,
        help=(
            "Solver(s) to test. Use names or numeric ids. "
            f"Available: {solver_names} (0=full, 1=covariance_eigh). If omitted, all solvers are used."
        ),
    )

    args = parser.parse_args(argv)
    available_solvers = ["full", "covariance_eigh"]
    solvers = None
    if args.solver is not None:
        solvers = []
        for value in args.solver:
            if value.isdigit():
                idx = int(value)
                if idx < 0 or idx >= len(available_solvers):
                    raise ValueError(f"Solver id {idx} out of range.")
                solver_name = available_solvers[idx]
            else:
                solver_name = value
                if solver_name not in available_solvers:
                    raise ValueError(f"Unknown solver '{solver_name}'.")
            if solver_name not in solvers:
                solvers.append(solver_name)

    csv_path = check_for_dataset()
    print("Using dataset:", csv_path)

    run_tests(input_dir=csv_path, results_dir=args.output_dir, solvers=solvers)

if __name__ == "__main__":
    SystemExit(main())
