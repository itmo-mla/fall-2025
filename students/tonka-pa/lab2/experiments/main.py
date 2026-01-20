import argparse

from src.utils import utils
from src.knn_model import Kernel
from .experiments import run_tests

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./results', help='Path to directory where to store test results.')
    kernel_names = ", ".join([k.name.lower() for k in Kernel])
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        help=(
            "Kernel(s) to test. Use names or numeric ids. "
            f"Available: {kernel_names}. If omitted, all kernels are used."
        ),
    )

    args = parser.parse_args(argv)

    csv_path = utils.check_for_alzheimers_dataset()
    print("Using dataset:", csv_path)

    run_tests(input_dir=csv_path, results_dir=args.output_dir, kernels=args.kernels)

if __name__ == "__main__":
    SystemExit(main())
