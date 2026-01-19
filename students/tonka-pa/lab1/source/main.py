import json
import argparse
import random
from pathlib import Path
from time import perf_counter

import numpy as np
from pandas import DataFrame

from utils import benchmark, data_prep
from logistic_regression import LogRegNumpy, LogisticRegressionSK, SGDClassifierSK

# --------------------------------------

SEED = 18092025
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run benchmark on the dataset.')
    parser.add_argument('downloadpath', type=str, help='The path where to download and save dataset.')
    parser.add_argument('paramspath', type=str, help='The path to the experimental model parameters.')
    parser.add_argument('-d', '--delim', type=str,
                        help='The delimiter to use to parse csv file.', default=',')
    parser.add_argument('-l', '--download', action='store_true',
                        help='Download dataset or not.')
    parser.add_argument('-s', "--save_output", default="",
                        help="Where to save output images and table during benchmarking. Should be directory")
    parser.add_argument('-v', "--verbose", action='store_true',
                        help='Print logs during models trainings.')
    args = parser.parse_args()

    # Загрузка датасета
    if args.download:
        file_path = data_prep.download_dataset(args.downloadpath)
    else:
        # Костыль
        print('Skip downloading dataset, assuming it already exists.')
        file_path = args.downloadpath + '/data.csv'

    # Проверка и создание при необходимости директории для картинок и таблиц
    output_path = args.save_output
    if output_path:
        output_path = Path(output_path)
        if not output_path.exists():
            print("Creating directory for output files.")
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path.as_posix() + "/"

    df = data_prep.read_data(file_path, delim=args.delim)
    X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, classes = data_prep.preprocess_data(df)

    params_path = Path(args.paramspath)
    with params_path.open(mode='r', encoding='utf-8') as file:
        model_test_params = json.load(file)

    results_on_models = dict()
    
    bench_begin = perf_counter()
    for model_class in (LogRegNumpy, LogisticRegressionSK, SGDClassifierSK):
        for test_suffix, model_params in model_test_params[model_class.__name__].items():
            
            model_dir = Path(output_path + model_class.__name__)
            model_dir.mkdir(parents=True, exist_ok=True)
            output_path_ext = model_dir.as_posix() + '/'
            
            model_params['verbose'] = args.verbose

            iter_begin = perf_counter()
            _, results = benchmark.benchmark_classifier(
                model_class,
                X_train_scaled, y_train_enc,
                X_test_scaled, y_test_enc,
                cv_folds=5,
                model_args=model_params,
                class_names=classes,
                plot_loss=True, display_loss=False,
                plot_roc=True, display_roc=False,
                plot_confusions=True, display_confusions=False,
                plot_margins=True, display_margins=False,
                output_path=output_path_ext,
                test_name_suffix=test_suffix
            )
            iter_duartion = perf_counter() - iter_begin
            print(f"Iteration duration: {iter_duartion:.2f}s")
            results_on_models[model_class.__name__ + '_' + test_suffix] = results
    bench_duration = perf_counter() - bench_begin
    print(f"Benchmark overall duartion: {bench_duration:.2f}s")

    comp_dict = {k: v['cv_mean'].values.ravel() for k, v in results_on_models.items()}
    cols = results['cv_mean'].columns
    comparison_table = DataFrame.from_dict(comp_dict, orient='index', columns=cols)
    benchmark._print_box_table(
        comparison_table,
        title = 'Overall models comparison',
        digits=4,
        index_name="Model",
        ascii_borders=True
    )

if __name__ == "__main__":
    main()