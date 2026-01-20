import numpy as np
import time
from typing import Dict
from linear_classifier import LinearClassifier
from utils import calculate_metrics


class ExperimentRunner:
    # Класс для проведения экспериментов с разными настройками обучения.
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_test: np.ndarray, y_test: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = X_train.shape[1]
    
    def run_multistart_experiment(self, n_starts: int = 5, 
                                 n_epochs: int = 50) -> Dict:
        # Эксперимент с мультистартом (многократный запуск из случайных начальных приближений).

        results = {
            'accuracies': [],
            'models': [],
            'training_times': []
        }
        
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ С МУЛЬТИСТАРТОМ ({n_starts} запусков)")
        print(f"{'='*60}")
        
        best_accuracy = 0
        best_model = None
        
        for i in range(n_starts):
            print(f"\nЗапуск {i+1}/{n_starts}:")
            
            # Создание нового классификатора с новым random_state
            classifier = LinearClassifier(
                n_features=self.n_features,
                learning_rate=0.01,
                reg_coef=0.001,
                momentum=0.9,
                random_state=42 + i
            )
            
            # Обучение
            start_time = time.time()
            history = classifier.fit_sgd(
                self.X_train, self.y_train,
                n_epochs=n_epochs,
                batch_size=1,
                adaptive_lr=False,
                margin_selection=False,
                verbose=False
            )
            training_time = time.time() - start_time

            y_pred = classifier.predict(self.X_test)
            metrics = calculate_metrics(self.y_test, y_pred)
            accuracy = metrics['accuracy']
            
            print(f"  Время обучения: {training_time:.2f} сек")
            print(f"  Точность на тесте: {accuracy:.4f}")

            results['accuracies'].append(accuracy)
            results['models'].append(classifier)
            results['training_times'].append(training_time)
            
            # Обновление лучшей модели
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = classifier
        
        # Статистика
        print(f"\n{'='*60}")
        print("СТАТИСТИКА МУЛЬТИСТАРТА:")
        print(f"Средняя точность: {np.mean(results['accuracies']):.4f}")
        print(f"Максимальная точность: {np.max(results['accuracies']):.4f}")
        print(f"Минимальная точность: {np.min(results['accuracies']):.4f}")
        print(f"Стандартное отклонение: {np.std(results['accuracies']):.4f}")
        print(f"{'='*60}")
        
        results['best_model'] = best_model
        results['best_accuracy'] = best_accuracy
        
        return results
    
    def compare_initialization_methods(self, n_epochs: int = 50) -> Dict:
        # Сравнение методов инициализации весов.

        print(f"\n{'='*60}")
        print("СРАВНЕНИЕ МЕТОДОВ ИНИЦИАЛИЗАЦИИ")
        print(f"{'='*60}")
        
        methods = [
            ('Случайная инициализация', 'random'),
            ('Корреляционная инициализация', 'correlation')
        ]
        
        comparison = {}
        
        for method_name, method_type in methods:
            print(f"\nМетод: {method_name}")
            
            classifier = LinearClassifier(
                n_features=self.n_features,
                learning_rate=0.01,
                reg_coef=0.001,
                momentum=0.9,
                random_state=42
            )
            
            if method_type == 'correlation':
                classifier.initialize_by_correlation(self.X_train, self.y_train)
                print("  Веса инициализированы через корреляцию")
            else:
                print("  Веса инициализированы случайно")

            start_time = time.time()
            history = classifier.fit_sgd(
                self.X_train, self.y_train,
                n_epochs=n_epochs,
                batch_size=1,
                adaptive_lr=False,
                margin_selection=False,
                verbose=False
            )
            training_time = time.time() - start_time
            
            y_pred = classifier.predict(self.X_test)
            metrics = calculate_metrics(self.y_test, y_pred)
            
            print(f"  Время обучения: {training_time:.2f} сек")
            print(f"  Точность: {metrics['accuracy']:.4f}")
            print(f"  F1-score: {metrics['f1_score']:.4f}")
            
            comparison[method_name] = {
                'model': classifier,
                'metrics': metrics,
                'training_time': training_time,
                'history': history
            }
        
        return comparison
    
    def compare_optimization_strategies(self) -> Dict:
        # Сравнение разных стратегий оптимизации.

        print(f"\n{'='*60}")
        print("СРАВНЕНИЕ СТРАТЕГИЙ ОПТИМИЗАЦИИ")
        print(f"{'='*60}")
        
        strategies = [
            ('Базовый SGD', {'adaptive_lr': False, 'margin_selection': False}),
            ('SGD с адаптивным LR', {'adaptive_lr': True, 'margin_selection': False}),
            ('SGD с выбором по отступу', {'adaptive_lr': False, 'margin_selection': True}),
            ('SGD с адаптивным LR и выбором по отступу', 
             {'adaptive_lr': True, 'margin_selection': True})
        ]
        
        strategies_results = {}
        
        for strategy_name, params in strategies:
            print(f"\nСтратегия: {strategy_name}")
            
            classifier = LinearClassifier(
                n_features=self.n_features,
                learning_rate=0.01,
                reg_coef=0.001,
                momentum=0.9,
                random_state=42
            )
            
            # Обучение с указанной стратегией
            start_time = time.time()
            history = classifier.fit_sgd(
                self.X_train, self.y_train,
                n_epochs=50,
                batch_size=1,
                adaptive_lr=params['adaptive_lr'],
                margin_selection=params['margin_selection'],
                verbose=False
            )
            training_time = time.time() - start_time

            y_pred = classifier.predict(self.X_test)
            metrics = calculate_metrics(self.y_test, y_pred)
            
            print(f"  Параметры: adaptive_lr={params['adaptive_lr']}, "
                  f"margin_selection={params['margin_selection']}")
            print(f"  Время обучения: {training_time:.2f} сек")
            print(f"  Точность: {metrics['accuracy']:.4f}")
            print(f"  F1-score: {metrics['f1_score']:.4f}")
            
            strategies_results[strategy_name] = {
                'model': classifier,
                'metrics': metrics,
                'training_time': training_time,
                'history': history,
                'params': params
            }
        
        return strategies_results
    
    def compare_with_sklearn(self, best_params: Dict = None) -> Dict:
        # Сравнение лучшей реализации с эталонной из sklearn.

        print(f"\n{'='*60}")
        print("СРАВНЕНИЕ С ЭТАЛОННОЙ РЕАЛИЗАЦИЕЙ (sklearn)")
        print(f"{'='*60}")
    
        if best_params is None:
            best_params = {
                'learning_rate': 0.01,
                'reg_coef': 0.001,
                'momentum': 0.9,
                'loss_type': 'logistic'
            }
    
        # 1. Наша реализация
        print("\n1. Наша реализация:")
        our_classifier = LinearClassifier(
            n_features=self.n_features,
            learning_rate=best_params['learning_rate'],
            reg_coef=best_params['reg_coef'],
            momentum=best_params['momentum'],
            random_state=42,
            loss_type=best_params.get('loss_type', 'logistic')
        )
    
        start_our = time.time()
        our_history = our_classifier.fit_sgd(
            self.X_train, self.y_train,
            n_epochs=50,
            batch_size=1,
            adaptive_lr=False,
            margin_selection=False,
            verbose=False
        )
        our_time = time.time() - start_our
    
        y_pred_our = our_classifier.predict(self.X_test)
        metrics_our = calculate_metrics(self.y_test, y_pred_our)
    
        print(f"  Время обучения: {our_time:.2f} сек")
        print(f"  Точность: {metrics_our['accuracy']:.4f}")
        print(f"  Precision: {metrics_our['precision']:.4f}")
        print(f"  Recall: {metrics_our['recall']:.4f}")
        print(f"  F1-score: {metrics_our['f1_score']:.4f}")
    
        # 2. Эталонная реализация (sklearn)
        print("\n2. Эталонная реализация (sklearn SGDClassifier):")

        if best_params.get('loss_type', 'logistic') == 'logistic':
            from sklearn.linear_model import LogisticRegression
            skl_classifier = LogisticRegression(
                penalty='l2',
                C=1/(best_params['reg_coef'] * len(self.X_train)) if best_params['reg_coef'] > 0 else 1.0,
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        else:
            from sklearn.linear_model import SGDClassifier
            skl_classifier = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=best_params['reg_coef'],
                learning_rate='constant',
                eta0=best_params['learning_rate'],
                max_iter=100,
                tol=1e-3,
                random_state=42,
                verbose=0
            )
    
        start_skl = time.time()
        skl_classifier.fit(self.X_train, self.y_train)
        skl_time = time.time() - start_skl
    
        y_pred_skl = skl_classifier.predict(self.X_test)
        metrics_skl = calculate_metrics(self.y_test, y_pred_skl)
    
        print(f"  Время обучения: {skl_time:.2f} сек")
        print(f"  Точность: {metrics_skl['accuracy']:.4f}")
        print(f"  Precision: {metrics_skl['precision']:.4f}")
        print(f"  Recall: {metrics_skl['recall']:.4f}")
        print(f"  F1-score: {metrics_skl['f1_score']:.4f}")
    
        # Сравнение
        print(f"\n{'='*60}")
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
        print(f"{'='*60}")
        print(f"{'Метрика':<15} {'Наша реализация':<20} {'Sklearn':<15} {'Разница':<10}")
        print(f"{'-'*60}")
    
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            our_val = metrics_our[metric]
            skl_val = metrics_skl[metric]
            diff = our_val - skl_val
        
            print(f"{metric:<15} {our_val:<20.4f} {skl_val:<15.4f} {diff:+.4f}")
    
        print(f"\nВремя обучения:")
        print(f"  Наша реализация: {our_time:.4f} сек")
        print(f"  Sklearn: {skl_time:.4f} сек")
        print(f"  Отношение: {our_time/skl_time:.2f}x")
    
        return {
            'our_model': our_classifier,
            'sklearn_model': skl_classifier,
            'our_metrics': metrics_our,
            'sklearn_metrics': metrics_skl,
            'our_time': our_time,
            'sklearn_time': skl_time
        }
    
    def run_comprehensive_experiment(self) -> Dict:
        # Проведение комплексного эксперимента со всеми настройками.
        
        all_results = {}
        
        # 1. Мультистарт
        all_results['multistart'] = self.run_multistart_experiment(
            n_starts=5, n_epochs=50
        )
        
        # 2. Сравнение инициализаций
        all_results['initialization'] = self.compare_initialization_methods(
            n_epochs=50
        )
        
        # 3. Сравнение стратегий оптимизации
        all_results['strategies'] = self.compare_optimization_strategies()
        
        # 4. Сравнение с sklearn
        all_results['sklearn_comparison'] = self.compare_with_sklearn()
        
        return all_results