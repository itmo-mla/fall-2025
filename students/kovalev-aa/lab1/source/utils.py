import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from ModelsClasses import ClassifierLogisticReg

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import kagglehub
from kagglehub import KaggleDatasetAdapter


class ModelWrapper:
    def __init__(self, model, test_accuracy=None,
                 precision=None, recall=None, f1=None):
        self.model = model
        self.test_accuracy = test_accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

def load_base():
        # Set the path to the file you'd like to load
    file_path = "Employee.csv"

    # Load the latest version
    clean_data = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "tawfikelmetwally/employee-dataset",
      file_path
    ) 

    for col in ['Education', 'Gender', 'EverBenched']:
        types = clean_data[col].unique()
        mapping = {cat: idx for idx, cat in enumerate(types)}
        clean_data[col] = clean_data[col].map(mapping)

    clean_data = pd.get_dummies(clean_data, columns=['City'])
    clean_data['LeaveOrNot'] = clean_data['LeaveOrNot'].replace(0,-1)
    clean_data['JoiningYear'] = datetime.today().year -  clean_data['JoiningYear'] 

    # Разделение на train/test ===
    train_df = clean_data.sample(frac=0.7, random_state=42)
    test_df = clean_data.drop(train_df.index)

 

    train_x = train_df.drop('LeaveOrNot', axis=1).to_numpy(dtype=float)
    train_y = train_df['LeaveOrNot'].to_numpy(dtype=float)
    test_x = test_df.drop('LeaveOrNot', axis=1).to_numpy(dtype=float)
    test_y = test_df['LeaveOrNot'].to_numpy(dtype=float) 

    return train_x,train_y,test_x,test_y


# Создание и обучение логистических регрессий
def train_models(train_x,train_y,test_x,test_y):
    models_dict = {}

    # Rand + Rand
    model_rand_rand = ClassifierLogisticReg()
    model_rand_rand.train(train_x, train_y, test_x, test_y,
                          epoches=1000, batching_method='random', init_method='random')
    models_dict['RandRand'] = ModelWrapper(
        model_rand_rand,
        model_rand_rand.test_accuracy,
        model_rand_rand.test_precision,
        model_rand_rand.test_recall,
        model_rand_rand.test_f1
    )

    # Rand + Corr
    model_rand_correlation = ClassifierLogisticReg()
    model_rand_correlation.train(train_x, train_y, test_x, test_y,
                                 epoches=1000, batching_method='random', init_method='correlation')
    models_dict['RandCorr'] = ModelWrapper(
        model_rand_correlation,
        model_rand_correlation.test_accuracy,
        model_rand_correlation.test_precision,
        model_rand_correlation.test_recall,
        model_rand_correlation.test_f1
    )

    # Margin + Rand
    model_margin_rand = ClassifierLogisticReg()
    model_margin_rand.train(train_x, train_y, test_x, test_y,
                            epoches=1000, batching_method='margin', init_method='random')
    models_dict['MarginRand'] = ModelWrapper(
        model_margin_rand,
        model_margin_rand.test_accuracy,
        model_margin_rand.test_precision,
        model_margin_rand.test_recall,
        model_margin_rand.test_f1
    )

    # Margin + Corr
    model_margin_correlation = ClassifierLogisticReg()
    model_margin_correlation.train(train_x, train_y, test_x, test_y,
                                   epoches=1000, batching_method='margin', init_method='correlation')
    models_dict['MarginCorr'] = ModelWrapper(
        model_margin_correlation,
        model_margin_correlation.test_accuracy,
        model_margin_correlation.test_precision,
        model_margin_correlation.test_recall,
        model_margin_correlation.test_f1
    )


    lin_model = LinearRegression()
    lin_model.fit(train_x, train_y)
    lin_pred = np.sign(lin_model.predict(test_x))

    lin_accuracy = accuracy_score(test_y, lin_pred)
    lin_precision = precision_score(test_y, lin_pred, pos_label=1)
    lin_recall = recall_score(test_y, lin_pred, pos_label=1)
    lin_f1 = f1_score(test_y, lin_pred, pos_label=1)

    print(f"Linear Regression (MSE) Accuracy: {lin_accuracy}")

    models_dict['LinearRegMSE'] = ModelWrapper(
        lin_model,
        lin_accuracy,
        lin_precision,
        lin_recall,
        lin_f1
    )
    return models_dict

def choose_best_model(models_dict,
                      w_acc=0.3, w_prec=0.25,
                      w_rec=0.25, w_f1=0.25): 
    best_name, best_wrapper = max(
        models_dict.items(),
        key=lambda x: (
            w_acc * x[1].test_accuracy +
            w_prec * x[1].precision +
            w_rec * x[1].recall +
            w_f1 * x[1].f1
        )
    ) 
    return best_name, best_wrapper



#Анализ лучшей модели
def analyse_model(model:ClassifierLogisticReg):
    model.losses_plot()
    model.margin_plot()
    model.desc_metrics()
