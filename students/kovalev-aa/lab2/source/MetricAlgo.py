import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import plot_risk_summary

class MetricClasifier: 
    def __init__(self,x_train,y_train):  
        self.nearest_edxes = None
        self.k = 0
        self.x_train = x_train
        self.y_train = y_train
        
        pass
 

    def evklid_distance_matrix(self,x1,x2):
        X1_sq = np.sum(x1**2,axis=1,keepdims=True)
        X2_sq = np.sum(x2**2,axis=1,keepdims=True)
        evclid_matrix_sq = X1_sq + X2_sq.T - 2 * (x1 @ x2.T)
        evclid_matrix_sq[evclid_matrix_sq<0] = 0
        return np.sqrt(evclid_matrix_sq)
    

    def gaus_k(self,u):
        return 1/(np.sqrt(2*np.pi)) * np.exp(-u**2/2)
    
    def parzen_window(self,x_test,x_train,y,k):
        # nearest_edx= self.nearest_edxes[:,:k]
        classes = np.unique(y)
        diff_matrix = self.evklid_distance_matrix(x_test,x_train)
        diff_matrix[diff_matrix==0] = np.inf
        weights = np.zeros((x_test.shape[0], len(classes)))
        for i,c in enumerate(classes):
            mask = (y==c)
            h = diff_matrix[np.arange(diff_matrix.shape[0]), self.nearest_edxes[:, k]]
            weights[:,i] = np.sum(self.gaus_k(diff_matrix[:,mask]/(h[:,None] + 3e-15)),axis=1)
        return weights
        


    def standart_select(self):
        x_train = self.x_train
        y_train = self.y_train

        n = x_train.shape[0]
        selected_mask = np.ones(n, dtype=bool)   

        
        prev_errors = np.array(self.loo_loss(x_train, y_train, self.k))
        prev_error_mean = np.mean(prev_errors)

        
        for i in range(n):
            if not selected_mask[i]:
                continue

            mask = selected_mask.copy()
            mask[i] = False   

            x_sub = x_train[mask]
            y_sub = y_train[mask]

             
            new_errors = np.array(self.loo_loss(x_sub, y_sub, self.k))
            new_error_mean = np.mean(new_errors)

            
            if new_error_mean > prev_error_mean:
                selected_mask[i] = True
            else:
                selected_mask[i] = False
                prev_error_mean = new_error_mean   

        self.x_train = x_train[selected_mask]
        self.y_train = y_train[selected_mask] 
 
    
    def train_k(self  , is_plot=True): 
        n = self.x_train.shape[0]
        k_array = np.arange(1, int(np.sqrt(n)))
        k_errors = np.zeros((k_array.shape[0], n))  
        for k_idx, k in enumerate(k_array): 
            k_errors[k_idx] = self.loo_loss(self.x_train,self.y_train,k)
 
        mean_errors = np.mean(k_errors, axis=1)
        self.k = k_array[np.argmin(mean_errors)]  

        if is_plot:
            plot_risk_summary(k_array, k_errors)   

        print(f"Лучшее k = {self.k}")
        return k_errors
    
    def loo_loss(self,x_train,y_train,k):
        n = x_train.shape[0]
        selected_mask = np.ones(n, dtype=bool)  
 
         
        errors = []
        for i in range(n):
            mask = selected_mask.copy()
            mask[i] = False
            x_sub = x_train[mask]
            y_sub = y_train[mask]
            x_test = x_train[i:i+1]
            y_true = y_train[i]

            diff_matrix = self.evklid_distance_matrix(x_test, x_sub)
            self.nearest_edxes = np.argsort(diff_matrix, axis=1)
            weights = self.parzen_window(x_test, x_sub, y_sub, k)
            p_true = weights[0, y_true] / np.sum(weights[0])
            error = -np.log(p_true + 1e-15)
            errors.append(error)

        return errors 
 

    def predict(self, x_test): 
        if self.x_train is None or self.y_train is None:
            raise ValueError("Модель не обучена. Сначала вызовите train_k.")
            
        diff_matrix = self.evklid_distance_matrix(x_test, self.x_train)
        self.nearest_edxes = np.argsort(diff_matrix, axis=1)
        weights = self.parzen_window(x_test, self.x_train, self.y_train, self.k)
         
        predictions = np.argmax(weights, axis=1)
        return predictions


    def metrics(self,y_true, y_pred, average='macro'): 
        metrics_dict = {}
        
         
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        metrics_dict['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics_dict['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics_dict['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
         
        metrics_dict['confusion_matrix'] = confusion_matrix(y_true, y_pred)
         
        return metrics_dict