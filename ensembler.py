from typing import Union, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from dataprocessor import DataProcessor
from modeler import Modeler

class Ensembler:
    def __init__(self, 
                 modelers : List[Modeler], 
                 data : DataProcessor, 
                 folder_path : str):
        '''
        

        Arguments: 
            modelers: 
            data: 
            folder_path: 
        '''
        self.n_modelers = len(modelers)
        self.modelers = modelers
        self.path = folder_path
        if self.path[-1] == '/':
            self.path[-1] == ''
        
        self.data = data
        self.validation_sets = []
        for fold in range(self.data.n_folds):
            _, _, _, yv = self.data.get_fold(fold)
            self.validation_sets.append(yv)
        
        self.weights = [1/self.n_modelers] * self.n_modelers # default
        pass

    def optimize_weights(self) -> list:
        '''
        Creates models on each of the n_folds previous created. Uses Linear Regression
        without intercept on each model to predict out-of-fold score. These coefficients
        are the ones used to form an ensembled prediction.

        Arguments:
            None
        '''
        Xs = []
        ys = []
        for fold in range(self.data.n_folds):
            _, _, _, y_val = self.data.get_fold(fold)
            y_val = np.array(y_val).reshape(-1)
            ys.append(y_val)
            
            Xs_per_model = []
            for modeler in self.modelers:
                model = modeler.model
                array = np.load(f'{self.path}/{model}_fold_{fold}.npy').reshape(-1, 1) #??
                Xs_per_model.append(array)
            
            Xs.append(np.hstack(Xs_per_model))
                
        X = np.vstack(Xs)
        y = np.hstack(ys)

        linear = LinearRegression(fit_intercept=False).fit(X, y)
        self.weights = linear.coef_
        return [*self.weights]
    
    def predict(self, X: pd.DataFrame) -> np.array:
        '''
        Predicts on X with last created model instances.
        
        Arguments:
            X: pandas dataframe used to generate predictions
        '''

        full_prediction = None
        for (coef, modeler) in zip(self.weights, self.modelers):
            pred = coef * np.array(modeler.predict(X))
            if full_prediction is not None:
                full_prediction += pred
            else:
                full_prediction = pred

        return full_prediction
    
    def generate_test_submission(self) -> pd.DataFrame:
        '''
        

        Arguments:
            None
        '''
        test = None
        for (coef, modeler) in zip(self.weights, self.modelers):
            model = modeler.model
            array = coef * np.load(f'{self.path}/{model}_test.npy').reshape(-1, 1) # ??
            if test is not None:
                test += array
            else: test = array

        submission = self.data.test_indices()
        submission[self.data.target] = test
        submission.to_csv(f'{self.path}/submission.csv', index=False)
        return submission

if __name__ == "__main__":

    train = pd.read_csv('./data/s4e5/train.csv').loc[:1000]
    test  = pd.read_csv('./data/s4e5/test.csv')

    data = DataProcessor(train_data=train, test_data=test, primary_column='id', target='FloodProbability')
    data.preprocess(n_clusters=0)
    data.set_cv()

    SAVE_PATH = './model_info'
    MODELS = [
        'hist_long',
        #'hist_wide',
        #'prox_long',
        'prox_wide',
        #'rndm_long',
        #'rndm_wide',
        #'linr'
    ]

    models = [Modeler(name, data, 'reg') for name in MODELS]
    for model in models:
        model.all_at_once(SAVE_PATH, n_trials=10)

    ensemble = Ensembler(models, data, SAVE_PATH)
    ensemble.optimize_weights()
    ensemble.generate_test_submission()

    print('Ensemble Weights')
    for model, weight in zip(models, ensemble.weights):
        print(f'{model.model} : {weight : .4f}')
