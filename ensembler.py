from typing import Union, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from dataprocessor import DataProcessor
from modeler import Modeler

class Ensembler(Modeler):
    def __init__(self, modelers : List[Modeler], data : DataProcessor):

        self.n_modelers = len(modelers)
        self.modelers = modelers
        self.scores = []
        
        self.predictions = [
            modeler.cv_predictions() for modeler in self.modelers
        ]
        self.data = data
        self.validation_sets = []
        for fold in range(self.data.n_folds):
            _, _, _, yv = self.data.get_fold(fold)
            self.validation_sets.append(yv)

        self.optimal_weights = None
        pass

    def simple_average(self):

        preds = np.array(self.predictions[0])
        for pred in self.predictions[1:]:
            preds += np.array(pred)

        preds /= self.n_modelers
        pass

    def optimize_weights(self) -> list:
        '''
        Creates models on each of the n_folds previous created. Linear Regression
        without bias used on each model to predict out-of-fold score. These coefficients
        are the ones used to form an ensembled prediction.
        '''
        model_predictions = (
            np.array(prediction) for prediction in self.predictions
        )
        prediction_array = np.stack(model_predictions, axis=0)
        # n_models x folds x length
        prediction_array = np.transpose(prediction_array, (1, 2, 0))
        # folds x length x n_models

        X_scores = np.concatenate([prediction_array[fold,:,:] for fold in self.n_folds], axis=0)
        y_scores = np.stack(self.predictions, axis=0)

        linear = LinearRegression(bias=False).fit(X_scores, y_scores)
        self.optimal_weights = linear.coef_
        return [*self.optimal_weights]
    
    def optimal_test_submission(self):
        if self.optimal_weights is None:
            self.optimize_weights()

        test_pred = None
        for (coef, modeler) in zip(self.optimal_weights, self.modelers):
            if test_pred is None:
                test_pred = coef * \
                    modeler.predict(self.data.get_test(
                        dummies=modeler.dummies))
            else:
                test_pred += coef * \
                    modeler.predict(self.data.get_test(
                        dummies=modeler.dummies))
        
        submission = pd.DataFrame()
        submission[self.data.targets] = test_pred
        submission[self.data.primary_column] = self.data.test.index
