from typing import Union, List

import sklearn
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import numpy as np
import optuna
from dataprocessor import Preprocessing
from modeler import Modeler

class Ensembler:
    def __init__(self, models : List[Modeler], n_folds : int = 5):

        self.n_models = len(models)
        self.n_folds = n_folds
        
        pass

    def optimize_weights(self, n_trials : Union[int, None] = None):
        
        if not n_trials:
            n_trials = min(200, 3**self.n_models)

        pass