from typing import Union

import pandas as pd
import numpy as np

from dataprocessor import Preprocessing

from sklearn import ensemble as ens, linear_model as lnr
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna

ALL_MODELS = {
    'xgb': (xgb.XGBRegressor, xgb.XGBClassifier),
    'lgb': (lgb.LGBMRegressor, lgb.LGBMClassifier),
    'cat': (cat.CatBoostRegressor, cat.CatBoostClassifier),
    'rf': (ens.RandomForestRegressor,
           ens.RandomForestClassifier),
    'hgb': (ens.HistGradientBoostingRegressor,
            ens.HistGradientBoostingClassifier),
    'lin': (lnr.LinearRegression, lnr.LogisticRegression),
    'rdg': (lnr.Ridge, lnr.RidgeClassifier)
}

HYPERPARAMETER_SPACE = {
    'xgb': ['learning_rate', 'max_depth', 'n_estimators', 'subsample'],
    'lgb': ['learning_rate', 'max_depth', 'num_leaves', 'feature_fraction'],
    'cat': ['learning_rate', 'max_depth', 'iterations', 'colsample_bylevel'],
    'hgb': ['learning_rate', 'max_depth', 'max_iter', 'max_bins'],
    'rf' : ['n_estimators', 'max_depth', 'min_samples_split', 'max_features'],
    'lin': [],
    'rdg': ['alpha', ]
}

class Modeler:

    def __init__(self, model_type : str, params : dict = {}, classify : bool = False, 
                 columns : list[str] = []):

        assert type in ALL_MODELS, "Not an implemented model type: "\
            + f"Choose from {ALL_MODELS.keys()}."
        
        # We want to remember model constructor and parameters, not an instance.
        self.constructor = ALL_MODELS[model_type][classify]
        self.params = None
        if params:
            try:
                self.constructor(**params) # check params by building throwaway instance
            except:
                raise ValueError("Parameters don\'t match model type.")
            
            self.params = params.copy()

        self.instance = None
        self.hyperparams = HYPERPARAMETER_SPACE[model_type]
        self.columns = columns.copy()

        pass

    def fit_instance(self, X : pd.DataFrame, y = pd.Series):
        self.instance = self.constructor(self.params).fit(X, y)
        return self.instance

    def predict(self, X : pd.DataFrame):
        assert self.instance, "Model instance not yet created and fit."

        try:
            ret = self.instance.predict(X)
        except:
            raise RuntimeError("Model cannot predict on input provided.")
        
        return ret

    def leave_one_out(self):
        pass

    def tune(self, n_trials : int = 50):
        pass



if __name__ == "__main__":
    print(Modeler('rdg', classify=True).constructor)