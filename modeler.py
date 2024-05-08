from typing import Union

import pandas as pd
import numpy as np

from dataprocessor import Preprocessing

from sklearn import ensemble as ens, \
    linear_model as lnr, svm
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna

from sklearn.metrics import roc_auc_score, mean_squared_error

ALL_MODELS = {
    'xgb': (xgb.XGBRegressor, xgb.XGBClassifier),
    'lgb': (lgb.LGBMRegressor, lgb.LGBMClassifier),
    'cat': (cat.CatBoostRegressor, cat.CatBoostClassifier),
    'rf': (ens.RandomForestRegressor,
           ens.RandomForestClassifier),
    'hgb': (ens.HistGradientBoostingRegressor,
            ens.HistGradientBoostingClassifier),
    'lin': (lnr.LinearRegression, lnr.LogisticRegression),
    'rdg': (lnr.Ridge, lnr.RidgeClassifier),
    'svm': (svm.SVR, svm.SVC)
}

HYPERPARAMETER_SPACE = {

    # model : [(param name, min of range, max of range, float?, log?)]

    'xgb': [('learning_rate', 1e-2, 1e-1, True, True), 
            ('max_depth', 4, 10, False, True), 
            ('n_estimators', 1e2, 1e3, False, True), 
            ('subsample', 2e-1, 1, True, True),
            ('lambda', 1, 1e2, True, True),
            ('alpha', 1e-8, 1e-2, True, True)],
    'lgb': [('learning_rate', 1e-2, 1e-1, True, True),
            ('max_depth', 4, 10, False, False), 
            ('num_iterations', 1e2, 1e3, False, True),
            ('num_leaves', 1e1, 1e2, False, True)],
    'cat': [('depth', 4, 10, False, False), 
            ('n_estimators', 1e2, 1e3, False, True)],
    # do not one-hot encode ^
    'hgb': [('learning_rate', 1e-2, 1e-1, True, True), 
            ('max_depth', 4, 10, False, True), 
            ('max_iter', 2e2, 2e3, False, True), 
            ('max_bins', 63, 255, False, True)],
    'rf' : [('n_estimators', 2e2, 2e3, False, True),
            ('max_depth', 3, 8, False, True),
            ('min_samples_split', 2, 10, False, False), 
            ('max_samples', 1e-1, 1, True, True)],
    'lin': [],
    'rdg': [('alpha', 1e-1, 1e1, True, True)],
    'svm': [('C', 1e-1, 1e1, True, True),
            ('gamma', 1e-2, 1, True, True)]
}

class Modeler:

    # Modeler represents a single model
    ### The goal is to find optimal features 
    ### and fix hyperparameters based on CV Scheme

    def __init__(self, model_type : str, params : dict = {}, classify : bool = False, 
                 columns : list[str] = [], criterion : Union[function, None] = None, 
                 goal : Union[str, None] = None):

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

        if criterion:
            assert goal, "With custom criterion, must state goal : \"maximize\" or \"minimize\""
            assert goal in ["maximize", "minimize"], \
                "\"goal\" must be \"maximize\" or \"minimize\""
            self.criterion = criterion
            self.maximize = goal == "maximize"
        elif classify:
            self.criterion = roc_auc_score
            self.maximize = True
        else:
            self.criterion = mean_squared_error
            self.maximize = False

        self.hyperparams = HYPERPARAMETER_SPACE[model_type]
        self.columns = columns.copy()

        self.dropped_columns = None
        self.instance = None

        pass

    def fit_instance(self, X : pd.DataFrame, y : pd.Series):
        self.instance = self.constructor(**self.params).fit(X, y)
        return self.instance

    def predict(self, X : pd.DataFrame):
        assert self.instance, "Model instance not yet created and fit."

        try:
            ret = self.instance.predict(X)
        except:
            raise RuntimeError("Model cannot predict on input provided.")
        
        return ret
    
    def score(self, X : pd.DataFrame, y : pd.Series):

        predictions = self.predict(X)
        
        try:
            score = self.criterion(y, predictions)
        except:
            raise RuntimeError("Model cannot compare given y with model predictions.")
        
        return score

    def leave_one_out(self, X, y, ids, col : str = None):
        n_splits = 5
        error = None
        gkf = GroupKFold(n_splits=n_splits)
        for _, (train_index, valid_index) in enumerate(gkf.split(X, y, ids)):

            if col is None:
                train_X = X.loc[train_index]
                valid_X = X.loc[valid_index]
            else:
                train_X = X.loc[train_index].drop(columns=[col])
                valid_X = X.loc[valid_index].drop(columns=[col])

            train_y = y.loc[train_index]
            valid_y = y.loc[valid_index]

            model = self.fit_instance(train_X, train_y)

            # Determine error
            score = model.score(valid_X, valid_y)
            if error is not None: # "if error" would be false if = 0
                error += score
            else:
                error = score
        
        average_error = error / 5
        return average_error
    
    def iterative_feature_selection(self, X, y, ids, threshold = 1e-4) -> list:
        
        baseline = self.leave_one_out(X, y, ids)
        full_cycle = False
        drop_cols = []
        cycle_columns = list(X.columns)
        while not full_cycle:
            
            full_cycle = True
            X_temp = X.drop(columns=drop_cols)

            for i, col in enumerate(cycle_columns):
                new_score = self.leave_one_out(X_temp, y, ids, col)
                diff = new_score - baseline
                if self.maximize:
                    diff *= -1

                if diff < threshold:
                    drop_cols = [*drop_cols, col]
                    # restarting cycle at next column
                    cycle_columns = [*cycle_columns[i+1:],*cycle_columns[:i]]
                    baseline = new_score
                    full_cycle = False
                    break

        self.dropped_columns = drop_cols
        return drop_cols


    def tune(self, n_trials : int = 50):
        pass

if __name__ == "__main__":
    print(Modeler('rdg', classify=True).constructor)