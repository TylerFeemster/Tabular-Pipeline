import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn import ensemble as ens, linear_model as lnr, svm
from sklearn.metrics import r2_score, roc_auc_score

ALL_MODELS = {
    # model : [(Regression, Classification), dummies?, y as array?]
    'xgb': [(xgb.XGBRegressor, xgb.XGBClassifier), True, False],
    'lgb': [(lgb.LGBMRegressor, lgb.LGBMClassifier), True, False],
    'cat': [(cat.CatBoostRegressor, cat.CatBoostClassifier), False, False],
    'rf' : [(ens.RandomForestRegressor,
            ens.RandomForestClassifier), True, True],
    'hgb': [(ens.HistGradientBoostingRegressor,
            ens.HistGradientBoostingClassifier), True, True],
    'lin': [(lnr.LinearRegression, lnr.LogisticRegression), True, True],
    'rdg': [(lnr.Ridge, lnr.RidgeClassifier), True, True],
    'svm': [(svm.SVR, svm.SVC), True, True]
}

class Model:
    def __init__(self, model_type : str, params : dict = {}, 
                 classification : bool = False, multiple_targets : bool = False):
        
        self.constructor = ALL_MODELS[model_type][0][classification]
        self.instance = self.constructor(**params)

        if classification:
            self.criterion = roc_auc_score
        else:
            self.criterion = r2_score

        self.multi_target = multiple_targets
        self.use_dummies = ALL_MODELS[model_type][1]
        self.target_as_numpy = ALL_MODELS[model_type][2]

    def format_target(self, y : pd.DataFrame):

        if self.target_as_numpy:
            y = np.array(y)

            if not self.multi_target:
                y = np.reshape(y, -1)

        return y

    def fit(self, X : pd.DataFrame, y : pd.DataFrame):
        y = self.format_target(y)
        self.instance.fit(X, y)
        return

    def predict(self, X : pd.DataFrame):
        return self.instance.predict(X)

    def fit_predict(self, X : pd.DataFrame, y : pd.DataFrame):
        y = self.format_target(y)
        self.instance.fit(X, y)
        return self.instance.predict(X)
    
    def score(self, X : pd.DataFrame, y : pd.DataFrame):
        y = self.format_target(y)
        predictions = self.predict(X)

        if not self.multi_target:
            return self.criterion(y, predictions)
        
        # when multitarget
        n_targets = predictions.shape[1]
        
        score = 0
        for idx in range(n_targets):
            score += self.criterion(y[:,idx], predictions[:,idx])
        
        avg = score / n_targets
        return avg
