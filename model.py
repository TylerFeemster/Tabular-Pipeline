import pandas as pd
import numpy as np

from utils import gpu_available
import xgboost as xgb
import lightgbm as lgb
from sklearn import ensemble as ens, linear_model as lnr, svm
from sklearn.metrics import r2_score, roc_auc_score

MODEL_INFO = {
    # model : [(Regression, Classification), y as numpy?]
    'xgb': [(xgb.XGBRegressor, xgb.XGBClassifier), False],
    'lgb': [(lgb.LGBMRegressor, lgb.LGBMClassifier), False],
    'rf' : [(ens.RandomForestRegressor,
            ens.RandomForestClassifier), True],
    'hgb': [(ens.HistGradientBoostingRegressor,
            ens.HistGradientBoostingClassifier), True],
    'lin': [(lnr.LinearRegression, lnr.LogisticRegression), True],
    'rdg': [(lnr.Ridge, lnr.RidgeClassifier), True],
    'svm': [(svm.SVR, svm.SVC), True]
}

class Model:
    def __init__(self, 
                 model_type : str, 
                 params : dict = {}, 
                 classify : bool = False, 
                 multiple_targets : bool = False):
        '''
        Defines model class which fits and predicts. It standardizes models
        which use slightly different syntax and formatting.

        Arguments:
            model_type: one of 'xgb', 'lgb', 'rf', 'hgb', 'lin', 'rdg', 'svm'; 
            these are xgboost, lightgbm, randomforest, histgradboost, 
            linear regression (logistic for classification), ridge
            params: parameters for initializing model
            classify: True for classification, False for regression
            multiple_targets: True for multiple targets, False for single target
        '''
        #in case of gpu
        gpu_params = {}
        if gpu_available():
            match model_type:
                case 'xgb':
                    gpu_params = {
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor',
                        'gpu_id': 0
                    }
                case 'lgb':
                    gpu_params = {
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    }

        ## in case of multiclass

        extra_params = {}
        self.fit_params = {}
        if multiple_targets:
            match model_type:
                case 'lgb':
                    extra_params = {
                        'objective': 'multiclass'
                    }
                    self.fit_params = {
                        'eval_metric':'multi_logloss'
                    }
                case 'lin':
                    extra_params = {
                        'solver': 'lbfgs',
                        'multi_class': 'ovr'
                    }
                case 'svm':
                    extra_params = {
                        'decision_function_shape': 'ovr'
                    }


        params = {**params, **gpu_params, **extra_params}
        
        self.constructor = MODEL_INFO[model_type][0][classify]
        self.instance = self.constructor(**params)
        self.model_type = model_type

        if classify:
            self.criterion = roc_auc_score
        else:
            self.criterion = r2_score
        self.classify = classify

        self.multi_target = multiple_targets
        self.target_as_numpy = MODEL_INFO[model_type][1]

    def format_target(self, y : pd.DataFrame):
        '''
        Some models take numpy format, some take pandas dataframes. This
        standardizes output data by taking pandas dataframe and converting 
        it to numpy when required. It also reshapes when required.

        Arguments:
            y: pandas dataframe with target data
        '''

        if self.target_as_numpy:
            y = np.array(y)
            if not self.multi_target:
                y = np.reshape(y, -1)
        return y

    def fit(self, X : pd.DataFrame, y : pd.DataFrame):
        '''
        Fits (X,y)-data to model.

        Arguments:
            X: pandas dataframe containing input
            y: pandas dataframe containing learnable target
        '''

        y = self.format_target(y)
        self.instance.fit(X, y, **self.fit_params)
        return

    def predict(self, X : pd.DataFrame):
        '''
        Uses model instance to predict outputs on given input dataframe.

        Arguments:
            X: input pandas dataframe for generating output
        '''
        if self.classify:
            if self.model_type == 'rdg':
                return self.instance.decision_function(X)
            return self.instance.predict_proba(X)
        return self.instance.predict(X)

    def fit_predict(self, X : pd.DataFrame, y : pd.DataFrame):
        '''
        Fits (X,y)-data and predicts outputs on same X.

        Arguments:
            X: input as pandas dataframe
            y: learnable targets as pandas dataframe
        '''

        y = self.format_target(y)
        self.instance.fit(X, y)
        return self.instance.predict(X)
    
    def score(self, X : pd.DataFrame, y : pd.DataFrame):
        '''
        Predicts outputs using X and scores against the given y.
        In the case of multitarget data, it scores each individually
        and then averages scores. It uses R2 for regression and
        AUROC for classification.

        Arguments:
            X: input dataframe to create predictions
            y: true output to score against predictions
        '''

        y = self.format_target(y)
        predictions = self.predict(X)
        if not self.multi_target:
            return self.criterion(y, predictions)

        # when multitarget
        if self.target_as_numpy:
            avg_score = np.mean([self.criterion(y[:,i], predictions[:,i]) 
                                 for i in range(predictions.shape[1])])
        else:
            avg_score = np.mean([self.criterion(y[col], predictions[:, i])
                                 for i, col in enumerate(y.columns)])

        return avg_score

if __name__ == "__main__":
    train = pd.read_csv('./data/s4e3/train.csv')
    test = pd.read_csv('./data/s4e3/test.csv')

    from dataprocessor import DataProcessor
    data = DataProcessor(train_data=train, test_data=test, primary_column='id')
    data.set_cv()
    Xt, yt, Xv, yv = data.get_fold(0)
    print(yt.shape)
    
    model = Model('xgb', classify=True, multiple_targets=True)
    model.fit(Xt, yt)
    print(model.score(Xv, yv))
    

