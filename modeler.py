from typing import Union, Tuple, Any, Dict, List
from gc import collect
from utils import title
import json

import xgboost as xgb
from xgboost import DMatrix

from sklearn.metrics import r2_score, log_loss

import pandas as pd
from pandas import DataFrame
import numpy as np
import optuna

from dataprocessor import DataProcessor
from model_info import ModelInfo


class Modeler:

    def __init__(self,
                 model: str,
                 data: DataProcessor,
                 goal: str,
                 device: Union[str, None] = None) -> None:

        assert goal in ['reg', 'clf', 'mclf'], \
            "goal must be \'reg\', \'clf\', or \'mclf\'."
        match goal:
            case 'reg':
                objective = 'reg:squarederror'
                self.criterion = r2_score
                self.direction = 'maximize'

            case 'clf':
                objective = 'binary:logistic'
                self.criterion = log_loss
                self.direction = 'minimize'

            case 'mclf':
                objective = 'multi:softmax'
                self.criterion = log_loss
                self.direction = 'minimize'

        self.info = ModelInfo(model, device=device, objective=objective)
        self.model = model

        self.instance = None
        self.data = data
        self.features = self.data.get_columns(dummies=True)
        return

    def fit(self,
            data: DMatrix,
            params: Union[dict, None] = None,
            num_boost_round: Union[int, None] = None) -> None:
        '''
        Fits new model with (X,y)-data and optionally custom parameters.

        Arguments:
            X: pandas dataframe for input data
            y: pandas dataframe for target / output data
            params: custom parameters
        '''

        default_params, default_num_boost_round = self.info.minimal_parameters()
        if params is None:
            params = default_params
        if num_boost_round is None:
            num_boost_round = default_num_boost_round

        self.instance = xgb.train(params, data, num_boost_round)
        return self.instance

    def predict(self, X: DataFrame) -> Any:
        '''
        Predicts on X with last created model instance.

        Arguments:
            X: pandas dataframe used to generate predictions
        '''

        dtest = DMatrix(X)
        return self.instance.predict(dtest)

    def get_fold(self, fold: int) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        '''
        Given fold, gives train data and validation data. Since the Modeler
        class selects features, it returns the X data on only the selected
        features.

        Arguments:
            fold: fold of cross-validation, 0-indexed
        '''

        Xt, yt, Xv, yv = self.data.get_fold(fold, dummies=True)

        return Xt[self.features], yt, Xv[self.features], yv

    def best_for_fold(self, fold: int) -> Any:
        '''
        Given fold, fits best model on train data and returns predictions on
        validation data. Ideally, this function should be called after
        hyperparameter tuning in order for best parameters to be determined.
        If not, the best parameters are taken to be the default parameters.

        Arguments:
            fold: cross validation fold, 0-indexed
        '''

        Xt, yt, Xv, _ = self.get_fold(fold)

        dtrain = DMatrix(Xt, yt)
        dfold = DMatrix(Xv)
        params, num_boost_round = self.info.get_best_parameters()

        self.fit(dtrain, params=params, num_boost_round=num_boost_round)
        return self.instance.predict(dfold)

    def best_for_test(self) -> Any:
        '''
        Fits best model on all train data and returns predictions on test data.
        Ideally, this function should be called after hyperparameter tuning in
        order for best parameters to be determined. If not, the best parameters
        are taken to be the default parameters.

        Arguments:
            None
        '''

        params, num_boost_round = self.info.get_best_parameters()

        Xt, yt = self.data.get_train(preprocessed=True, dummies=True)
        dtrain = DMatrix(Xt, yt)

        Xtest = self.data.get_test(preprocessed=True, dummies=True)
        dtest = DMatrix(Xtest)

        self.fit(dtrain, params=params, num_boost_round=num_boost_round)
        return self.instance.predict(dtest)

    def score(self, X: DataFrame, y: DataFrame) -> float:
        '''
        Uses (X,y)-data on last model instance to make predictions and
        returns a score.

        Arguments:
            X: pandas dataframe used to generate predictions
            y: pandas dataframe with outputs to score against predictions
        '''
        data = DMatrix(X)
        prediction = self.instance.predict(data)
        return self.criterion(y, prediction)

### Feature Selection

    def feature_importances(self) -> Dict[str, float]:

        importances = self.instance.get_score(importance_type='total_gain')
        for col in self.features:
            if col not in importances:
                importances[col] = 0.0

        return importances

    def __ascending_features(self) -> List[str]:

        importances = self.feature_importances()
        ascending = np.argsort(list(importances.values()))[::-1]
        keys = list(importances.keys())
        ascending_features = [keys[i] for i in ascending]
        return ascending_features

    def leave_one_out(self, col: str = '') -> float:
        '''
        Trains and scores model on data with the given column excluded. If no column
        is given, it trains and scores model on full data.

        Arguments:
            col: column name to exclude
        '''

        params, num_boost_round = self.info.feature_selection_parameters()

        error = 0
        for fold in range(self.data.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)
            if col:
                Xt = Xt.drop(columns=[col])
                Xv = Xv.drop(columns=[col])

            dtrain = DMatrix(Xt, yt)
            self.fit(dtrain, params=params, num_boost_round=num_boost_round)

            error += self.score(Xv, yv)

        average_error = error / self.data.n_folds
        return average_error

    def dimension_reduction(self, threshold: float = 0.0, display: bool = True) -> list:
        '''
        Iteratively removes features that do not improve model score beyond threshold.

        Arguments:
            threshold: the minimum improvement in score to justify keeping feature
            display: whether or not to print info
        '''

        baseline = self.leave_one_out()
        if display:
            print(f'Baseline: {baseline:.6f}')
        full_cycle = False
        cycle_columns = self.__ascending_features()
        while not full_cycle:
            full_cycle = True
            for col in cycle_columns:
                new_score = self.leave_one_out(col)
                diff = baseline - new_score
                
                if self.direction == 'minimize':
                    diff = -diff

                if diff < threshold:
                    self.features.remove(col)
                    cycle_columns = self.__ascending_features()
                    baseline = new_score
                    full_cycle = False
                    if display:
                        print(f'New Score: {new_score:.6f} || Dropping {col} ...')
                    collect()  # quick clean-up
                    break

                if display:
                    print(f'Tried {col}.')
        
        collect() # quick clean-up
        return [*self.features]

### Tuning

    def __cv_score_tune(self, params: dict, num_boost_round: int) -> float:
        '''
        Given model parameters, this function returns the average score across all
        folds on the out-of-fold data. Private method.

        Arguments:
            params: model parameters
        '''

        total_score = 0
        for fold in range(self.data.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)

            dtrain = DMatrix(Xt, yt)
            self.fit(dtrain, params=params, num_boost_round=num_boost_round)
            total_score += self.score(Xv, yv)

        avg_score = total_score / self.data.n_folds
        return avg_score

    def __optuna_objective(self, trial: optuna.Trial):
        '''
        Creates an objective function for optuna trials based on underlying 
        parameters.

        Arguments:
            trial: optuna.Trial object
        '''

        parameters, num_boost_round = self.info.trial_parameters(trial)
        return self.__cv_score_tune(parameters, num_boost_round)

    def tune(self, n_trials : int = 100) -> None:
        '''
        Uses optuna library to tune hyperparameters.

        Arguments:
            n_trials: number of trials
        '''

        study = optuna.create_study(direction=self.direction,
                                    sampler=optuna.samplers.TPESampler())

        study.optimize(self.__optuna_objective, n_trials=n_trials)

        self.info.set_best_parameters(study.best_params)
        return

### Save/Load

    def save_model(self, save_path: str) -> None:
        '''
        Saves model to save path.

        Arguments:
            save_path: path on disk to save model
        '''

        # allow flexibility in path notation
        if save_path[-1] == '/':
            save_path[-1] == ''

        self.instance.save_model(f'{save_path}/model_{self.model}.json')
        self.instance.save_config(f'{save_path}/config_{self.model}.json')

        class_config = {'features' : [*self.features],
                        'best_params' : {**self.info.best_params},
                        'best_boost' : self.info.best_boost}
        
        with open(f'{save_path}/class_config_{self.model}.json', 'w') as fn:
            json.dump(class_config, fn)

        return

    def load_model(self, load_path: str) -> None:
        '''
        Loads previously saved model from load path.

        Arguments:
            load_path: path on disk where model is located
        '''

        # allow flexibility in path notation
        if load_path[-1] == '/':
            load_path[-1] == ''
        
        self.instance = xgb.Booster()
        self.instance.load_model(f'{load_path}/model_{self.model}.json')
        self.instance.load_config(f'{load_path}/config_{self.model}.json')

        with open(f'{load_path}/class_config_{self.model}.json', 'r') as fn:
            class_config = json.load(fn)

        self.features = class_config['features']
        self.info.best_params = class_config['best_params']
        self.info.best_boost = class_config['best_boost']
        return

    def save_predictions(self, save_path: str):
        '''
        Saves predictions for test set and each fold to the given path.

        Arguments:
            save_path: string path of folder in which predictions are saved
        '''

        # allow flexibility in path notation
        if save_path[-1] == '/':
            save_path[-1] == ''

        for fold in range(self.data.n_folds):
            prediction = np.array(self.best_for_fold(fold))
            np.save(f'{save_path}/{self.model}_fold_{fold}', prediction)

        prediction = np.array(self.best_for_test())
        np.save(f'{save_path}/{self.model}_test', prediction)
        return

    def all_at_once(self, save_path: str, loss_threshold: float = 0.0, n_trials: int = 100):
        '''
        Performs dimension reduction, hyperparameter tuning, and prediction saving
        sequentially.

        Arguments:
            save_path: path for saving predictions
            loss_threshold: minimum improvement in score to justify keeping feature
        '''

        title("Dimension Reduction")
        self.dimension_reduction(threshold=loss_threshold)

        title("Hyperparameter Tuning")
        self.tune(n_trials = n_trials)

        title("Saving Predictions")
        self.save_predictions(save_path)

        collect()
        return


if __name__ == "__main__":

    # train = pd.read_csv('./data/s4e5/train.csv').loc[:8000]
    # test = pd.read_csv('./data/s4e5/test.csv')
    # data = DataProcessor(train, test_data=test, primary_column='id')
    # data.preprocess(n_clusters=3)
    # data.set_cv()

    # xgboost = Modeler("xgb", data, goal='r')
    # xgboost.dimension_reduction()
    # xgboost.tune()

    train = pd.read_csv('./data/s4e5/train.csv')
    test = pd.read_csv('./data/s4e5/test.csv')
    data = DataProcessor(train, test_data=test, primary_column='id')
    data.set_cv()

    Xt, yt, Xv, yv = data.get_fold(0)

    for modeltype in [
        'hist_long',
        'hist_wide',
        'prox_long',
        'prox_wide',
        'rndm_long',
        'rndm_wide',
        'linr'
    ]:
        model = Modeler(modeltype, data, 'reg')
        dtrain = DMatrix(Xt, yt)
        model.fit(dtrain)
        print(f'{modeltype} : {model.score(Xv, yv) : .6f}')
