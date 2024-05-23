from typing import Union, Any
from gc import collect
from utils import title

import joblib
import pandas as pd
import numpy as np
import optuna

from dataprocessor import DataProcessor
from model import Model

HYPERPARAMETER_SPACE = {
    # model : [(param name, min of range, max of range, float?, log?)]
    'xgb': [('learning_rate', 1e-2, 1e-1, True, True),
            ('max_depth', 4, 10, False, True),
            ('n_estimators', 1e2, 1e3, False, True),
            ('subsample', 2e-1, 1, True, True),
            ('lambda', 1, 1e2, True, True),
            ('alpha', 1e-8, 1e-2, True, True)],
    'lgb': [('learning_rate', 1e-2, 1e-1, True, True),
            ('max_depth', 5, 11, False, False),
            ('num_iterations', 1e2, 1e3, False, True),
            ('num_leaves', 1e1, 1e2, False, True)],
    'hgb': [('learning_rate', 1e-2, 1e-1, True, True),
            ('min_samples_leaf', 10, 50, False, True),
            ('max_iter', 1e2, 1e3, False, True),
            ('max_bins', 63, 255, False, True)],
    'rf' : [('n_estimators', 1e2, 1e3, False, True),
            ('max_depth', 3, 10, False, True),
            ('min_samples_split', 10, 50, False, True),
            ('max_samples', 5e-2, 5e-1, True, True)],
    'lin': [],
    'rdg': [('alpha', 1e-1, 1e1, True, True)],
    'svm': [('C', 1e-1, 1e1, True, True),
            ('gamma', 1e-2, 1, True, True)]
}

DEFAULT_PARAMS = {
    'xgb': {'n_jobs':-1},
    'lgb': {'verbosity':-1, 'n_jobs':-1},
    'hgb': {},
    'rf':  {'n_jobs':-1},
    'lin': {},
    'rdg': {},
    'svm': {}
}

TOY_PARAMS = {
    'xgb': {'learning_rate': 0.05,
            'max_depth': 7,
            'n_estimators': 250,
            'subsample': 0.7,
            'n_jobs': -1},
    'lgb': {'learning_rate': 0.05,
            'num_leaves': 50,
            'num_iterations': 250,
            'n_jobs':-1,
            'verbosity':-1},
    'hgb': {'learning_rate': 0.05,
            'max_depth': 6,
            'max_iter': 250,
            'max_bins': 63},
    'rf': {'n_estimators': 250,
           'max_depth': 5,
           'min_samples_split': 10,
           'max_samples': 0.1,
           'n_jobs': -1},
    'lin': {},
    'rdg': {'alpha': 1},
    'svm': {'gamma': 0.1}
}

class Modeler:

    def __init__(self, 
                 model_type: str, 
                 data: DataProcessor,
                 goal: str) -> None:

        # Verify model type is supported.
        assert model_type in HYPERPARAMETER_SPACE, "Not an implemented model type: "\
            + f"Choose from {HYPERPARAMETER_SPACE.keys()}."
        
        # Determine if multi- or single-target
        self.multi_target = data.multi_target
        
        # Determine classification / regression
        assert goal in ['C', 'R', 'c', 'r'], "Goal must be 'C' or 'R' for classification or regression."
        self.classify = goal in ['C', 'c']

        self.model_type = model_type
        self.model = Model(model_type=model_type, classify=self.classify, multiple_targets=self.multi_target)

        self.params = DEFAULT_PARAMS[model_type]
        self.hyperparams = HYPERPARAMETER_SPACE[model_type]

        self.data = data
        self.n_folds = self.data.n_folds
        self.features = self.data.get_columns(dummies=True)
        self.toy_params = TOY_PARAMS[model_type]
        self.best_params = {**self.params}
        return

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.DataFrame,
            params: Union[dict, None] = None) -> Any:
        '''
        Fits new model with (X,y)-data and optionally custom parameters.

        Arguments:
            X: pandas dataframe for input data
            y: pandas dataframe for target / output data
            params: custom parameters
        '''
        if params is None:
            params=self.params

        self.model = Model(self.model_type, params=params,
                           classify=self.classify)
        self.model.fit(X, y)
        
        collect()  # old instance may be inaccessible now; reclaim memory
        return self.model

    def predict(self, X : pd.DataFrame) -> Any:
        '''
        Predicts on X with last created model instance.
        
        Arguments:
            X: pandas dataframe used to generate predictions
        '''

        return self.model.predict(X)

    def get_fold(self, fold: int) -> tuple:
        '''
        Given fold, gives train data and validation data. Since the Modeler
        class selects features, it returns the X data on only the selected
        features.
        
        Arguments:
            fold: fold of cross-validation, 0-indexed
        '''

        Xt, yt, Xv, yv = self.data.get_fold(fold, dummies=True)
        Xt = Xt[self.features]
        Xv = Xv[self.features]
        return (Xt, yt, Xv, yv)

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
        self.model = Model(model_type=self.model_type, params=self.best_params,
                           classify=self.classify)
        self.model.fit(Xt, yt)
        return self.model.predict(Xv)

    def best_for_test(self) -> Any:
        '''
        Fits best model on all train data and returns predictions on test data. 
        Ideally, this function should be called after hyperparameter tuning in 
        order for best parameters to be determined. If not, the best parameters
        are taken to be the default parameters.

        Arguments:
            None
        '''

        Xt, yt = self.data.get_train(preprocessed=True, dummies=True)
        self.model = Model(model_type=self.model_type, params=self.best_params,
                           classify=self.classify)
        self.model.fit(Xt, yt)
        Xtest = self.data.get_test(preprocessed=True, dummies=True)
        return self.model.predict(Xtest)

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        '''
        Uses (X,y)-data on last model instance to make predictions and 
        returns a score.

        Arguments:
            X: pandas dataframe used to generate predictions
            y: pandas dataframe with outputs to score against predictions
        '''

        return self.model.score(X, y)

    def leave_one_out(self, col: str = '') -> float:
        '''
        Trains and scores model on data with the given column excluded. If no column
        is given, it trains and scores model on full data.

        Arguments:
            col: column name to exclude 
        '''

        error = 0
        for fold in range(self.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)
            if col:
                Xt = Xt.drop(columns=[col])
                Xv = Xv.drop(columns=[col])

            self.fit(Xt, yt, params=self.toy_params)
            error += self.score(Xv, yv)

        average_error = error / self.n_folds
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
            print(f'Baseline: {baseline}')
        full_cycle = False
        cycle_columns = self.features[::-1]
        while not full_cycle:
            full_cycle = True
            for i, col in enumerate(cycle_columns):
                new_score = self.leave_one_out(col)
                diff = baseline - new_score

                if diff < threshold:
                    self.features.remove(col)
                    # restarting cycle at next column
                    cycle_columns = [*cycle_columns[i+1:], *cycle_columns[:i]]
                    baseline = new_score
                    full_cycle = False
                    if display:
                        print(f'New Score: {new_score} // Dropping {col} ...')
                    collect()  # quick clean-up
                    break

                if display:
                    print(f'Tried {col}.')

        return [*self.features]
    
## Tuning

    def __cv_score_tune(self, params: dict) -> float:
        '''
        Given model parameters, this function returns the average score across all 
        folds on the out-of-fold data. Private method.

        Arguments:
            params: model parameters
        '''

        total_score = 0
        for fold in range(self.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)

            self.fit(Xt, yt, params=params)
            total_score += self.score(Xv, yv)

        avg_score = total_score / self.n_folds
        return avg_score

    def __objective_tune(self, hyperparams: dict, trial: optuna.Trial,
                         rigidparams: dict = {}):
        '''
        Creates an objective function for optuna trials, given variable hyperparameters
        and rigid parameters. Private method.

        Arguments:
            hyperparams: hyperparameters from global dictionary for tuning
            trial: optuna.Trial object
            rigidparams: hyperparameters that are constant
        '''

        params = rigidparams.copy()
        for _, param in hyperparams.items():
            name, min, max, flt, log = param
            if flt:
                params[name] = trial.suggest_float(name, min, max, log=log)
            else:
                params[name] = trial.suggest_int(name, min, max, log=log)

        return self.__cv_score_tune(params)

    def tune(self, two_stage: bool = True) -> None:
        '''
        Uses optuna library to tune hyperparameters. If two_stage is True,
        it performs a first wave to freeze unimportant hyperparameters.

        Arguments:
            two_stage: if true, performs two-stage hyperparameter tuning
        '''

        if len(self.hyperparams) == 0:
            return

        # no need to eliminate hyperparams if there's only one
        if len(self.hyperparams) == 1:
            two_stage = False

        n_initial = 30

        rigidparams = {**self.params}
        hyperparams = {
            param[0]: param for param in self.hyperparams
        }

        # Inital Pass
        if two_stage:
            initial_study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.RandomSampler())
            
            # creating single-argument callable for study, based on hyperparameters
            def initial_objective(trial):
                return self.__objective_tune(hyperparams, trial)

            initial_study.optimize(initial_objective, n_trials=n_initial)
            importances = optuna.importance.get_param_importances(
                initial_study)
            for param, importance in importances.items():
                if importance < 1e-1: # less than 10% of importance
                    del hyperparams[param]
                    rigidparams[param] = initial_study.best_params[param]

        collect()  # quick clean-up

        # Main Study
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler())
        
        # creating single-argument callable for study, based on hyperparameters
        def objective(trial):
            return self.__objective_tune(hyperparams, trial, rigidparams=rigidparams)
        
        n_trials = 15 * len(hyperparams)
        study.optimize(objective, n_trials = n_trials)

        self.best_params = {**rigidparams, **study.best_params}
        return

    def save_model(self, save_path: str) -> None:
        '''
        Saves model to save path.
        
        Arguments:
            save_path: path on disk to save model
        '''

        # allow flexibility in path notation
        if save_path[-1] == '/':
            save_path[-1] == ''

        joblib.dump(self.model, f'{save_path}/model_{self.model_type}')
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

        self.model = joblib.load(f'{load_path}/model_{self.model_type}')
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

        for fold in range(self.n_folds):
            prediction = np.array(self.best_for_fold(fold))
            path = save_path + f'/{self.model_type}_fold_{fold}'
            np.save(path, prediction)

        prediction = np.array(self.best_for_test())
        path = save_path + f'/{self.model_type}_test'
        np.save(path, prediction)
        return

    def all_at_once(self, save_path: str, loss_threshold: float = 1e-4):
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
        self.tune()

        title("Saving Predictions")
        self.save_predictions(save_path)

        collect()
        return

if __name__ == "__main__":

    #train = pd.read_csv('./data/s4e5/train.csv').loc[:8000]
    #test = pd.read_csv('./data/s4e5/test.csv')
    #data = DataProcessor(train, test_data=test, primary_column='id')
    #data.preprocess(n_clusters=3)
    #data.set_cv()

    #xgboost = Modeler("xgb", data, goal='r')
    #xgboost.dimension_reduction()
    #xgboost.tune()

    train = pd.read_csv('./data/s4e3/train.csv')
    test = pd.read_csv('./data/s4e3/test.csv')
    data = DataProcessor(train, test_data=test, primary_column='id')
    data.set_cv()

    Xt, yt, Xv, yv = data.get_fold(0)

    for modeltype in HYPERPARAMETER_SPACE:
        model = Modeler(modeltype, data, goal='c')
        model.fit(Xt, yt)
        print(f'{modeltype} : {model.score(Xv, yv) : .4f}')

