from typing import Union, Any
from gc import collect

from utils import title

import pandas as pd
import numpy as np

from dataprocessor import DataProcessor
from model import Model

import optuna

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
    'cat': [('depth', 4, 10, False, False),
            ('n_estimators', 1e2, 1e3, False, True)],
    'hgb': [('learning_rate', 1e-2, 1e-1, True, True),
            ('max_depth', 4, 10, False, True),
            ('max_iter', 2e2, 2e3, False, True),
            ('max_bins', 63, 255, False, True)],
    'rf': [('n_estimators', 2e2, 2e3, False, True),
           ('max_depth', 3, 8, False, True),
           ('min_samples_split', 2, 10, False, False),
           ('max_samples', 1e-1, 1, True, True)],
    'lin': [],
    'rdg': [('alpha', 1e-1, 1e1, True, True)],
    'svm': [('C', 1e-1, 1e1, True, True),
            ('gamma', 1e-2, 1, True, True)]
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
            'n_jobs': -1},
    'cat': {'depth': 6,
            'n_estimators': 250,
            'n_jobs': -1},
    'hgb': {'learning_rate': 0.05,
            'max_depth': 6,
            'max_iter': 250,
            'max_bins': 63,
            'n_jobs': -1},
    'rf': {'n_estimators': 250,
           'max_depth': 5,
           'min_samples_split': 10,
           'max_samples': 0.1,
           'n_jobs': -1},
    'lin': {},
    'rdg': {'alpha': 1},
    'svm': {'C': 1,
            'gamma': 0.1}
}


class Modeler:

    # Modeler represents a single model
    # The goal is to find optimal features
    # and fix hyperparameters based on CV Scheme

    def __init__(self, model_type: str, data: DataProcessor, params: dict = {},
                 classify: bool = False, columns: list = [],
                 dummies: bool = False) -> None:

        assert model_type in HYPERPARAMETER_SPACE, "Not an implemented model type: "\
            + f"Choose from {HYPERPARAMETER_SPACE.keys()}."

        # We want to remember model constructor and parameters, not an instance.
        self.model = Model(model_type=model_type, params=params,
                           classification=classify)

        self.model_type = model_type
        self.params = params
        self.classification = classify

        self.hyperparams = HYPERPARAMETER_SPACE[model_type]
        self.columns = columns.copy()

        self.data = data
        self.n_folds = self.data.n_folds
        self.features = self.data.get_columns(dummies=dummies)
        self.dummies = dummies

        self.toy_params = TOY_PARAMS[model_type]

        self.best_params = {}
        return

    def fresh_fit(self, X: pd.DataFrame, y: pd.DataFrame,
                  params: dict = {}) -> Any:
        '''
        Fits new model with (X,y)-data and optionally custom parameters.

        Arguments:
            X: pandas dataframe for input data
            y: pandas dataframe for target / output data
            params: custom parameters (default: {})
        '''
        self.model = Model(self.model_type, params=params,
                           classification=self.classification)
        self.model.fit(X, y)
        collect()  # old instance may be inaccessible now; reclaim memory
        return self.model

    def predict(self, X : pd.DataFrame):
        '''
        Predicts on X with last created model instance.
        
        Arguments:
            X: pandas dataframe used to generate predictions
        '''
        return self.model.predict(X)

    def get_fold(self, fold: int):
        '''
        Given fold, gives train data and validation data. Since the Modeler
        class selects features, it returns the X data on only the selected
        features.
        
        Arguments:
            fold: fold of cross-validation, 0-indexed
        '''
        Xt, yt, Xv, yv = self.data.get_fold(fold, dummies=self.dummies)
        Xt = Xt[self.features]
        Xv = Xv[self.features]
        return (Xt, yt, Xv, yv)

    def best_for_fold(self, fold: int):
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
                           classification=self.classification)
        self.model.fit(Xt, yt)
        return self.model.predict(Xv)

    def best_for_test(self):
        '''
        Fits best model on all train data and returns predictions on test data. 
        Ideally, this function should be called after hyperparameter tuning in 
        order for best parameters to be determined. If not, the best parameters
        are taken to be the default parameters.

        Arguments:
            None
        '''
        Xt, yt = self.data.get_train(preprocessed=True, dummies=self.dummies)
        self.model = Model(model_type=self.model_type, params=self.best_params,
                           classification=self.classification)
        self.model.fit(Xt, yt)
        Xtest = self.data.get_test(preprocessed=True, dummies=self.dummies)
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
        error = 0
        for fold in range(self.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)
            if col:
                Xt = Xt.drop(columns=[col])
                Xv = Xv.drop(columns=[col])

            self.fresh_fit(Xt, yt, params=self.toy_params)
            error += self.score(Xv, yv)

        average_error = error / self.n_folds
        return average_error

    # DONE
    def dimension_reduction(self, threshold: float = 0.0, display: bool = True) -> list:

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

    # DONE
    def cv_score_for_tune(self, params: dict) -> float:
        '''
        cv scheme scoring
        '''
        total_score = 0
        for fold in range(self.n_folds):

            Xt, yt, Xv, yv = self.get_fold(fold)

            self.fresh_fit(Xt, yt, params=params)
            total_score += self.score(Xv, yv)

        avg_score = total_score / self.n_folds
        return avg_score

    # DONE
    def objective_for_tune(self, hyperparams: dict, trial: optuna.Trial,
                           rigid_params: dict = {}):

        params = rigid_params.copy()
        for _, param in hyperparams.items():
            name, min, max, flt, log = param
            if flt:
                params[name] = trial.suggest_float(name, min, max, log=log)
            else:
                params[name] = trial.suggest_int(name, min, max, log=log)

        return self.cv_score_for_tune(params)

    # DONE
    def tune(self, n_trials: int = 50, two_stage: bool = True) -> None:

        if len(self.hyperparams) == 0:
            self.best_params = {}
            return

        # no need to eliminate hyperparams if there's only one
        if len(self.hyperparams) == 1:
            two_stage = False

        n_initial = 30

        rigid_params = {'random_state': 0}
        hyperparams = {
            param[0]: param for param in self.hyperparams
        }

        # Inital Pass
        if two_stage:
            initial_study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.RandomSampler())

            def initial_objective(trial):
                return self.objective_for_tune(hyperparams, trial)

            initial_study.optimize(initial_objective, n_trials=n_initial)
            importances = optuna.importance.get_param_importances(
                initial_study)
            for param, importance in importances.items():
                if importance < 1e-1:
                    del hyperparams[param]
                    rigid_params[param] = initial_study.best_params[param]

        collect()  # quick clean-up

        # Main Study
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler())

        def objective(trial):
            return self.objective_for_tune(hyperparams, trial, rigid_params=rigid_params)

        study.optimize(objective, n_trials=n_trials)

        self.best_params = {**rigid_params, **study.best_params}
        return

    def save_predictions(self, save_path: str):
        for fold in range(self.n_folds):
            prediction = np.array(self.best_for_fold(fold))
            path = save_path + f'/{self.model_type}_fold_{fold}'
            np.save(path, prediction)

        prediction = np.array(self.best_for_test())
        path = save_path + f'/{self.model_type}_test'
        np.save(path, prediction)
        return

    def all_at_once(self, save_path: str, loss_threshold: float = 1e-4):
        title("Dimension Reduction")
        self.dimension_reduction(threshold=loss_threshold)
        title("Hyperparameter Tuning")
        self.tune()
        title("Saving Predictions")
        self.save_predictions(save_path)
        collect()
        return


if __name__ == "__main__":

    train = pd.read_csv('./data/s4e5/train.csv').loc[:8000]
    test = pd.read_csv('./data/s4e5/test.csv')
    data = DataProcessor(train, test_data=test, primary_column='id')
    data.preprocess(n_clusters=3)
    data.set_cv()

    xgboost = Modeler("xgb", data, dummies=True)
    xgboost.dimension_reduction()
    xgboost.tune()
