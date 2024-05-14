import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator, title, subtitle, align_integer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from explorer import Explorer

from gc import collect
from typing import Union, Tuple

## Handle Missing Values (choice : delete / impute)
## Normalize Data
## PCA Features
## Cluster Analysis
## Feature Generation

class DataProcessor(Explorer):
    def __init__(self, train_data: pd.DataFrame,
                 test_data: Union[pd.DataFrame, None] = None,
                 primary_column: Union[str, None] = None,
                 target: Union[str, list, None] = None) -> None:
        '''

        Args:
            train_data: train dataframe
            test_data: test dataframe (optional)
            primary_column: primary key in dataframes (optional)
        '''
        if primary_column:
            # verify primary key is a dataframe column
            assert primary_column in train_data.columns
            self.train = train_data.set_index(primary_column, drop=True)
            self.primary_column = primary_column
        else:
            self.train = train_data.copy()
            self.primary_column = None

        self.targets = []
        if target:
            if type(target) is list:
                self.targets = target.copy()
            else:
                self.targets = [target]
        
        self.test = None
        if test_data is not None:
            if primary_column:
                assert primary_column in test_data.columns
                self.test = test_data.set_index(primary_column, drop=True)
            else:
                self.test = test_data.copy()

            # verify test columns don't have any new columns
            train_cols = set(train_data.columns)
            test_cols = set(test_data.columns)
            assert test_cols <= train_cols, "Some test columns not in train columns."
            
            set_difference = train_cols - test_cols
            assert set(self.targets) <= set_difference, "Targets must be in train set and not test set."
            
            if not self.targets:
                self.targets = list(set_difference)
    
        self.y = train_data[self.targets]

        self.X = self.train.drop(columns=self.targets)
        self.X_cols = self.X.columns

        self.train_num = self.X.select_dtypes(include=['number'])
        self.train_cat = self.X.select_dtypes(include=['object'])
        self.train_dum = None
        if self.test is not None:
            self.test_num = self.test.select_dtypes(include=['number'])
            self.test_cat = self.test.select_dtypes(include=['object'])
            self.test_dum = None

        self.cols = self.train.columns

        # Unsupervised: use all X available, train and test
        self.X_unsupervised = self.train_num.copy()
        if self.test is not None:
            self.X_unsupervised = pd.concat([self.X_unsupervised, 
                                             self.test_num],
                                             axis=0)
        stats = self.X_unsupervised.describe().T
        self.X_unsupervised = (self.X_unsupervised - stats['mean']) / stats['std']

        # Explorer
        super().__init__(self.train, targets=self.targets)

        # CV Scheme
        self.cv = None
        self.n_folds = None

### Adding Features

    def add_logs(self, force : bool = False) -> None:

        test = self.test is not None

        for col in self.train_num.columns:
            minimum = min(self.train_num[col])
            if test:
                minimum = min(minimum, min(self.test_num[col]))
            
            if minimum > 0:
                self.train_num[f'log {col}'] = np.log(self.train_num[col])
                if test:
                    self.test_num[f'log {col}'] = np.log(self.test_num[col])
            
            elif force:
                self.train_num[f'log {col}'] = np.log(self.train_num[col] - minimum + 1)
                if test:
                    self.test_num[f'log {col}'] = np.log(self.test_num[col] - minimum + 1)

        return

    def add_pca_components(self, n_components : Union[int, None] = None) -> None:

        if n_components is None or n_components > len(self.X_unsupervised.columns):
            n_components = len(self.X_unsupervised.columns)

        pca = PCA(n_components=n_components).fit(self.X_unsupervised)
        self.train_num[[f'pca {i+1}' for i in range(n_components)]] = pca.transform(
            self.X_unsupervised.loc[self.X.index])
        
        if self.test is not None:
            self.test_num[[f'pca {i+1}' for i in range(n_components)]] = pca.transform(
                self.X_unsupervised.loc[self.test.index])

        return
    
    def add_clusters(self, n_clusters : int = 8) -> None:
        
        kmeans = KMeans(n_clusters=n_clusters).fit(self.X_unsupervised)
        self.train_cat['Cluster'] = kmeans.predict(
            self.X_unsupervised.loc[self.X.index]).astype(str) # avoid meaningless numeric
        
        if self.test is not None:
            self.test_cat['Cluster'] = kmeans.predict(
                self.X_unsupervised.loc[self.test.index]).astype(str)

        return
    
    def all_additions(self, n_components : Union[int, None] = None, 
                      n_clusters : int = 8, force_log : bool = False) -> None:
        
        self.add_logs(force=force_log)
        self.add_clusters(n_clusters=n_clusters)
        self.add_pca_components(n_components=n_components)

        return

### Preprocessing

    def make_dummies(self):

        self.train_dum = pd.get_dummies(self.train_cat, dtype=int)
        if self.test is not None:
            self.test_dum = pd.get_dummies(self.test_cat, dtype=int)

        return

    def scale(self):

        if self.test is not None:
            num_scaler = RobustScaler(unit_variance=True)
            num_scaler.fit(pd.concat([self.train_num, self.test_num], axis=0))
            self.train_num[self.train_num.columns] = num_scaler.transform(self.train_num)
            self.test_num[self.train_num.columns] = num_scaler.transform(self.test_num)

            if self.train_dum is not None:
                dum_scaler = RobustScaler(unit_variance=True)
                dum_scaler.fit(pd.concat([self.train_dum, self.test_dum], axis=0))
                self.train_dum[self.train_dum.columns] = dum_scaler.transform(self.train_dum)
                self.test_dum[self.test_dum.columns] = dum_scaler.transform(self.test_dum)

            return
        
        # in the case of no test data
        num_scaler = RobustScaler(unit_variance=True)
        self.train_num[self.train_num.columns] = num_scaler.fit_transform(self.train_num)

        if self.train_dum is not None:
            dum_scaler = RobustScaler(unit_variance=True)
            self.train_dum[self.train_dum.columns] = dum_scaler.fit_transform(self.train_dum)

        return

    def set_cv(self, n_folds : int = 5) -> None:
        
        if self.cv is not None and \
            len(self.cv['Fold'].unique()) == n_folds:
            return self.cv

        self.cv = pd.DataFrame(index=self.train.index)
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (_, valid_idx) in \
            enumerate(gkf.split(self.X, self.y, self.train.index)):
            self.cv.loc[valid_idx, 'Fold'] = fold

        self.n_folds = n_folds
        collect() # collect garbage
        return
    
    def get_fold(self, fold : int, dummies : bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.n_folds is None:
            print("Warning: CV not set. Setting CV with 5 folds.")
            self.set_cv()

        assert fold < self.n_folds, "Not a valid fold"

        valid_idx = self.cv['Fold'] == fold
        train_idx = self.cv['Fold'] != fold

        if dummies:
            X = pd.concat([self.train_num, self.train_dum], axis=1)
        else:
            X = pd.concat([self.train_num, self.train_cat], axis=1)

        return (X[train_idx], self.y.loc[train_idx],
                X[valid_idx], self.y.loc[valid_idx])
    
    def get_train(self, preprocessed : bool = True, dummies : bool = False):

        if not preprocessed:
            return self.train
        
        if dummies:
            X = pd.concat([self.train_num, self.train_dum], axis=1)
        else:
            X = pd.concat([self.train_num, self.train_cat], axis=1)

        return (X, self.y)
    
    def get_test(self, preprocessed : bool = True, dummies : bool = False):

        if not preprocessed:
            return self.test
        
        if dummies:
            X = pd.concat([self.test_num, self.test_dum], axis=1)
        else:
            X = pd.concat([self.test_num, self.test_cat], axis=1)

        return X
    
### Preprocess all at once

    def preprocess(self, n_folds: int = 5, pca_components: Union[int, None] = None,
                   n_clusters: int = 8, force_log: bool = False, make_dummies : bool = True,
                   scale : bool = True):
        
        self.all_additions(n_components=pca_components, n_clusters=n_clusters, force_log=force_log)
        self.set_cv(n_folds=n_folds)
        if make_dummies:
            self.make_dummies()
        if scale:
            self.scale()
        return
    
    def get_columns(self, dummies : bool = False):
        if dummies:
            assert self.train_dum is not None, "Dummies not yet created"
            return [*self.train_num.columns, *self.train_dum.columns]
        return [*self.train_num.columns, *self.train_cat.columns]

if __name__ == "__main__":
    df = pd.read_csv('./data/s4e5/train.csv')
    test = pd.read_csv('./data/s4e5/test.csv')

    prep = DataProcessor(df, test_data=test, primary_column='id')
    prep.datatypes()
    prep.missing_values()
    #prep.correlation()

    prep.set_cv(n_folds=5)
    print(prep.cv)

    prep.get_columns()
    #prep.unique_values('Sex')
    
    #prep.compare('Whole weight.1', 'Whole weight.2')
    prep.cluster_analysis(n_clusters=2)
    prep.pca_analysis()

    prep.add_logs()
    prep.add_clusters()
    prep.add_pca_components(n_components=4)

    #prep.distribution('Whole weight.2')

    prep.preprocess()

    print(prep.train_num)

    import xgboost as xgb
    model = xgb.XGBRegressor().fit(prep.train_num, prep.y)
    print(model.score(prep.train_num, prep.y))
    print(model.feature_importances_)
