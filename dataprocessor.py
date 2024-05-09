import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator, title, subtitle, align_integer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from explorer import Explorer

from gc import collect
from typing import Union

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
            self.train = train_data.set_index(primary_column)
        else:
            self.train = train_data.copy()

        self.targets = []
        if target:
            if type(target) is list:
                self.targets = target.copy()
            else:
                self.targets = [target]
        
        if test_data is not None:
            if primary_column:
                assert primary_column in test_data.columns
                self.test = test_data.set_index(primary_column)
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

        self.cols = self.train.columns

        self.cat_cols = self.X.select_dtypes(include=['object']).columns
        self.num_cols = self.X.select_dtypes(include=['number']).columns

        stats = self.X[self.num_cols].describe().T[['mean', 'std']]
        self.X_norm = (self.X[self.num_cols] - stats['mean']) / stats['std']

        # Explorer
        super().__init__(train_data, targets=self.targets)

        # CV Scheme
        self.cv = None
        self.n_folds = None

### Preprocessing

    def get_cv(self, n_folds : int = 5) -> pd.DataFrame:
        
        if self.cv is not None and \
            len(self.cv['fold'].unique()) == n_folds:
            return self.cv

        self.cv = pd.DataFrame(index=self.train.index)
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (_, valid_idx) in \
            enumerate(gkf.split(self.X, self.y, self.train.index)):
            self.cv.loc[valid_idx, 'Fold'] = fold

        self.n_folds = n_folds

        collect() # collect garbage
        return self.cv

if __name__ == "__main__":
    df = pd.read_csv('./data/s4e4/train.csv')
    test = pd.read_csv('./data/s4e4/test.csv')

    prep = DataProcessor(df, test_data=test, primary_column='id')
    prep.datatypes()
    prep.missing_values()
    prep.correlation()

    help(prep.missing_values)

    prep.get_cv(n_folds=5)
    print(prep.cv)

    prep.get_columns()
    prep.unique_values('Sex')
    
    prep.compare('Whole weight.1', 'Whole weight.2')
    prep.cluster_analysis(n_clusters=2)
    prep.pca_analysis()

    prep.distribution('Whole weight.2')
