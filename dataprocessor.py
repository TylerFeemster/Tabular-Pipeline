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
        
        self.test = None
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

        self.X_new = self.X.copy()

        self.cols = self.train.columns

        self.cat_cols = self.X.select_dtypes(include=['object']).columns
        self.num_cols = self.X.select_dtypes(include=['number']).columns

        # Unsupervised: use all X available, train and test
        self.X_unsupervised = self.X[self.num_cols].copy()
        if self.test is not None:
            self.X_unsupervised = pd.concat([self.X_unsupervised, 
                                             self.test[self.num_cols]], 
                                             axis=0)
        stats = self.X_unsupervised.describe().T
        self.X_unsupervised = (self.X_unsupervised - stats['mean']) / stats['std']

        # Explorer
        super().__init__(train_data, targets=self.targets)

        # CV Scheme
        self.cv = None
        self.n_folds = None

### Adding Features

    def add_logs(self, force : bool = False) -> None:

        for col in self.num_cols:
            if min(self.X[col]) > 0:
                self.X_new[f'log {col}'] = np.log(self.X[col])
            
            elif force:
                minimum = min(self.X[col])
                self.X_new[f'log {col}'] = np.log(self.X[col] - minimum + 1)

    def add_pca_components(self, n_components : Union[int, None] = None) -> None:
        
        if n_components is None or n_components > len(self.X_unsupervised.columns):
            n_components = len(self.X_unsupervised.columns)

        pca = PCA(n_components=n_components).fit(self.X_unsupervised)
        self.X_new[[f'pca {i+1}' for i in range(n_components)]] = pca.transform(
            self.X_unsupervised.loc[self.X.index])

        return
    
    def add_clusters(self, n_clusters : int = 8) -> None:
        
        kmeans = KMeans(n_clusters=n_clusters).fit(self.X_unsupervised)
        self.X_new['Cluster'] = kmeans.predict(
            self.X_unsupervised.loc[self.X.index]).astype(str) # avoid meaningless numeric
        
        return

### Preprocessing

    def scale(self):

        scaler = RobustScaler(unit_variance=True)
        self.X_scaled = scaler.fit_transform(self.X_new)

        return


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
    
    #prep.compare('Whole weight.1', 'Whole weight.2')
    prep.cluster_analysis(n_clusters=2)
    prep.pca_analysis()

    prep.add_logs()
    prep.add_clusters()
    prep.add_pca_components(n_components=4)

    prep.distribution('Whole weight.2')

    print(prep.X_new)
