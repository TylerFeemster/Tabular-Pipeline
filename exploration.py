import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator, title, align_integer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from typing import Union

class Explorer:
    def __init__(self, data: pd.DataFrame,
                 target: Union[str, list, None] = None,
                 primary_column: Union[str, None] = None) -> None:
        '''

        Args:
            train: dataframe for exploration
            targets: prediction columns
        '''
        # verify target(s) are in dataframe columns
        if target is not None:
            assert set(target) <= set(data.columns)
            
            if type(target) is str:
                target = [target]
        
        if primary_column is not None:
            # verify primary key is a dataframe column
            assert primary_column in data.columns
            self.data = data.set_index(primary_column)
        else:
            self.data = data.copy()

        self.targets = target
        self.cols = self.data.columns

        drop_cols = []
        if self.targets is not None:
            drop_cols = self.targets
            self.y = self.data[self.targets]
        
        self.X = self.data.drop(columns=drop_cols)
        self.X_cols = self.X.columns

        self.categ_cols = self.X.select_dtypes(include=['object']).columns
        self.numer_cols = self.X.select_dtypes(include=['number']).columns

        self.disc_cols = self.X.select_dtypes(include=['int']).columns
        self.cont_cols = self.X.select_dtypes(include=['float']).columns
        
        stats = self.X[self.numer_cols].describe().T[['mean', 'std']]
        self.X_norm = (self.X[self.numer_cols] - stats['mean']) / stats['std']


### Basic Methods

    def describe(self) -> None:
        title('Basic Statistics')
        print(self.data.describe().T)
        return

    def datatypes(self, display : bool = True) -> dict:
        '''
        
        Args:
            display: whether or not to print to terminal
        '''
        if display:
            title('Data Types of Variables:')
            print(self.data.dtypes)

        return self.data.dtypes.to_dict()

    def missing_values(self, display : bool = True) -> Union[dict, None]:
        '''

        Args:
            display: whether or not to print to terminal
        '''
        if display:
            title('Checking Missing Values...')

        dct = {}
        no_missing_values = True
        for col in self.cols:
            na_count = self.data[col].isna().sum()
            if na_count != 0:
                no_missing_values = False
                dct[col] = na_count

        if no_missing_values:
            if display: 
                print('No missing values found in dataset.')
            return None
        
        # Code below executed only when there is missing data.

        if display:
            print('Missing values in dataset by feature: ')
            for key in dct.keys():
                print(f'{key} : {dct[key]}')

        return dct        

### Column Exploration

    def get_columns(self, display : bool = True):
        if display:
            title('Columns')
            for col in self.data.columns:
                print(col)
        return self.data.columns
    
    def unique_values(self, column : str, display : bool = True):
        unique_vals = np.sort(self.data[column].unique())

        if display:
            title(f'Unique values of {column}')
            print(unique_vals)

        return unique_vals
    
    def histogram(self, column : str):
        title(f'Histogram for {column}')
        sns.displot(self.data, x=column, kind='hist')
        plt.show()
        return
    
    def ecdf(self, column: str):
        title(f'ECDF-Box Visual for {column}')
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8),
                                     gridspec_kw={'height_ratios': [2, 1]})

        sns.ecdfplot(self.data, x=column, stat='proportion', ax=ax1)
        ax1.set_title("ECDF Plot")
        ax1.set_xlabel("")

        sns.boxplot(self.data, x=column, ax=ax2)
        ax2.set_title("Box Plot")

        plt.subplots_adjust(hspace=0.3)

        ax1.set_yticks(np.linspace(0, 1, 11))

        plt.show()
        return

### Unsupervised Learning

    def pca_analysis(self, n_components : Union[int, None] = None, 
                     display : bool = True) -> PCA:
        
        if n_components is not None and n_components >= len(self.X_norm.columns):
            n_components = None
        
        if display:
            if n_components is None:
                title('Performing pca analysis with all components...')
            else:
                title('Performing pca analysis for top {n_components} components...')

        params = {'n_components' : n_components,
                  'random_state' : 0}
        pca = PCA(**params).fit(self.X_norm)

        if display:
            print('Ratio of explained variance:')
            separator(symbol='-')
            total = pca.n_components_
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                n = align_integer(i, total)
                print(f'Component {n}: {ratio:.4f}')

        return pca

    def cluster_analysis(self, n_clusters : int = 8, 
                         display : bool = True) -> KMeans:
        if display:
            title(f'Performing cluster analysis with {n_clusters} clusters...')

        # since mean of X_norm is 0, this is inertia for clusters = 1
        base_inertia = (self.X_norm.to_numpy()**2).sum() # both axes summed

        params = {'n_clusters' : n_clusters,
                  'random_state' : 0}
        kmeans = KMeans(**params).fit(self.X_norm)

        if display:
            print(f'Base Inertia (one cluster): {base_inertia:.3f}')
            print(f'Inertia with {n_clusters} clusters : {kmeans.inertia_:.3f}')
            print(f'Ratio : {base_inertia / kmeans.inertia_:.3f}')

        return kmeans


    def distribution(self, col, discrete=False):
        assert col in self.cols
        sns.displot(self.data, x=col, discrete=discrete)
        plt.title(f'Distribution of {col}')
        plt.show()

    # TODO: Fix this; maybe add choices manually
    def compare(self, col1, col2) -> None:
        assert col1 in self.cols and col2 in self.cols

        # storing boolean values for compact code
        numeric = (col1 in self.cont_cols, 
                   col2 in self.cont_cols)

        if numeric[0] and numeric[1]: # both continuous
            sns.displot(self.data, x=col1, y=col2)
        elif numeric[0]: # col1 continuous, col2 not
            sns.displot(self.data, x=col2, y=col1, kind="kde")
        elif numeric[1]: # col1 not, col2 continuous
            sns.displot(self.data, x=col1, y=col2, kind="kde")
        else: # fully categorical or integral
            sns.displot(self.data, x=col1, y=col2, kind="hist")

        plt.title(f'Comparing {col2} with {col1}')    
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./data/s4e5/train.csv')

    prep = Explorer(df, primary_column='id')
    #prep.describe()
    #prep.datatypes()
    #prep.missing_values()
    
    #prep.get_columns()
    #prep.histogram('FloodProbability')
    #prep.unique_values('InadequatePlanning')
    
    #prep.compare('InadequatePlanning', 'FloodProbability')
    #prep.cluster_analysis(n_clusters=10)
    #prep.pca_analysis()

    prep.ecdf('InadequatePlanning')
