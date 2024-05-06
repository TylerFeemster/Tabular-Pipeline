import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator, title, subtitle, align_integer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from typing import Union

class Explorer:
    def __init__(self, data: pd.DataFrame,
                 targets: Union[str, list, None] = None,
                 primary_column: Union[str, None] = None) -> None:
        '''

        Args:
            train: dataframe for exploration
            targets: prediction columns
        '''

        if primary_column:
            # verify primary key is a dataframe column
            assert primary_column in data.columns
            self.data = data.set_index(primary_column)
        else:
            self.data = data.copy()
        
        drop_cols = []
        self.targets = []
        self.y = None
        if targets:
            if type(targets) is str:
                targets = [targets]

            # verify target(s) are in dataframe columns
            assert set(targets) <= set(data.columns)
            self.targets = targets.copy()

            drop_cols = self.targets
            self.y = self.data[self.targets]

        self.X = self.data.drop(columns=drop_cols)
        self.X_cols = self.X.columns

        self.cols = self.data.columns

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
            title('Checking missing values...')

        problem_cols = {}
        no_missing_values = True
        for col in self.cols:
            na_count = self.data[col].isna().sum()
            if na_count != 0:
                no_missing_values = False
                problem_cols[col] = na_count

        if no_missing_values:
            if display: 
                print('No missing values found in dataset.')
            return None
        
        # Code below executed only when there is missing data.

        if display:
            print('Missing values in dataset by feature: ')
            for key, val in problem_cols.items():
                print(f'{key} : {val}')

        return problem_cols

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
    
    def distribution(self, column : str):
        title(f'Distribution Visual for {column}')
        _, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 6),
                                          gridspec_kw={'height_ratios': [2, 2, 1]})
        
        # Histogram (top)
        sns.histplot(self.data, x=column, stat='proportion', ax=ax1)
        ax1.set_title(f"Distribution of {column}")

        # Empirical CDF (middle)
        sns.ecdfplot(self.data, x=column, stat='proportion', ax=ax2)
        ax2.set_yticks(np.arange(0, 1, 0.25))

        # Box Plot (bottom)
        sns.boxplot(self.data, x=column, ax=ax3)

        plt.subplots_adjust(hspace=0) # no spacing
        plt.show()
        return

### Unsupervised Learning

    def pca_analysis(self, n_components : Union[int, None] = None, 
                     display : bool = True) -> PCA:
        
        if n_components and n_components >= len(self.X_norm.columns):
            n_components = None
        
        if display:
            if n_components:
                title('Performing pca analysis for top {n_components} components...')
            else:
                title('Performing pca analysis with all components...')
                

        params = {'n_components' : n_components,
                  'random_state' : 0}
        pca = PCA(**params).fit(self.X_norm)

        if display:
            subtitle('Fraction of explained variance:')
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

    # TODO: Fix this; maybe add choices manually
    def compare(self, col1, col2) -> None:
        assert col1 in self.cols and col2 in self.cols

        # storing boolean values for compact code
        bool1 = col1 in self.cont_cols 
        bool2 = col2 in self.cont_cols

        if bool1 and bool2: # both continuous
            sns.displot(self.data, x=col1, y=col2)
        elif bool1: # col1 continuous, col2 not
            sns.displot(self.data, x=col2, y=col1, kind="kde")
        elif bool2: # col1 not, col2 continuous
            sns.displot(self.data, x=col1, y=col2, kind="kde")
        else: # fully categorical or integral
            sns.displot(self.data, x=col1, y=col2, kind="hist")

        plt.title(f'Comparing {col2} with {col1}')
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./data/s4e4/train.csv')

    prep = Explorer(df, primary_column='id')
    prep.describe()
    prep.datatypes()
    prep.missing_values()
    
    prep.get_columns()
    prep.unique_values('Sex')
    
    prep.compare('Whole weight.1', 'Whole weight.2')
    prep.cluster_analysis(n_clusters=2)
    prep.pca_analysis()

    prep.distribution('Whole weight.2')
