import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import title, subtitle, align_integer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from typing import Union

class Explorer:

    def __init__(self, data: pd.DataFrame,
                 classify: bool = False,
                 target: Union[str, list] = []) -> None:
        '''
        This class contains several common methods for exploring a dataframe 
        without mutating it.

        Arguments:
            data: dataframe for exploration
            classify: classification when True, regression when False
            target: prediction columns
        '''

        self.data = data # since all methods are passive, no copy
        self.classify = classify

        self.target = []
        if target:
            if type(target) is str:
                target = [target]

            # verify target(s) are in dataframe columns
            assert set(target) <= set(data.columns)
            self.target = target.copy()

        self.X = self.data.drop(columns=self.target)
        self.X_cols = self.X.columns

        self.cols = self.data.columns

        self.cat_cols = self.X.select_dtypes(include=['object']).columns
        self.num_cols = self.X.select_dtypes(include=['number']).columns

        self.int_cols = self.X.select_dtypes(include=['int']).columns
        self.flt_cols = self.X.select_dtypes(include=['float']).columns
        
        stats = self.X[self.num_cols].describe().T[['mean', 'std']]
        self.X_norm = (self.X[self.num_cols] - stats['mean']) / stats['std']


# Basic Exploration

    def datatypes(self, display: bool = True) -> dict:
        '''
        Returns a dictionary giving the datatype of each column in the object's 
        train dataframe. It also prints this info by default.

        Arguments:
            display: whether or not to print
        '''
        if display:
            title('Data Types of Variables:')
            print(self.data.dtypes)

        return self.data.dtypes.to_dict()

    def missing_values(self, display: bool = True) -> dict:
        '''
        Returns a dictionary whose keys are the columns with missing values and 
        whose values are the number of missing values of the associated column.
        It also prints this info by default.

        Arguments:
            display: whether or not to print
        '''
        if display:
            title('Checking missing values...')

        problem_cols = dict()
        no_missing_values = True
        for col in self.cols:
            na_count = self.data[col].isna().sum()
            if na_count != 0:
                no_missing_values = False
                problem_cols[col] = na_count

        if display:
            if no_missing_values:
                print('No missing values found in dataset.')
            else:
                print('Missing values in dataset by feature: ')
                for key, val in problem_cols.items():
                    print(f'{key} : {val}')

        return problem_cols

# Column Exploration

    def get_columns(self, display: bool = True) -> list:
        '''
        Returns list of features/columns. It also prints this info by default.

        Arguments:
            display: whether or not to print
        '''
        if display:
            title('Columns')
            for col in self.data.columns:
                print(col)
        return self.data.columns

    def unique_values(self, column: Union[str, None] = None, display: bool = True) -> dict:
        '''
        By default, returns dictionary whose key-value pairs are categorical columns and lists
        of their respective unique values. These values are also sorted. When column is provided,
        the dictionary has the key-value pair for this column only. It also prints this info by
        default.

        Arguments:
            column: the column whose unique values we want
            display: whether or not to print
        '''

        if column:
            assert column in self.cols, f"{column} is not a column of dataframe."

            unique_vals = {column : sorted(self.data[column].unique())} # make it nice with sort

            if display:
                title(f'Unique values of {column}')
                print(unique_vals[column])

            return unique_vals

        # code below executed when column is None

        if display:
            title(f'Unique Values')

        unique_vals = dict()
        for col in self.cat_cols:
            unique_vals[col] = sorted(self.data[col].unique())
            if display:
                print(f'{col} : {unique_vals[col]}')


        return unique_vals

    def distribution(self, column: str) -> None:
        '''
        Provides a stacked plot for easy visual analysis of the given column's distribution. 
        The three plots, top to bottom, are a histogram, a ecdf plot, and a box plot.

        Arguments:
            column: the column whose distribution we want to analyze
        '''
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

        plt.subplots_adjust(hspace=0)  # no spacing
        plt.show()
        return

# Global Exploration

    def correlation(self):
        '''
        Provides heatmap for correlation matrix of numerical columns.

        Arguments:
            None
        '''
        title('Correlation Heat Map')
        sns.heatmap(self.data[self.num_cols].corr())
        return

# Unsupervised Learning

    def pca_analysis(self, n_components: Union[int, None] = None,
                     display: bool = True) -> PCA:
        '''
        Performs principal component analysis, returning PCA object. By default,
        the function prints the fraction of explained variance for each component.

        Arguments:
            n_components: number of components; None means all
            display: whether to print info or not
        '''

        if n_components and n_components >= len(self.X_norm.columns):
            n_components = None

        if display:
            if n_components:
                title(
                    'PCA analysis for top {n_components} components')
            else:
                title('Full PCA analysis')

        pca = PCA(n_components=n_components).fit(self.X_norm)

        if display:
            subtitle('Fraction of explained variance:')
            total = pca.n_components_
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                n = align_integer(i, total)
                print(f'Component {n}: {ratio:.4f}')

        return pca

    def cluster_analysis(self, n_clusters: int = 8,
                         display: bool = True) -> KMeans:
        '''
        Performs k-Means clustering with \'n_clusters\' clusters and returns KMeans
        object. By default, the function prints base inertia, new inertia, and the 
        ratio to help with any analyses.

        Arguments:
            n_clusters: number of clusters
            display: whether or not to print info
        '''
        
        if display:
            title(f'Performing cluster analysis with {n_clusters} clusters...')

        # since mean of X_norm is 0, this is inertia for clusters = 1
        base_inertia = (self.X_norm.to_numpy()**2).sum()  # both axes summed

        params = {'n_clusters': n_clusters,
                  'random_state': 0}
        kmeans = KMeans(**params).fit(self.X_norm)

        if display:
            print(f'Base Inertia (one cluster): {base_inertia:.3f}')
            print(
                f'Inertia with {n_clusters} clusters : {kmeans.inertia_:.3f}')
            print(f'Ratio : {base_inertia / kmeans.inertia_:.3f}')

        return kmeans
    
    def compare(self, col1 : str, col2 : str) -> None:
        '''
        Provides a visualization showing the relationship between column \'col1\'
        and column \'col2\'. The function is aware of the nature of the columns:
        categorical, integral, or float. If both are categorical, it provides a 
        countplot; if only one is categorical, separate histograms are overlaid 
        across the category showing the distribution of the other column. If both
        are numerical, a 2D histplot is given; for integral data, the function
        sets the binwidth to 1 for naturalness.

        Arguments:
            col1: first column
            col2: second column
        '''
        assert col1 in self.cols and col2 in self.cols

        # storing boolean values for compact code
        cat1 = col1 in self.cat_cols
        cat2 = col2 in self.cat_cols

        flt1 = col1 in self.flt_cols
        flt2 = col2 in self.flt_cols

        if cat1 or cat2:
            if cat1 and cat2:
                sns.countplot(self.data, x=col1, y=col2)
            elif cat1:
                if flt2:
                    sns.histplot(self.data, x=col2, hue=col1)
                else: # integral data
                    sns.histplot(self.data, x=col2, hue=col1, binwidth=1)
            else: # cat2
                if flt1:
                    sns.histplot(self.data, x=col1, hue=col2)
                else: # integral data
                    sns.histplot(self.data, x=col1, hue=col2, binwidth=1)

        else: # here, there's only numerical variables
            if flt1 or flt2:
                if flt1 and flt2:
                    sns.histplot(self.data, x=col1, y=col2)
                elif flt1:
                    width = max(self.data[col1]) - min(self.data[col1])
                    sns.histplot(self.data, x=col1, y=col2, binwidth=(width/100, 1))
                else: # flt2
                    width = max(self.data[col2]) - min(self.data[col2])
                    sns.histplot(self.data, x=col1, y=col2, binwidth=(1, width/100))

            else:
                sns.histplot(self.data, x=col1, y=col2, binwidth=(1, 1))


        plt.title(f'{col2} - {col1} Relationship')
        plt.show()
        return
    
    def mutual_info(self, column: str, display: bool = True) -> dict:
        '''
        Calculates mutual information between given column and all numeric
        columns.

        Arguments:
            column: feature with which to calculate each mutual information
            display: whether or not to print info
        '''

        if display:
            title(f'Mutual Info with {column}')

        numeric = self.data.select_dtypes(include=['number'])
        X = numeric.drop(columns=[column])
        y = numeric[column]

        if self.classify:
            mutual_info = mutual_info_classif(X, y)
        else:
            mutual_info = mutual_info_regression(X, y)

        mi_map = dict()
        for mi, col in zip(mutual_info, X.columns):
            mi_map[col] = mi
            if display:
                print(f'{col} ... {mi : .4f}')

    def target_mutual_info(self, display : bool = True) -> None:
        '''
        Calculate mutual information of each column relative to each target.

        Arguments:
            display: whether or not to print info
        '''
        for target in self.target:
            self.mutual_info(target, display=display)

        return

if __name__ == "__main__":
    df = pd.read_csv('./data/s4e4/train.csv')

    prep = Explorer(df)
    prep.datatypes()
    prep.missing_values()
    
    prep.get_columns()
    prep.unique_values('Sex')
    prep.unique_values()
    
    prep.compare('Whole weight.1', 'Whole weight.2')
    Explorer(pd.read_csv('./data/s4e5/train.csv')).compare('FloodProbability', 'PoliticalFactors')
    prep.cluster_analysis(n_clusters=2)
    prep.pca_analysis()

    prep.mutual_info('Whole weight.1')

    #prep.distribution('Whole weight.2')
