import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator

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

        # verify primary key is a dataframe column
        if primary_column is not None:
            assert primary_column in data.columns

        self.targets = target
        self.data = data.set_index(primary_column)
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
        pass

    def datatypes(self, display=True) -> dict:
        '''
        
        Args:
            display: whether or not to print to terminal
        '''
        if display:
            separator()
            print('Data Types of Variables:')
            separator(symbol='-')
            print(self.data.dtypes)

        return self.data.dtypes.to_dict()

    def missing_values(self, display=True) -> Union[dict, None]:
        '''

        Args:
            display: whether or not to print to terminal
        '''
        if display:
            separator()
            print('Checking Missing Values...')
            separator(symbol='-')

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
            print('Total missing values in training AND testing set: ')
            for key in dct.keys():
                print(f'{key} : {dct[key]}')

        return dct        
    
    def distribution(self, col, discrete=False):
        assert col in self.cols
        sns.displot(self.data, x=col, discrete=discrete)
        plt.title(f'Distribution of {col}')
        plt.show()

    def describe(self) -> None:
        separator()
        print('Numerical Data')
        print(self.X.describe().T)

        separator(symbol='-')
        print('Categorial Data')

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
    #prep.missing_values()
    #prep.datatypes()
    #prep.describe()
    prep.compare('InadequatePlanning', 'FloodProbability')
