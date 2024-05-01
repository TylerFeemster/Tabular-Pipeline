import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separator

from typing import Union

class Preprocessing:
    def __init__(self, train : pd.DataFrame, 
                 test : Union[pd.DataFrame, None] = None,
                 targets: Union[list, None] = None) -> None :
        '''

        Args:
            train: train dataframe for train set
            test: pandas dataframe for test set
            targets: column names of targets in train
        '''
        
        self.test_provided = test is not None
        self.targets_provided = targets is not None

        self.train_df = train.copy()
        self.train_cols = self.train_df.columns

        if self.test_provided:
            self.test_df = test.copy()
            self.test_cols = self.test_df.columns

        pass

    def datatypes(self) -> dict :
        '''
        
        Args:
            None
        '''
        separator()
        print('Data Types of Variables:')
        separator(symbol='-')
        print(self.train_df.dtypes)
        return self.train_df.dtypes.to_dict()

    def missing_values(self) -> dict :
        '''

        Args:
            None
        '''
        separator()
        print('Checking Missing Values...')
        separator(symbol='-')

        dct = {}
        no_missing_values = True
        for col in self.train_cols:
            na_count = self.train_df[col].isna().sum()
            if na_count != 0:
                no_missing_values = False
                dct[col] = na_count

        if self.test_provided:
            for col in self.test_cols:
                na_count = self.test_df[col].isna().sum()
                if na_count != 0:
                    no_missing_values = False
                    if col in dct.keys():
                        dct[col] += na_count
                    else:
                        dct[col] = na_count

        if no_missing_values:
            print('No missing values found in train or test sets.')
        else:
            print('Total missing values in training AND testing set: ')
            for key in dct.keys():
                print(f'{key} : {dct[key]}')

        return dct

    def view_distributions(self):
        for col in self.train_cols:
            sns.displot(self.train_df, x=col, kde=True)
            plt.show()

    def describe(self) -> None:
        separator()
        print('Train Data')

        separator(symbol='-')
        print('Numerical Data')
        print(self.train_df.describe().T)

        separator(symbol='-')
        print('Categorial Data')

        print("TODO")

        if self.test_provided:
            print(self.test_df.describe().T)
        
            
if __name__ == "__main__":
    train_df = pd.read_csv('./data/s4e5/train.csv')
    test_df = pd.read_csv('./data/s4e5/test.csv')

    prep = Preprocessing(train_df, test_df)
    prep.missing_values()
    prep.datatypes()
    prep.describe()