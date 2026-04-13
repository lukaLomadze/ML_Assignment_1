from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#


class Preprocessor(Protocol):
    def __init__(self):
        pass
    def fit(self, X, y=None):
       ...
    def transform(self, X, y=None):
       ...

class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_cols = []
        self.num_cols = []
        self.medians = []

    def fit(self, X, y=None):
        x= X.copy()
        self.cat_cols = x.select_dtypes(include='object').columns
        self.num_cols = x.select_dtypes(exclude='object').columns
        self.medians=x[self.num_cols].median()
        return self
    def transform(self, X, y=None):
        x= X.copy()
        x= x.drop(columns=self.cat_cols)
        # print(self.cat_cols)
        # print(self.num_cols)


        x[self.num_cols] = x[self.num_cols].fillna(self.medians)

        return x

