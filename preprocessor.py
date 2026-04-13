from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



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
        x[self.num_cols] = x[self.num_cols].fillna(self.medians)

        return x





class QualityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ORDINAL_COLUMNS):
        self.ORDINAL_COLUMNS = ORDINAL_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.ORDINAL_COLUMNS.items():
            if col in X.columns:
                X[col] = X[col].fillna("NA").map(mapping)
        return X

class NAFiller(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        num_cols = X.select_dtypes(exclude="object").columns
        cat_cols = X.select_dtypes(include="object").columns

        self.num_medians_ = X[num_cols].median()
        self.cat_modes_ = {col: X[col].mode()[0] for col in cat_cols if X[col].notna().any()}


        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_medians_.index] = X[self.num_medians_.index].fillna(self.num_medians_)


        for col, mode in self.cat_modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mode)

        return X

class FeatureAdder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalSF"] = X["TotalBsmtSF"]+ X["1stFlrSF"]+X["2ndFlrSF"]
        X["TotalPorchSF"] = X["OpenPorchSF"]+ X["EnclosedPorch"] +X["3SsnPorch"]+ X["ScreenPorch"]
        X["TotalBathrooms"] = X["FullBath"] + 0.5 * X["HalfBath"]+ X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]

        X["WasRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)

        X["QualityXArea"] = X["OverallQual"] * X["GrLivArea"]

        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)

        return X

class WOEEncoder(BaseEstimator, TransformerMixin):


    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        y_binary = (y > y.median()).astype(int)
        self.woe_maps_ = {}

        for col in self.columns:
            if col not in X.columns:
                continue

            df = pd.DataFrame({"cat": X[col].values, "target": y_binary.values})
            g = df.groupby("cat")["target"].agg(["count", "sum"])
            g.columns = ["n_total", "n_pos"]
            g["n_neg"] = g["n_total"] - g["n_pos"]

            total_pos = max(g["n_pos"].sum(), 1)
            total_neg = max(g["n_neg"].sum(), 1)

            g["pct_pos"] = (g["n_pos"] / total_pos).clip(lower=1e-4)
            g["pct_neg"] = (g["n_neg"] / total_neg).clip(lower=1e-4)
            g["woe"] = np.log(g["pct_pos"] / g["pct_neg"])

            self.woe_maps_[col] = g["woe"].to_dict()

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue

            X[col + "_woe"] = X[col].map(self.woe_maps_[col]).fillna(0)
            X = X.drop(columns=[col])
        return X

class OneHotEncoderSafe(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include="object").columns.tolist()
        dummies = pd.get_dummies(X[self.cat_cols_], drop_first=True, dtype=int)
        self.dummy_cols_ = dummies.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        dummies = pd.get_dummies(X[self.cat_cols_], drop_first=True, dtype=int)
        dummies = dummies.reindex(columns=self.dummy_cols_, fill_value=0)
        X = X.drop(columns=self.cat_cols_)
        return pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


