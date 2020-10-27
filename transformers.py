from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import skew
#from typing import final requires python 3.8
from scipy.special import boxcox1p, boxcox
import pandas as pd
import logging
import config


logger = logging.getLogger(__name__)

class SkippableTransformer(ABC, BaseEstimator, TransformerMixin):

    def __init__(self, skip = False):
        self.skip = skip
    
    def do_fit(self, X, y = None):
        pass

    def do_transform(self, X, y = None):
        pass

    #@final
    def fit(self, X, y = None):
        if not self.skip:
            self.do_fit(X, y)
        return self 
    
    #@final
    def transform(self, X, y = None):
        if not self.skip:
            return self.do_transform(X, y)
        return X

    def get_params(self, deep = True):
        params = super().get_params(deep)
        params['skip'] = self.skip
        return params

class FeatureDropTransformer(SkippableTransformer):

    def __init__(self, droppable_features, skip = False):
        super().__init__(skip)
        self.droppable_features = droppable_features
    
    def do_fit(self, X, y = None):
        return self 
    
    def do_transform(self, X, y = None):
        return X.drop(self.droppable_features, axis = 1)

    def get_params(self, deep = True):
        params = super().get_params(deep)
        params['droppable_features'] = self.droppable_features
        return params

class ClipTransformer(SkippableTransformer):

    def __init__(self, col_to_clip, skip = False):
        super().__init__(skip)
        self.col_to_clip = col_to_clip

    def do_fit(self, X, y = None):
        return self 
    
    def do_transform(self, X, y = None):
        df = X.copy()
        for col, (clip_lower, clip_upper) in self.col_to_clip.items():
            df[col] = df[col].clip(clip_lower, clip_upper)

        return df

    def get_params(self, deep = True):
        params = super().get_params(deep)
        params['col_to_clip'] = self.col_to_clip
        return params

class FixTypoTransformer(SkippableTransformer):

    def __init__(self, col_to_typos, skip = False):
        super().__init__(skip)
        self.col_to_typos = col_to_typos
    
    def do_fit(self, X, y = None):
        return self 
    
    def do_transform(self, X, y = None):
        df = X.copy()
        for col, typo_mapping in self.col_to_typos.items():
            mapping = {value:typo_mapping.get(value) or value for value in df[col].unique()}
            df[col] = df[col].map(mapping)

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['col_to_typos'] = self.col_to_typos
        return params

class FillNATransformer(SkippableTransformer):

    def __init__(self, na_features, fill_value, skip = False):
        super().__init__(skip)
        self.na_features = na_features
        self.fill_value = fill_value
    
    def do_fit(self, X, y = None):
        return self 
    
    def do_transform(self, X, y = None):
        df = X.copy()
        for col in self.na_features:
            df[col] = df[col].fillna(self.fill_value)

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['na_features'] = self.na_features
        params['fill_value'] = self.fill_value
        return params

class FillNAWithModeTransformer(SkippableTransformer):

    def __init__(self, na_features, skip = False):
        super().__init__(skip)
        self.na_features = na_features
    
    def do_fit(self, X, y = None):
        self.col_to_mode = {col:X[col].mode()[0] for col in self.na_features}
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        for col in self.na_features:
            df[col] = df[col].fillna(self.col_to_mode[col])

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['na_features'] = self.na_features
        return params

class TypeTransformer(SkippableTransformer):

    def __init__(self, feature_to_type, skip = False):
        super().__init__(skip)
        self.feature_to_type = feature_to_type
    
    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        for col, typ in self.feature_to_type.items():
            df[col] = df[col].astype(typ)

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['feature_to_type'] = self.feature_to_type
        return params

class OrdinalEncoderTransformer(SkippableTransformer):

    def __init__(self, encoded_features_to_orders, skip = False):
        super().__init__(skip)
        self.encoded_features_to_orders = encoded_features_to_orders
        self.encoded_features_to_mappings = dict()
        for col, ordered_list in encoded_features_to_orders.items():
            self.encoded_features_to_mappings[col] = {k:v for k,v in zip(ordered_list, range(len(ordered_list)))}

    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        for col, mapping in self.encoded_features_to_mappings.items():
            if any([x not in mapping.keys() for x in df[col].unique()]):
                error = f"Not every unique value |{str(df[col].unique())}| in the column: |{col}| has a mapping in: |{str(mapping.keys())}|"
                logger.error(error)
                raise ValueError(error)
            df[col] = df[col].map(mapping)

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['encoded_features_to_orders'] = self.encoded_features_to_orders
        return params


class LotFrontageTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        None

    def do_fit(self, X, y = None):
        self.neighbourhood_lotfrontage_medians = X.groupby('Neighborhood')['LotFrontage'].median()
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        lotfrontage_fills = df.apply(axis = 1, func = lambda row: self.neighbourhood_lotfrontage_medians.get(row['Neighborhood']) or 
                                                                        self.neighbourhood_lotfrontage_medians.values.mean())
        df['LotFrontage'] = df['LotFrontage'].fillna(lotfrontage_fills)
        return df


class TotalSFTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        None

    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        return df

class SkewnessTransformer(SkippableTransformer):

    def __init__(self, transform_treshold, lamda = 0.15, skip = False): # the B is silent due to lambda being a keyword in python
        super().__init__(skip)
        self.transform_treshold = transform_treshold
        self.lamda = lamda

    def do_fit(self, X, y = None):
        numeric_cols = X.dtypes[X.dtypes != 'object'].index
        col_skewness = X[numeric_cols].apply(lambda series: skew(series.dropna())).sort_values(ascending=False)
        self.skewness_df = pd.DataFrame({'Skew': col_skewness})
        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        transformable = self.skewness_df[abs(self.skewness_df.Skew) > self.transform_treshold].index
        logger.debug("There are |%d| skewed numerical features to Box Cox transform. These are: |%s|", transformable.shape[0], str(transformable))
        for col in transformable:
            df[col] = boxcox1p(df[col], self.lamda) 

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['transform_treshold'] = self.transform_treshold
        params['lamda'] = self.lamda
        return params

class VarianceThresholdTransformer(SkippableTransformer):

    def __init__(self, keep_treshold, skip = False):
        super().__init__(skip)
        self.keep_treshold = keep_treshold

    def do_fit(self, X, y = None):
        variances = X.var(axis = 0)
        self.droppable_cols = variances.index[variances < self.keep_treshold]
        logger.debug('Will drop |%d| columns due to having varience below keep_treshold of |%f|. Column names: |%s|',
            len(self.droppable_cols), self.keep_treshold, self.droppable_cols)
        return self

    def do_transform(self, X, y = None):
        return X.drop(self.droppable_cols, axis = 1)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['keep_treshold'] = self.keep_treshold
        return params

class OneHotEncodingTransformer(SkippableTransformer):

    def __init__(self, categorical_features, drop_strategy, skip = False):
        super().__init__(skip)
        self.feature_to_encoder = dict()
        self.categorical_features = categorical_features
        self.drop_strategy = drop_strategy

    def do_fit(self, X, y = None):
        for col, possible_values in self.categorical_features.items():
            encoder = OneHotEncoder(categories = [possible_values], drop = self.drop_strategy)
            encoder.fit(X[col].values.reshape(-1, 1))
            self.feature_to_encoder[col] = encoder 

        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        for col in self.categorical_features.keys():
            added_cols = self.feature_to_encoder[col].transform(df[col].values.reshape(-1, 1))
            added_cols = added_cols.toarray()
            for added_cat, col_index in zip(self.feature_to_encoder[col].categories_[0], range(added_cols.shape[1])):
                df[col + '_' + added_cat] = added_cols[:, col_index]
            df = df.drop(col, axis = 1)

        return df

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['categorical_features'] = self.categorical_features
        params['drop_strategy'] = self.drop_strategy
        return params

class StandardizationTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        self.feature_to_scaler = dict()

    def do_fit(self, X, y = None):
        numeric_cols = X.dtypes[X.dtypes != 'object'].index
        logger.debug("Columns that will get standardized: |%s|", numeric_cols)
        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(X[col].values.reshape(-1, 1))
            self.feature_to_scaler[col] = scaler

        return self

    def do_transform(self, X, y = None):
        df = X.copy()
        for col in self.feature_to_scaler.keys():
            df[col] = self.feature_to_scaler[col].transform(X[col].values.reshape(-1, 1)).flatten()

        return df

class CheckNoMoreMissingValuesTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        None

    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        missing_per_column = X.isnull().sum()
        if missing_per_column.sum() > 0:
            logger.error("There are still missing values! See: |%s|", str(missing_per_column))
            raise ValueError(f"There are still missing values! See: |{str(missing_per_column)}|")

        return X

class DiagnosticTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        None

    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        logger.debug("Shape of X: |%s|", X.shape)
        logger.debug("Columns of X: |%s|", X.columns.to_list())
        return X


class PandasToNumpyTransformer(SkippableTransformer):

    def __init__(self, skip = False):
        super().__init__(skip)
        None

    def do_fit(self, X, y = None):
        return self

    def do_transform(self, X, y = None):
        return X.values
