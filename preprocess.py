import logging
from transformers import *
from sklearn.pipeline import Pipeline
import config

logger = logging.getLogger(__name__)


def _outlier_drop(df):
    return df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

def _target_transform(y):
    return np.log1p(y).values


def construct_preprocess_pipe():
    preprocessing_pipe = Pipeline(steps =   [   ('feature_drop', FeatureDropTransformer(config.DROP_FEATURES)),
                                                ('typo_fix', FixTypoTransformer(config.COLS_WITH_TYPOS)),
                                                ('none_fill', FillNATransformer(config.NONE_FEATURES, 'None')),
                                                ('zero_fill', FillNATransformer(config.ZERO_FEATURES, 0)),
                                                ('mode_fill', FillNAWithModeTransformer(config.MODE_FEATURES)),
                                                ('special_fill', FillNATransformer(['Functional'], 'Typ')),
                                                ('lotfrontage_fill', LotFrontageTransformer()),
                                                ('missing_check', CheckNoMoreMissingValuesTransformer()),
                                                ('type_transform', TypeTransformer(config.FEATURE_TO_TYPE)),
                                                ('totalsf_transform', TotalSFTransformer()),
                                                ('label_encoding', OrdinalEncoderTransformer(config.ORDINAL_FEATURES)),
                                                ('skew_fix', SkewnessTransformer()),
                                                ('get_dummies', OneHotEncodingTransformer(config.CATEGORICAL_FEATURES, config.drop_strategy)),
                                                ('pandas_to_numpy', PandasToNumpyTransformer())
                                            ] )
    
    return preprocessing_pipe


def prepare_data(df):
    df = _outlier_drop(df)
    X = df.drop('SalePrice', axis = 1)
    y = _target_transform(df['SalePrice'])
    
    return X, y