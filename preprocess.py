import logging
from transformers import *
from sklearn.pipeline import Pipeline
import config
from collections import Counter

logger = logging.getLogger(__name__)


def _outlier_drop(df):
    def detect_outliers(df, drop_treshold, features):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        outlier_indices = []
        
        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col],75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1
            
            # outlier step
            outlier_step = 1.5 * IQR
            
            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
            
            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)
            
        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
        multiple_outliers = [k for k, v in outlier_indices.items() if v > drop_treshold]
        
        return multiple_outliers   

    numeric_cols = df.loc[:,'MSSubClass':'SaleCondition'].select_dtypes(exclude=['object']).columns
    logger.debug('Detecting outliers in columns: |%s|', numeric_cols.to_list())
    outliers_to_drop = detect_outliers(df, config.outlier_drop_treshold, numeric_cols)
    logger.info('Dropping |%d| rows that have more than |%d| outlier values', len(outliers_to_drop), config.outlier_drop_treshold)
    return df.drop(outliers_to_drop, axis = 0).reset_index(drop=True)

def _target_transform(y):
    return np.log1p(y).values


def construct_preprocess_pipe():
    preprocessing_pipe = Pipeline(steps =   [   ('feature_drop', FeatureDropTransformer(config.DROP_FEATURES)),
                                                ('type_transform', TypeTransformer(config.FEATURE_TO_TYPE)),
                                                ('typo_fix', FixTypoTransformer(config.COLS_WITH_TYPOS)),
                                                ('none_fill', FillNATransformer(config.NONE_FEATURES, 'None')),
                                                ('zero_fill', FillNATransformer(config.ZERO_FEATURES, 0)),
                                                ('mode_fill', FillNAWithModeTransformer(config.MODE_FEATURES)),
                                                ('special_fill', FillNATransformer(['Functional'], 'Typ')),
                                                ('lotfrontage_fill', LotFrontageTransformer()),
                                                ('missing_check', CheckNoMoreMissingValuesTransformer()),
                                                ('clipping', ClipTransformer(config.CLIP_FEATURES)),
                                                ('totalsf_add', TotalSFTransformer()),
                                                ('skew_fix', SkewnessTransformer(config.skew_transform_treshold)),
                                                ('scale_numeric', StandardizationTransformer()),
                                                ('label_encoding', OrdinalEncoderTransformer(config.ORDINAL_FEATURES)),
                                                ('get_dummies', OneHotEncodingTransformer(config.CATEGORICAL_FEATURES, config.drop_strategy)),
                                                ('feature_selection', VarianceThresholdTransformer(config.variance_threshold)),
                                                ('diagnostics', DiagnosticTransformer()),
                                                ('pandas_to_numpy', PandasToNumpyTransformer())
                                            ] )
    
    return preprocessing_pipe


def prepare_training_data(df):
    df = _outlier_drop(df)
    X = df.drop('SalePrice', axis = 1)
    y = _target_transform(df['SalePrice'])
    
    return X, y