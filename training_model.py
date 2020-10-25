import pandas as pd
import numpy as np
import config

from sklearn.model_selection import cross_val_score, KFold
from preprocess import construct_preprocess_pipe, prepare_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)



df_train = pd.read_csv('data/train.csv')


X, y = prepare_data(df_train)

preprocessing_pipe = construct_preprocess_pipe()


def rmse_cv(model, X, y):
    kf = KFold(config.n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return rmse

full_pipeline = Pipeline( steps = [ ( 'preprocessing', preprocessing_pipe),
                                    ( 'model', LinearRegression() ) ] )

rmse = rmse_cv(full_pipeline, X, y)

logger.info("RMSE for LinearRegression: %f", np.mean(rmse))

#X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )


