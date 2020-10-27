import pandas as pd
import numpy as np
import config
import os

from sklearn.model_selection import cross_val_score, KFold
from preprocess import construct_preprocess_pipe, prepare_training_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import joblib
from pactools.grid_search import GridSearchCVProgressBar

import logging

logging.basicConfig(
    format=config.log_format,
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


df_train = pd.read_csv('data/train.csv')


X, y = prepare_training_data(df_train)

preprocessing_pipe = construct_preprocess_pipe()


logger.info('Optimizing Regressors one by one.')

kf = KFold(config.n_folds, shuffle=True, random_state=config.random_seed).get_n_splits(X)


lgb_regressor = Pipeline( steps = [ ( 'preprocessing', preprocessing_pipe),
                                    ( 'model', lgb.LGBMRegressor(random_state=config.random_seed) ) ] )


lgb_grid = GridSearchCVProgressBar(lgb_regressor, config.lgb_grid_params, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = kf, verbose = 1)
logger.info('Optimizing LGBMRegressor. Paramgrid: |%s|', config.lgb_grid_params)
lgb_grid.fit(X, y)
logger.info('Optimized LGBMRegressor. Best params: |%s|', lgb_grid.best_params_)


lasso_regressor = Pipeline( steps = [ ( 'preprocessing', preprocessing_pipe),
                                    ( 'model', Lasso(random_state=config.random_seed) ) ] )

lasso_grid = GridSearchCVProgressBar(lasso_regressor, config.lasso_grid_params, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = kf, verbose = 1)
logger.info('Optimizing LassoRegressor. Paramgrid: |%s|', config.lasso_grid_params)
lasso_grid.fit(X, y)
logger.info('Optimized LassoRegressor. Best params: |%s|', lasso_grid.best_params_)


rf_regressor = Pipeline( steps = [ ( 'preprocessing', preprocessing_pipe),
                                    ( 'model', RandomForestRegressor(random_state=config.random_seed) ) ] )

rf_grid = GridSearchCVProgressBar(rf_regressor, config.rf_grid_params, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = kf, verbose = 1)
logger.info('Optimizing RandomForestRegressor. Paramgrid: |%s|', config.rf_grid_params)
rf_grid.fit(X, y)
logger.info('Optimized RandomForestRegressor. Best params: |%s|', rf_grid.best_params_)

svr_regressor = Pipeline( steps = [ ( 'preprocessing', preprocessing_pipe),
                                    ( 'model', SVR() ) ] )

svr_grid = GridSearchCVProgressBar(svr_regressor, config.svr_grid_params, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = kf, verbose = 1)
logger.info('Optimizing SupportVectorRegressor. Paramgrid: |%s|', config.svr_grid_params)
svr_grid.fit(X, y)
logger.info('Optimized SupportVectorRegressor. Best params: |%s|', svr_grid.best_params_)

logger.info('Stacking the regressors and retraining the stack using the optimized hyperparameters.')

def remove_prefix_of_params(prefix, param_dict):
    len_prefix = len(prefix)
    return { k[len_prefix:]:v for k,v in param_dict.items() }

preprocessing_pipe = construct_preprocess_pipe()
X_transformed = preprocessing_pipe.fit_transform(X)

lgb_regressor = lgb.LGBMRegressor(**remove_prefix_of_params('model__', lgb_grid.best_params_),
                                random_state=config.random_seed)
lasso_regressor = Lasso(**remove_prefix_of_params('model__', lasso_grid.best_params_),
                                random_state=config.random_seed)
rf_regressor = RandomForestRegressor(**remove_prefix_of_params('model__', rf_grid.best_params_),
                                random_state=config.random_seed)

svr_regressor = SVR(**remove_prefix_of_params('model__', svr_grid.best_params_))

meta_regressor = Lasso()

stack = StackingCVRegressor(regressors=[lgb_regressor, lasso_regressor, rf_regressor, svr_regressor],
                            meta_regressor=meta_regressor,
                            random_state=config.random_seed)

stack.fit(X_transformed, y)

logger.info('Stacked model training finished. Saving stacked model. Stacked model params: |%s|', stack.get_params())


if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(preprocessing_pipe, 'models/preprocessor.pipe', compress = 1)
joblib.dump(stack, 'models/stacked.model', compress = 1)