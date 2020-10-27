import numpy as np

lgb_grid_params = {
    'model__num_leaves': [5],
    'model__max_depth': [3],
    'model__learning_rate': [0.1],
    'model__n_estimators': [140],
    'model__min_child_weight': [0.4]
}

lasso_grid_params = {
    'model__alpha': [0.0005],
    'model__max_iter': [100000]
}

svr_grid_params = {
    'model__cache_size': [5000], 
    'model__C':  [0.5],
    'model__epsilon': [0.01642857142857143],
    'model__coef0': [1.0],
    'model__kernel': ['poly']
}

rf_grid_params = {
    'model__n_estimators': [75],
    'model__max_depth': [12],
    'model__min_samples_split': [2],
    'model__min_samples_leaf': [3],
    'model__max_features': [50]
}