import numpy as np

model_to_use = 'prod_stacked.model'
pipe_to_use = 'prod_preprocessor.pipe'

lgb_grid_params = {
    'model__num_leaves': range(5, 21),
    'model__max_depth': range(3, 9),
    'model__learning_rate': np.linspace(0.05, 0.2, 10),
    'model__n_estimators': range(30, 151, 10),
    'model__min_child_weight': np.linspace(0.2, 0.6, 4)
}

lasso_grid_params = {
    'model__alpha': np.linspace(0.0001, 0.01, 25),
    'model__max_iter': [100000]
}

svr_grid_params = {
    'model__cache_size': [5000], 
    'model__C':  np.linspace(0.1, 0.5, 15),
    'model__epsilon': np.linspace(0.01, 0.1, 15),
    'model__coef0': np.linspace(0.5, 1, 15),
    'model__kernel': ['poly']
}

rf_grid_params = {
    'model__n_estimators': range(50, 251, 25),
    'model__max_depth': range(5, 13),
    'model__min_samples_split': range(2, 9, 2),
    'model__min_samples_leaf': range(3, 7),
    'model__max_features': range(10, 51, 10)
}