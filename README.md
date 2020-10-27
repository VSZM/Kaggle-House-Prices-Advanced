[![Build Status](https://travis-ci.com/VSZM/Kaggle-House-Prices-Advanced.svg?branch=master)](https://travis-ci.com/VSZM/Kaggle-House-Prices-Advanced.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/VSZM/Kaggle-House-Prices-Advanced/badge.svg?branch=master)](https://coveralls.io/github/VSZM/Kaggle-House-Prices-Advanced?branch=master)

# About

This repository contains my take on the Kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

The top 100-200 of the Leaderboard on kaggle exploit a data leakage of the test set. My goal was to build the model decently, only using the train set like in a real world setting. 

A lot of other notebooks also fall into the issue of leaking test data into training. For example calculating the mean for a column using the train+test set, instead of calculating on the train only. This leads to our model overfitting to the test set as well, and our model could perform worse on truly unknown values because of this.

The solution I provide here looks like a real world application, instead of the commonly used notebook format. I provide configuration framework, logging, and testing like in a real application. 

I did not have resources to do extensive hyperparameter tuning, but I have created the framework to do so.

### **Using the Model**

I provide a trained model, that can be used out of the box. The input of the model is a DataFrame. Column names must follow the original columns of the dataset, as seen in [the description](data/data_description.txt). 

```python
from model import HousePriceModel

model = HousePriceModel.getInstance()

df_test = # load your data here in a DataFrame format
predictions = model.predict(df_test)
```

You can integrate this snippet into the interface of your liking, be it Rest, WebService or a simple script like in [submission.py](submission.py).

You also have to make sure the correct environment is defined using the *ENV* environment variable. Current supported values: test, prod

See config_*.py for more configuration details.

### **Training Models**

Training models can be done by running [training_model.py](training_model.py). The correct environment must be set here as well before running. 

The hyper-parameter search grid can be tuned in *config_${ENV}.py*. You can also fine tune the preprocessing pipeline here and even search for combinations of steps by turning them on/off in the grid search. 

### **Credits**

There are a few nice resources I used during creating this repository:

- https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
- https://www.kaggle.com/orhankaramancode/ensemble-stacked-regressors-92-acc
- https://www.kaggle.com/pavan9065/house-prices
- https://github.com/ksator/continuous-integration-with-python