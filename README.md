# e2e ML
> An end to end solution for automl. .

Pass in your data, add some information about it and get a full pipelines in return. Data preprocessing,
feature creation, modelling and evaluation with just a few lines of code.

![](header.png)

## Installation

From Pypi:

```sh
pip install e2eml
```
We highly recommend to create a new virtual environment first. Then install e2e-ml into it. In the environment also download
the pretrained spacy model with:
```sh
python3 -m spacy download en
```
or
```sh
python -m spacy download en
```
(depending on your operating system.)

## Usage example

e2e has been designed to create state-of-the-art machine learning pipelines with a few lines of code. Basic example of usage:
```sh
import e2eml
from e2eml.classification import classification_blueprints
import pandas as pd
# import data
df = pd.read_csv("Your.csv")

# split into a test/train & holdout set (holdout for prediction illustration here, but not required at all)
train_df = df.head(1000).copy()
holdout_df = df.tail(200).copy() # make sure
# saving the holdout dataset's target for later and delete it from holdout dataset
target = "target_column"
holdout_target = holdout_df[target].copy()
del holdout_df[target]

# instantiate the needed blueprints class
from classification import classification_blueprints # regression bps are available with from regression import regression_blueprints
test_class = classification_blueprints.ClassificationBluePrint(datasource=train_df, 
                        target_variable=target,
                        train_split_type='cross',
                        preferred_training_mode='auto', # Auto will automatically identify, if LGBM & Xgboost can use GPU acceleration*
                        tune_mode='accurate' # hyperparameter sets will be validated with 10-fold CV Set this to 'simple' for 1-fold CV
                        #categorical_columns=cat_columns # you can define categorical columns, otherwise e2e does this automatically
                        #date_columns=date_columns # you can also define date columns (expected is YYYY-MM-DD format)
                                                               )
                                                                 
"""
*
'Auto' is recommended for preferred_training_mode parameter, but with 'CPU' and 'GPU' it can also be controlled manually.
If you install Xgboost & LGBM into the same environment as GPU accelerated versions, you can set preferred_training_mode='gpu'.
This will massively improve training times and speed up SHAP feature importance for LGBM and Xgboost related tasks.
For Xgboost this should work out of the box, if installed into a RAPIDS environment.
"""
# run actual blueprint
test_class.ml_bp01_multiclass_full_processing_xgb_prob(preprocessing_type='nlp')
"""
When choosing blueprints several options are available:

Multiclass blueprints can handle binary and multiclass tasks:
- ml_bp00_train_test_binary_full_processing_log_reg_prob()
- ml_bp01_multiclass_full_processing_xgb_prob()
- ml_bp02_multiclass_full_processing_lgbm_prob()
- ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble()
- ml_bp04_multiclass_full_processing_ngboost()
- ml_special_binary_full_processing_boosting_blender()
- ml_special_multiclass_auto_model_exploration()

There are regression blueprints as well (in regression module):
- ml_bp10_train_test_regression_full_processing_linear_reg()
- ml_bp11_regression_full_processing_xgboost()
- ml_bp12_regressions_full_processing_lgbm()
- ml_bp13_regression_full_processing_sklearn_stacking_ensemble()
- ml_bp14_regressions_full_processing_ngboost()
- ml_special_regression_full_processing_boosting_blender()
- ml_special_regression_auto_model_exploration()

The preproccesing_type has 2 modes as of now:
- full (default), which runs all steps except NLP specific ones
- nlp: Adds some NLP related feature enginering steps.
"""
# After running the blueprint the pipeline is done. I can be saved with:
test_class.save_load_model_file(action='save')

# The blueprint can be loaded with
loaded_test_class = save_load_model_file(action='load')

# predict on new data (in this case our holdout) with loaded blueprint
loaded_test_class.ml_bp01_multiclass_full_processing_xgb_prob(holdout_df, preprocessing_type='nlp')

# predictions can be accessed via a class attribute
print(churn_class.predicted_classes['xgboost'])
```
# Disclaimer
e2e is not designed to quickly iterate over several algorithms and suggest you the best. It is made to deliver
state-of-the-art performance as ready-to-go blueprints. e2e-ml blueprints contain:
- preprocessing (outlier, rare feature, datetime, categorical and NLP handling)
- feature creation (binning, clustering, categorical and NLP features)
- automated feature selection
- model training with crossfold validation
- automated hyperparameter tuning
- model evaluation
  This comes at the cost of runtime. Depending on your data we recommend strong hardware.

## Release History

* 0.9.8
  * Enabled tune_mode parameter during class instantiation.
  * Updated docstings across all functions and changed model defaults.
  * Multiple bug fixes (LGBM regression accurate mode, label encoding and permutation tests).
  * Enhanced user information & better ROC_AUC display
  * Added automated GPU detection for LGBM and Xgboost.
* 0.9.4
  * First release with classification and regression blueprints. (not available anymore)

## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)

Consultant: Gabriel Stephen Alexander – [Github](https://github.com/bitsofsteve)


[e2e-ml Github repository](https://github.com/ThomasMeissnerDS/e2e_ml)

