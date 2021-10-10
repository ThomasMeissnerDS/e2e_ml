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
the pretrained spacy model with. Otherwise e2eml will do this automatically during runtime.

e2eml can also be installed into a RAPIDS environment. For this we recommend to create a fresh environment following
[RAPIDS](https://rapids.ai/start.html) instructions. After environment installation and activation, a special installation is needed to not run into installation issues.
Just run:
```sh
pip install e2eml[rapids]
```
This will additionally install cupy and cython to prevent issues. Additionally it is needed to run:
```sh
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# also spacy supports GPU acceleration
pip install -U spacy[cuda112] #cuda112 depends on your actual cuda version, see: https://spacy.io/usage
```
Otherwise Pytorch will fail trying to run on GPU.
If e2eml shall be installed together with Jupyter core and ipython, please install with:
```sh
pip install e2eml[full]
```
instead.

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
test_class.ml_bp01_multiclass_full_processing_xgb_prob() 

"""
When choosing blueprints several options are available:

Multiclass blueprints can handle binary and multiclass tasks:
- ml_bp00_train_test_binary_full_processing_log_reg_prob()
- ml_bp01_multiclass_full_processing_xgb_prob()
- ml_bp02_multiclass_full_processing_lgbm_prob()
- ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble()
- ml_bp04_multiclass_full_processing_ngboost()
- ml_bp05_multiclass_full_processing_vowpal_wabbit
- ml_bp06_multiclass_full_processing_bert_transformer() # for NLP specifically
- ml_bp07_multiclass_full_processing_tabnet()
- ml_bp08_multiclass_full_processing_ridge()
- ml_bp09_multiclass_full_processing_catboost()
- ml_bp10_multiclass_full_processing_sgd()
- ml_special_binary_full_processing_boosting_blender()
- ml_special_multiclass_auto_model_exploration()
- ml_special_multiclass_full_processing_multimodel_max_voting()

There are regression blueprints as well (in regression module):
- ml_bp10_train_test_regression_full_processing_linear_reg()
- ml_bp11_regression_full_processing_xgboost()
- ml_bp12_regressions_full_processing_lgbm()
- ml_bp13_regression_full_processing_sklearn_stacking_ensemble()
- ml_bp14_regressions_full_processing_ngboost()
- ml_bp15_regression_full_processing_vowpal_wabbit_reg()
- ml_bp16_regressions_full_processing_bert_transformer()
- ml_bp17_regression_full_processing_tabnet_reg()
- ml_bp18_regression_full_processing_ridge_reg
- ml_bp20_regression_full_processing_catboost()
- ml_bp20_regression_full_processing_sgd()
- ml_special_regression_full_processing_multimodel_avg_blender()
- ml_special_regression_auto_model_exploration()

In ensembles algorithms can be chosen via the class attribute:
test_class.special_blueprint_algorithms = {"ridge": True,
                                             "xgboost": True,
                                             "ngboost": True,
                                             "lgbm": True,
                                             "tabnet": False,
                                             "vowpal_wabbit": True,
                                             "sklearn_ensemble": True
                                             }
                                             
Also preprocessing steps can be selected:
test_class.blueprint_step_selection_non_nlp = {
            "automatic_type_detection_casting": True,
            "remove_duplicate_column_names": True,
            "reset_dataframe_index": True,
            "handle_target_skewness": True,
            "holistic_null_filling": True,
            "fill_infinite_values": True,
            "datetime_converter": True,
            "pos_tagging_pca": True,
            "append_text_sentiment_score": False,
            "tfidf_vectorizer_to_pca": True,
            "rare_feature_processing": True,
            "cardinality_remover": True,
            "delete_high_null_cols": True,
            "numeric_binarizer_pca": True,
            "onehot_pca": True,
            "category_encoding": True,
            "fill_nulls_static": True,
            "data_binning": True,
            "outlier_care": True,
            "remove_collinearity": True,
            "skewness_removal": True,
            "clustering_as_a_feature_dbscan": True,
            "clustering_as_a_feature_kmeans_loop": True,
            "clustering_as_a_feature_gaussian_mixture_loop": True,
            "reduce_memory_footprint": False,
            "automated_feature_selection": True,
            "bruteforce_random_feature_selection": True, # This might run to long runtimes, but usually improves performance
            "sort_columns_alphabetically": True,
            "scale_data": False,
            "smote": False
        }
        
The bruteforce_random_feature_selection step is experimental. It showed promising results. The number of trials can be controlled 
like test_class.hyperparameter_tuning_rounds["bruteforce_random"] = 400 .

Generally the class instance is a control center and gives room for plenty of customization:

test_class.tabnet_settings = {f"batch_size": rec_batch_size,
                                "virtual_batch_size": virtual_batch_size,
                                # pred batch size?
                                "num_workers": 0,
                                "max_epochs": 1000}

test_class.hyperparameter_tuning_rounds = {"xgboost": 100,
                                             "lgbm": 100,
                                             "tabnet": 25,
                                             "ngboost": 25,
                                             "sklearn_ensemble": 10,
                                             "ridge": 100,
                                             "bruteforce_random": 400}

test_class.hyperparameter_tuning_max_runtime_secs = {"xgboost": 24*60*60,
                                             "lgbm": 24*60*60,
                                             "tabnet": 24*60*60,
                                             "ngboost": 24*60*60,
                                             "sklearn_ensemble": 24*60*60,
                                             "ridge": 24*60*60,
                                             "bruteforce_random": 24*60*60}
"""
# After running the blueprint the pipeline is done. I can be saved with:
save_to_production(test_class, file_name='automl_instance')

# The blueprint can be loaded with
loaded_test_class = load_for_production(file_name='automl_instance')

# predict on new data (in this case our holdout) with loaded blueprint
loaded_test_class.ml_bp01_multiclass_full_processing_xgb_prob(holdout_df)

# predictions can be accessed via a class attribute
print(churn_class.predicted_classes['xgboost'])
```
# Disclaimer
e2e is not designed to quickly iterate over several algorithms and suggest you the best. It is made to deliver
state-of-the-art performance as ready-to-go blueprints. e2e-ml blueprints contain:
- preprocessing (outlier, rare feature, datetime, categorical and NLP handling)
- feature creation (binning, clustering, categorical and NLP features)
- automated feature selection
- model training (with crossfold validation)
- automated hyperparameter tuning
- model evaluation
  This comes at the cost of runtime. Depending on your data we recommend strong hardware.

## Release History
* 2.3.0
 - Changed feature selection backend from Xgboost to LGBM
 - POS tagging is off on default from this version
* 2.2.9
 - bug fixes
 - added an experimental feature to optimize training data with synthetic data
 - added optional early feature selection (numeric only)
* 2.2.2
 - transformers can be loaded into Google Colab from Gdrive
* 2.1.2
 - Improved TFIDF vectorizer performance & non transformer NLP applications
 - Improved POS tagging stability
* 2.1.1
 - Completely overworked preprocessing setup (changed API). Preprocessing blueprints can be customized through a class
   attribute now
 - Completely overworked special multimodel blueprints. The paricipating algorithms can be customized through a class
   attribute now
 - Improved NULL handling & regression performance
 - Added Catboost & Elasticnet
 - Updated Readme
 - First unittests
 - Added Stochastic Gradient classifier & regressor
* 1.8.2
 - Added Ridge classifier and regression as new blueprints
* 1.8.1
 - Added another layer of feature selection
* 1.8.0
 - Transformer padding length will be max text length + 20% instead of static 300
 - Transformers use AutoModelForSequenceClassification instead of hardcoded transformers now
 - Hyperparameter tuning rounds and timeout can be controlled globally via class attribute now
* 1.7.8
  - Instead of a global probability threshold, e2eml stores threshold for each tested model
  - Deprecated binary boosting blender due to lack of performance
  - Added filling of inf values
* 1.7.3
  - Improved preprocessing
  - Improved regression performance
  - Deprecated regression boosting blender and replaced my multi model/architecture blender
  - Transformers can optionally discard worst models, but will keep all 5 by default
  - e2eml should be installable on Amazon Sagemaker now
* 1.7.0
  - Added TabNet classifier and regressor with automated hyperparameter optimization
* 1.6.5
  - improvements of NLP transformers
* 1.5.8
  - Fixes bug around preprocessing_type='nlp'
  - replaced pickle with dill for saving and loading objects
* 1.5.3
  - Added transformer blueprints for NLP classification and regression
  - renamed Vowpal Wabbit blueprint to fit into blueprint naming convention
  - Created "extras" options for library installation: 'rapids' installs extras, so e2eml can be installed into
    into a rapids environment while 'jupyter' adds jupyter core and ipython. 'full' installs all of them.
* 1.3.9
  - Fixed issue with automated GPU-acceleration detection and flagging
  - Fixed avg regression blueprint where eval function tried to call classification evaluation
  - Moved POS tagging + PCA step into non-NLP pipeline as it showed good results in general
  - improved NLP part (more and better feature engineering and preprocessing) of blueprints for better performance
  - Added Vowpal Wabbit for classification and regression and replaced stacking ensemble in automated model exploration
    by Vowpal Wabbit as well
  - Set random_state for train_test splits for consistency
  - Fixed sklearn dependency to 0.22.0 due to six import error
* 1.0.1
  - Optimized package requirements
  - Pinned LGBM requirement to version 3.1.0 due to the bug "LightGBMError: bin size 257 cannot run on GPU #3339"
* 0.9.9
  * Enabled tune_mode parameter during class instantiation.
  * Updated docstings across all functions and changed model defaults.
  * Multiple bug fixes (LGBM regression accurate mode, label encoding and permutation tests).
  * Enhanced user information & better ROC_AUC display
  * Added automated GPU detection for LGBM and Xgboost.
  * Added functions to save and load blueprints
  * architectural changes (preprocessing organized in blueprints as well)
* 0.9.4
  * First release with classification and regression blueprints. (not available anymore)

## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)

Consultant: Gabriel Stephen Alexander – [Github](https://github.com/bitsofsteve)


[e2e-ml Github repository](https://github.com/ThomasMeissnerDS/e2e_ml)

