# e2e ML

> An end to end solution for automl.

Pass in your data, add some information about it and get a full pipelines in
return. Data preprocessing, feature creation, modelling and evaluation with just
a few lines of code.

![Header image](header.png)

## Contents

<!-- toc -->

* [Installation](#installation)
* [Usage example](#usage-example)
* [Linting and Pre-Commit](#linting-and-pre-commit)
* [Disclaimer](#disclaimer)
* [Development](#development)
  * [Adding or Removing Dependencies](#adding-or-removing-dependencies)
  * [Building and Publishing](#building-and-publishing)
  * [Documentation](#documentation)
  * [Pull Requests](#pull-requests)
* [Release History](#release-history)
* [References](#references)
* [Meta](#meta)

<!-- tocstop -->

## Installation

From PyPI:

```sh
pip install e2eml
```

We highly recommend to create a new virtual environment first. Then install
e2e-ml into it. In the environment also download the pretrained spacy model
with. Otherwise e2eml will do this automatically during runtime.

e2eml can also be installed into a RAPIDS environment. For this we recommend to
create a fresh environment following [RAPIDS](https://rapids.ai/start.html)
instructions. After environment installation and activation, a special
installation is needed to not run into installation issues.

Just run:

```sh
pip install e2eml[rapids]
```

This will additionally install cupy and cython to prevent issues. Additionally
it is needed to run:

```sh
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# also spacy supports GPU acceleration
pip install -U spacy[cuda112] #cuda112 depends on your actual cuda version, see: https://spacy.io/usage
```

Otherwise Pytorch will fail trying to run on GPU.

If e2eml shall be installed together with Jupyter core and ipython, please
install with:

```sh
pip install e2eml[full]
```

instead.

## Usage example

e2e has been designed to create state-of-the-art machine learning pipelines with
a few lines of code. Basic example of usage:

```python
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
                        rapids_acceleration=True, # if installed into a conda environment with NVIDIA Rapids, this can be used to accelerate preprocessing with GPU
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
                                            "elasticnet": False,
                                             "xgboost": True,
                                             "ngboost": True,
                                             "lgbm": True,
                                             "tabnet": False,
                                             "vowpal_wabbit": True,
                                             "sklearn_ensemble": True,
                                             "catboost": False
                                             }

Also preprocessing steps can be selected:
test_class.blueprint_step_selection_non_nlp = {
            "automatic_type_detection_casting": True,
            "early_numeric_only_feature_selection": True,
            "remove_duplicate_column_names": True,
            "reset_dataframe_index": True,
            "regex_clean_text_data": False,
            "handle_target_skewness": False,
            "holistic_null_filling": True, # slow
            "iterative_null_imputation": False, # very slow
            "fill_infinite_values": True,
            "datetime_converter": True,
            "pos_tagging_pca": False, # slow with many categories
            "append_text_sentiment_score": False,
            "tfidf_vectorizer_to_pca": True, # slow with many categories
            "tfidf_vectorizer": False,
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
            "clustering_as_a_feature_kmeans_loop": True, # slow for big data, but can be heavily accelerated using rapids_acceleration=True during class instantiation
            "clustering_as_a_feature_gaussian_mixture_loop": True, # slow for big data, but can be heavily accelerated using rapids_acceleration=True during class instantiation (will run a Kmeans on GPU)
            "pca_clustering_results": True,
            "reduce_memory_footprint": False,
            "automated_feature_selection": True,
            "bruteforce_random_feature_selection": False, # slow, this feature is experimental!
            "sort_columns_alphabetically": True,
            "synthetic_data_augmentation": False, # this feature is experimental, can be heavily accelerated using rapids_acceleration=True during class instantiation
            "delete_unpredictable_training_rows": False, # this feature is experimental!
            "scale_data": False,
            "smote": False,
            "autoencoder_based_oversampling": False, # perfect for imbalanced binary and multiclass data
            "final_pca_dimensionality_reduction": False
        }

The bruteforce_random_feature_selection step is experimental. It showed promising results. The number of trials can be controlled.
This step is useful, if the model overfitted (which should happen rarely), because too many features with too little
feature importance have been considered.
like test_class.hyperparameter_tuning_rounds["bruteforce_random"] = 400 .

Generally the class instance is a control center and gives room for plenty of customization.
Never update the class attributes like shown below.

test_class.tabnet_settings = "batch_size": rec_batch_size,
                                "virtual_batch_size": virtual_batch_size,
                                # pred batch size?
                                "num_workers": 0,
                                "max_epochs": 1000}

test_class.hyperparameter_tuning_rounds = {"xgboost": 100,
                                             "lgbm": 500,
                                             "tabnet": 25,
                                             "ngboost": 25,
                                             "sklearn_ensemble": 10,
                                             "ridge": 500,
                                             "elasticnet": 100,
                                             "catboost": 25,
                                             "sgd": 2000,
                                             "svm": 50,
                                             "svm_regression": 50,
                                             "ransac": 50,
                                             "multinomial_nb": 100,
                                             "bruteforce_random": 400,
                                             "synthetic_data_augmentation": 100,
                                             "autoencoder_based_oversampling": 200,
                                             "final_kernel_pca_dimensionality_reduction": 50,
                                             "final_pca_dimensionality_reduction": 50}

test_class.hyperparameter_tuning_max_runtime_secs = {"xgboost": 2*60*60,
                                                       "lgbm": 2*60*60,
                                                       "tabnet": 2*60*60,
                                                       "ngboost": 2*60*60,
                                                       "sklearn_ensemble": 2*60*60,
                                                       "ridge": 2*60*60,
                                                       "elasticnet": 2*60*60,
                                                       "catboost": 2*60*60,
                                                       "sgd": 2*60*60,
                                                       "svm": 2*60*60,
                                                       "svm_regression": 2*60*60,
                                                       "ransac": 2*60*60,
                                                       "multinomial_nb": 2*60*60,
                                                       "bruteforce_random": 2*60*60,
                                                       "synthetic_data_augmentation": 1*60*60,
                                                       "autoencoder_based_oversampling": 2*60*60,
                                                       "final_kernel_pca_dimensionality_reduction": 4*60*60,
                                                       "final_pca_dimensionality_reduction": 2*60*60}

When these parameters have to updated, please overwrite the keys individually to not break the blueprints eventually.
I.e.: test_class.hyperparameter_tuning_max_runtime_secs["xgboost"] = 12*60*60 would work fine.

Working with big data can bring all hardware to it's needs. e2eml has been tested with:
- Ryzen 5950x (16 cores CPU)
- Geforce RTX 3090 (24GB VRAM)
- 64GB RAM
e2eml has been able to process 100k rows with 200 columns approximately using these specs stable for non-blended
blueprints. Blended blueprints consume more resources as e2eml keep the trained models in memory as of now.

For data bigger than 100k rows it is possible to limit the amount of data for various preprocessing steps:
- test_class.feature_selection_sample_size = 100000 # for feature selection
- test_class.hyperparameter_tuning_sample_size = 100000 # for model hyperparameter optimization
- test_class.brute_force_selection_sample_size = 15000 # for an experimental feature selection

For binary classification a sample size of 100k datapoints is sufficient in most cases. Hyperparameter tuning sample size can be much less,
depending on class imbalance.

For multiclass we recommend to start with small samples as algorithms like Xgboost and LGBM will easily grow in memory consumption
with growing number of classes.

Whenever classes are imbalanced (binary & multiclass) we recommend to use the preprocessing step "autoencoder_based_oversampling".
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

## Linting and Pre-Commit

This project uses pre-commit to enforce style.

To install the pre-commit hooks, first install pre-commit into the project's
virtual environment:

```sh
pip install pre-commit
```

Then install the project hooks:

```sh
pre-commit install
```

Now, whenever you make a commit, the linting and autoformatting will
automatically run.

## Disclaimer

e2e is not designed to quickly iterate over several algorithms and suggest you
the best. It is made to deliver state-of-the-art performance as ready-to-go
blueprints. e2e-ml blueprints contain:

* preprocessing (outlier, rare feature, datetime, categorical and NLP handling)
* feature creation (binning, clustering, categorical and NLP features)
* automated feature selection
* model training (with crossfold validation)
* automated hyperparameter tuning
* model evaluation

This comes at the cost of runtime. Depending on your data we recommend strong
hardware.

## Development

This project uses [poetry](https://python-poetry.org/).

To install the project for development, run:

```sh
poetry install
```

This will install all dependencies and development dependencies into a virtual
environment.

### Adding or Removing Dependencies

To add or remove a dependency, use `poetry add <package>` or
`poetry remove <package>` respectively. Use the `--dev` flag for development
dependencies.

### Building and Publishing

To build and publish the project, run

```sh
poetry publish --build
```

### Documentation

This project comes with documentation. To build the docs, run:

```sh
cd docs
make docs
```

You may then browse the HTML docs at `docs/build/docs/index.html`.

### Pull Requests

We welcome Pull Requests! Please make a PR against the `develop` branch.

## Release History

* 2.10.02
  * Adjusted dependencies for Pandas and Spacy
* 2.10.01
  * Added references & citations to Readme
  * Added is_imbalanced flag to Timewalk
  * Removed babel from dependencies & updated some of them
* 2.9.96
  * Timewalk got adjustments
  * Fixed a bug where row deletion has been incompatible with Tabnet
* 2.9.95
  * SHAP based feature selection increased to 20 folds (from 10)
  * less unnecessary print outs
* 2.9.93
  * Added SHAP based feature selection
  * Removed Xgboost from Timewalk as default due to computational and runtime costs
  * Suppress all warnings of LGBM focal during multiclass tasks
* 2.9.92
  * e2eml uses poetry
  * introduction of Github actions to check linting
  * bug fix of LGBM focal failing due to missing hyperparameter tuning specifications
  * preparation for Readthedocs implementation
* 2.9.9
  * Added Multinomial Bayes Classifier
  * Added SVM for regression
  * Refined Sklearn ensembles
* 2.9.8
  * Added Quadrant Discriminent Analysis
  * Added Support Vector machines
  * Added Ransac regressor
* 2.9.7
  * updated Plotly dependency to 5.4.0
  * Improved Xgboost for imbalanced data
* 2.9.6
  * Added TimeTravel and timewalk: TimeTravel will save the class instance after
    each preprocessing step, timewalk will automatically try different
    preprocessing steps with different algorithms to find the best combination
  * Updated dependencies to use newest versions of scikit-learn and
    category-encoders
* 2.9.0
  * bug fixes with synthetic data augmentation for regression
  * bug fix of target encoding during regression
  * enhanced hyperparameter space for autoencoder based oversampling
  * added final PCA dimensionality reduction as optional preprocessing step
* 2.8.1
  * autoencoder based oversampling will go through hyperprameter tuning first
    (for each class individually)
  * optimized TabNet performance
* 2.7.5
  * added oversampling based on variational autoencoder (experimental)
* 2.7.4
  * fixed target encoding for multiclass classification
  * improved performance on multiclass tasks
  * improved Xgboost & TabNet performance on binary classification
  * added auto-tuned clustering as a feature
* 2.6.3
  * small bugfixes
* 2.6.1
  * Hyperparameter tuning does happen on a sample of the train data from now on
    (sample size can be controlled)
  * An experimental feature has been added, which tries to find unpredictable
    training data rows to delete them from the training (this accelerates
    training, but costs a bit model performance)
  * Blueprints can be accelerated with Nvidia RAPIDS (works on clustering only f
    or now)
* 2.5.9
  * optimized loss function for TabNet
* 2.5.1
  * Optimized loss function for synthetic data augmentation
  * Adjusted library dependencies
  * Improved target encoding
* 2.3.1
  * Changed feature selection backend from Xgboost to LGBM
  * POS tagging is off on default from this version
* 2.2.9
  * bug fixes
  * added an experimental feature to optimize training data with synthetic data
  * added optional early feature selection (numeric only)
* 2.2.2
  * transformers can be loaded into Google Colab from Gdrive
* 2.1.2
  * Improved TFIDF vectorizer performance & non transformer NLP applications
  * Improved POS tagging stability
* 2.1.1
  * Completely overworked preprocessing setup (changed API). Preprocessing
    blueprints can be customized through a class attribute now
  * Completely overworked special multimodel blueprints. The paricipating
    algorithms can be customized through a class attribute now
  * Improved NULL handling & regression performance
  * Added Catboost & Elasticnet
  * Updated Readme
  * First unittests
  * Added Stochastic Gradient classifier & regressor
* 1.8.2
  * Added Ridge classifier and regression as new blueprints
* 1.8.1
  * Added another layer of feature selection
* 1.8.0
  * Transformer padding length will be max text length + 20% instead of static
    300
  * Transformers use AutoModelForSequenceClassification instead of hardcoded
    transformers now
  * Hyperparameter tuning rounds and timeout can be controlled globally via
    class attribute now
* 1.7.8
  * Instead of a global probability threshold, e2eml stores threshold for each
    tested model
  * Deprecated binary boosting blender due to lack of performance
  * Added filling of inf values
* 1.7.3
  * Improved preprocessing
  * Improved regression performance
  * Deprecated regression boosting blender and replaced my multi
    model/architecture blender
  * Transformers can optionally discard worst models, but will keep all 5 by
    default
  * e2eml should be installable on Amazon Sagemaker now
* 1.7.0
  * Added TabNet classifier and regressor with automated hyperparameter
    optimization
* 1.6.5
  * improvements of NLP transformers
* 1.5.8
  * Fixes bug around preprocessing_type='nlp'
  * replaced pickle with dill for saving and loading objects
* 1.5.3
  * Added transformer blueprints for NLP classification and regression
  * renamed Vowpal Wabbit blueprint to fit into blueprint naming convention
  * Created "extras" options for library installation: 'rapids' installs extras,
    so e2eml can be installed into into a rapids environment while 'jupyter'
    adds jupyter core and ipython. 'full' installs all of them.
* 1.3.9
  * Fixed issue with automated GPU-acceleration detection and flagging
  * Fixed avg regression blueprint where eval function tried to call
    classification evaluation
  * Moved POS tagging + PCA step into non-NLP pipeline as it showed good results
    in general
  * improved NLP part (more and better feature engineering and preprocessing) of
    blueprints for better performance
  * Added Vowpal Wabbit for classification and regression and replaced stacking
    ensemble in automated model exploration by Vowpal Wabbit as well
  * Set random_state for train_test splits for consistency
  * Fixed sklearn dependency to 0.22.0 due to six import error
* 1.0.1
  * Optimized package requirements
  * Pinned LGBM requirement to version 3.1.0 due to the bug "LightGBMError: bin
    size 257 cannot run on GPU #3339"
* 0.9.9
  * Enabled tune_mode parameter during class instantiation.
  * Updated docstings across all functions and changed model defaults.
  * Multiple bug fixes (LGBM regression accurate mode, label encoding and
    permutation tests).
  * Enhanced user information & better ROC_AUC display
  * Added automated GPU detection for LGBM and Xgboost.
  * Added functions to save and load blueprints
  * architectural changes (preprocessing organized in blueprints as well)
* 0.9.4
  * First release with classification and regression blueprints. (not available
    anymore)

## References

* Focal loss
  * [Focal loss for LGBM](https://maxhalford.github.io/blog/lightgbm-focal-loss/#first-order-derivative)
  * [Focal loss for LGBM multiclass](https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872)
* Autoencoder
  * [Variational Autoencoder for imbalanced data](https://github.com/lschmiddey/Autoencoder/blob/master/VAE_for_imbalanced_data.ipynb)
* Target Encoding
  * [Target encoding for multiclass](https://towardsdatascience.com/target-encoding-for-multi-class-classification-c9a7bcb1a53)
* Pytorch-TabNet
  * [Arik, S. O., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.](https://arxiv.org/pdf/1908.07442.pdf)
  * [Implementing TabNet in Pytorch](https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279)
* Ngboost
  * [NGBoost: Natural Gradient Boosting for Probabilistic Prediction, arXiv:1910.03225](https://arxiv.org/abs/1910.03225)
* Vowpal Wabbit
  * [Vowpal Wabbit Research overview](https://vowpalwabbit.org/research.html)

## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)

Consultant: Gabriel Stephen Alexander – [Github](https://github.com/bitsofsteve)

Special thanks to: Alex McKenzie - [LinkedIn](https://de.linkedin.com/in/alex-mckenzie)

[e2eml Github repository](https://github.com/ThomasMeissnerDS/e2e_ml)
