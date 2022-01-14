import gc
import logging
import os
import random
import time
import warnings

import dill as pickle
import lightgbm
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import psutil
import shap
import torch
import torch.nn.functional as F
import xgboost as xgb
from boostaroota import BoostARoota
from catboost import CatBoostClassifier
from category_encoders import (
    GLMMEncoder,
    LeaveOneOutEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    WOEEncoder,
)
from imblearn.over_sampling import SMOTE
from pandas.core.common import SettingWithCopyWarning
from scipy.stats import (
    binom,
    dweibull,
    expon,
    gamma,
    halfcauchy,
    halfnorm,
    levy,
    norm,
    pareto,
    poisson,
    powernorm,
    rdist,
    semicircular,
    tukeylambda,
)
from sklearn import model_selection
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest, RandomTreesEmbedding
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    make_scorer,
    matthews_corrcoef,
    mean_squared_error,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.svm import OneClassSVM
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from vowpalwabbit.sklearn_vw import VWClassifier, VWRegressor

warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = 1000
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_scaler(param):
    """
    Takes the Optuna dictionary and returns scaler accordingly.
    :param param: Dictionary
    :return: scaler object from sklearn
    """
    if param["transformer"] == "quantile":
        scaler = QuantileTransformer()
    elif param["transformer"] == "maxabs":
        scaler = MaxAbsScaler()
    elif param["transformer"] == "robust":
        scaler = RobustScaler()
    elif param["transformer"] == "minmax":
        scaler = MinMaxScaler()
    elif param["transformer"] == "yeo-johnson":
        scaler = PowerTransformer(method="yeo-johnson")
    elif param["transformer"] == "box_cox":
        scaler = PowerTransformer(method="box-cox")
    elif param["transformer"] == "l1":
        scaler = Normalizer(norm="l1")
    elif param["transformer"] == "l2":
        scaler = Normalizer(norm="l2")
    else:
        scaler = QuantileTransformer(n_quantiles=param["n_quantiles"], random_state=23)
    return scaler


class PreProcessing:
    """
    This class stores all pipeline relevant information. The attribute "df_dict" always holds train and test as well as
    to predict data. The attribute "preprocess_decisions" stores encoders and other information generated during the
    model training. The attributes "predicted_classes" and "predicted_probs" store dictionaries (model names are dictionary keys)
    with predicted classes and probabilities (classification tasks) while "predicted_values" stores regression based
    predictions. The attribute "evaluation_scores" keeps track of model evaluation metrics (in dictionary format).
    :param datasource: Expects a Pandas dataframe (containing the target feature as a column)
    :param target_variable: Name of the target feature's column within the datasource dataframe.
    :param date_columns: Date columns can be passed as lists additionally for respective preprocessing. If not provided
    e2eml will try to detect datetime columns automatically. Date format is expected as YYYY-MM-DD anyway.
    :param categorical_columns: Categorical columns can be passed as lists additionally for respective preprocessing.
    If not provided e2eml will try to detect categorical columns automatically.
    :param nlp_columns: NLP columns expect a string declaring one text column.
    :param unique_identifier: A unique identifier (i.e. an ID column) can be passed as well to preserve this information
     for later processing.
    :param ml_task: Can be 'binary', 'multiclass' or 'regression'. On default will be determined automatically.
    :param preferred_training_mode: Must be 'cpu', if e2eml has been installed into an environment without LGBM and Xgboost on GPU.
    Can be set to 'gpu', if LGBM and Xgboost have been installed with GPU support. The default 'auto' will detect GPU support
    and optimize accordingly. (Default: 'auto')
    :param tune_mode: 'Accurate' will lead use K-fold cross validation per hyperparameter set durig optimization. 'Simple'
    will make use of use 1-fold validation only, which leads to much faster training times.
    :param logging_file_path: Preferred location to save the log file. Will otherwise stored in the current folder.
    :param low_memory_mode: Adds a preprocessing feature to reduce dataframe memory footprint. Will lead to a loss in
    model performance. Will be extended by further memory savings features in future releases.
    However we highly recommend GPU usage to heavily decrease model training times.
    """

    def __init__(  # noqa: C901
        self,
        datasource,
        target_variable,
        date_columns=None,
        categorical_columns=None,
        num_columns=None,
        rarity_cols=None,
        unique_identifier=None,
        selected_feats=None,
        cat_encoded=None,
        cat_encoder_model=None,
        nlp_columns=None,
        nlp_transformer_columns=None,
        transformer_chosen="bert-base-uncased",
        transformer_model_load_from_path=None,
        transformer_model_save_states_path=None,
        transformer_epochs=25,
        prediction_mode=False,
        preferred_training_mode="auto",
        preprocess_decisions=None,
        tune_mode="accurate",
        trained_model=None,
        ml_task=None,
        logging_file_path=None,
        low_memory_mode=False,
        save_models_path=None,
        train_split_type="cross",
        rapids_acceleration=False,
    ):

        self.dataframe = datasource
        self.kfolds_column = None
        self.low_memory_mode = low_memory_mode
        self.save_models_path = save_models_path
        self.logging_file_path = logging_file_path
        logging.basicConfig(
            filename=f"{self.logging_file_path}.log",
            format="%(asctime)s %(message)s",
            level=logging.DEBUG,
        )
        logging.info("Class instance created.")

        # check which type the data source is
        if isinstance(datasource, np.ndarray):
            self.source_format = "numpy array"
        elif isinstance(datasource, pd.DataFrame):
            self.source_format = "Pandas dataframe"
            self.dataframe.columns = self.dataframe.columns.astype(str)
        else:
            self.source_format = "Unknown, not recommened"

        # check if we face a classification problem and check how many classes we have
        if not ml_task:
            try:
                if datasource[target_variable].nunique() > 10:
                    self.class_problem = "regression"
                elif datasource[target_variable].nunique() > 2:
                    self.class_problem = "multiclass"
                    self.num_classes = datasource[target_variable].nunique()
                elif datasource[target_variable].nunique() == 2:
                    self.class_problem = "binary"
                    self.num_classes = 2
                else:
                    self.class_problem = "regression"
            except Exception:
                if len(np.unique(np.array(target_variable))) > 2:
                    self.class_problem = "multiclass"
                    self.num_classes = len(np.unique(np.array(target_variable)))
                elif len(np.unique(np.array(target_variable))) == 2:
                    self.class_problem = "binary"
                    self.num_classes = 2
                else:
                    pass
        else:
            self.class_problem = ml_task
            if ml_task == "multiclass":
                self.num_classes = datasource[target_variable].nunique()
            elif ml_task == "binary":
                self.num_classes = 2
        print(f"Ml task is {self.class_problem}")

        self.binary_unbalanced = False

        if preferred_training_mode == "cpu":
            message = """
            CPU mode has been chosen. Installing e2eml into an environment where LGBM and Xgboost have been installed with GPU acceleration
            is recommended to be able to use preferred_training_mode='gpu'. This will speed up model training and feature importance
            via SHAP.
            """
            logging.warning(f"{message}")
            print(f"{message}")
            self.preferred_training_mode = preferred_training_mode
        elif preferred_training_mode == "gpu":
            print("GPU acceleration chosen.")
            self.preferred_training_mode = preferred_training_mode
        elif preferred_training_mode == "auto":
            print(
                "Preferred training mode auto has been chosen. e2eml will automatically detect, if LGBM and Xgboost can "
                "use GPU acceleration and optimize the workflow accordingly."
            )
            self.preferred_training_mode = preferred_training_mode
        else:
            self.preferred_training_mode = "cpu"
            print("No preferred_training_mode chosen. Fallback to CPU.")
        self.tune_mode = tune_mode
        self.train_split_type = train_split_type
        self.rapids_acceleration = rapids_acceleration
        self.date_columns = date_columns
        self.date_columns_created = None
        self.categorical_columns = categorical_columns
        self.rarity_cols = rarity_cols
        if isinstance(nlp_columns, list):
            print("Please provide nlp_columns parameter with a string.")
            self.nlp_columns = nlp_columns
        else:
            self.nlp_columns = nlp_columns
        self.nlp_transformer_columns = nlp_transformer_columns
        self.nlp_transformers = {}
        self.transformer_chosen = transformer_chosen
        self.transformer_epochs = transformer_epochs
        self.cat_columns_encoded = None
        self.num_columns_encoded = None
        self.unique_identifier = unique_identifier
        self.target_variable = target_variable
        self.labels_encoded = False
        self.new_sin_cos_col_names = None
        self.df_dict = None
        self.blueprint_step_selection_non_nlp = {
            "automatic_type_detection_casting": True,
            "remove_duplicate_column_names": True,
            "reset_dataframe_index": True,
            "fill_infinite_values": True,
            "early_numeric_only_feature_selection": True,
            "delete_high_null_cols": True,
            "data_binning": True,
            "regex_clean_text_data": False,
            "handle_target_skewness": False,
            "datetime_converter": True,
            "pos_tagging_pca": False,  # slow with many categories
            "append_text_sentiment_score": False,
            "tfidf_vectorizer_to_pca": True,  # slow with many categories
            "tfidf_vectorizer": False,
            "rare_feature_processing": True,
            "cardinality_remover": True,
            "categorical_column_embeddings": False,
            "holistic_null_filling": True,  # slow
            "numeric_binarizer_pca": True,
            "onehot_pca": True,
            "category_encoding": True,
            "fill_nulls_static": True,
            "outlier_care": True,
            "delete_outliers": False,
            "remove_collinearity": True,
            "skewness_removal": True,
            "automated_feature_transformation": False,
            "random_trees_embedding": False,
            "clustering_as_a_feature_dbscan": True,
            "clustering_as_a_feature_kmeans_loop": True,
            "clustering_as_a_feature_gaussian_mixture_loop": True,
            "pca_clustering_results": True,
            "svm_outlier_detection_loop": False,
            "autotuned_clustering": False,
            "reduce_memory_footprint": False,
            "scale_data": False,
            "smote": False,
            "automated_feature_selection": True,
            "bruteforce_random_feature_selection": False,  # slow
            "autoencoder_based_oversampling": False,
            "synthetic_data_augmentation": False,
            "final_pca_dimensionality_reduction": False,
            "final_kernel_pca_dimensionality_reduction": False,
            "delete_low_variance_features": True,
            "shap_based_feature_selection": True,
            "delete_unpredictable_training_rows": False,
            "sort_columns_alphabetically": True,
        }

        self.checkpoints = {
            "automatic_type_detection_casting": False,
            "early_numeric_only_feature_selection": False,
            "remove_duplicate_column_names": False,
            "reset_dataframe_index": False,
            "regex_clean_text_data": False,
            "handle_target_skewness": False,
            "holistic_null_filling": True,  # slow
            "iterative_null_imputation": True,
            "fill_infinite_values": True,
            "datetime_converter": False,
            "pos_tagging_pca": True,  # slow with many categories
            "append_text_sentiment_score": True,
            "tfidf_vectorizer_to_pca": True,  # slow with many categories
            "tfidf_vectorizer": True,
            "rare_feature_processing": True,
            "cardinality_remover": True,
            "categorical_column_embeddings": False,
            "delete_high_null_cols": True,
            "numeric_binarizer_pca": True,
            "onehot_pca": True,
            "category_encoding": True,
            "fill_nulls_static": True,
            "data_binning": True,
            "outlier_care": True,
            "delete_outliers": False,
            "remove_collinearity": True,
            "skewness_removal": True,
            "automated_feature_transformation": True,
            "random_trees_embedding": False,
            "autotuned_clustering": True,
            "clustering_as_a_feature_dbscan": True,
            "clustering_as_a_feature_kmeans_loop": True,
            "clustering_as_a_feature_gaussian_mixture_loop": True,
            "pca_clustering_results": True,
            "svm_outlier_detection_loop": True,
            "reduce_memory_footprint": True,
            "automated_feature_selection": True,
            "bruteforce_random_feature_selection": True,  # slow
            "sort_columns_alphabetically": True,
            "synthetic_data_augmentation": True,
            "scale_data": True,
            "smote": False,
            "autoencoder_based_oversampling": True,
            "final_kernel_pca_dimensionality_reduction": False,
            "final_pca_dimensionality_reduction": False,
            "delete_low_variance_features": False,
            "shap_based_feature_selection": True,
            "delete_unpredictable_training_rows": True,
        }
        self.checkpoint_reached = {}
        for key in self.checkpoints.keys():
            self.checkpoint_reached[key] = False

        self.preprocessing_funcs = None

        self.blueprint_step_selection_nlp_transformers = {
            "train_test_split": True,
            "regex_clean_text_data": False,
            "rare_feature_processing": True,
            "sort_columns_alphabetically": True,
            "random_synonym_replacement": False,
            "oversampling": False,
            "synonym_language": "english",
        }
        self.feature_selection_backend = "lgbm"

        self.pos_tagging_languages = {
            # https://spacy.io/models/es
            "german": "de_core_news_sm",
            "french": "fr_core_news_sm",
            "english": "en_core_web_sm",
            "italian": "it_core_news_sm",
            "portugese": "pt_core_news_sm",
            "spanish": "es_core_news_sm",
        }

        """
        NLTK compatible languages
        ['hungarian',
         'swedish',
         'kazakh',
         'norwegian',
         'finnish',
         'arabic',
         'indonesian',
         'portuguese',
         'turkish',
         'azerbaijani',
         'slovene',
         'spanish',
         'danish',
         'nepali',
         'romanian',
         'greek',
         'dutch',
         'README',
         'tajik',
         'german',
         'english',
         'russian',
         'french',
         'italian']
        """
        self.special_blueprint_algorithms = {
            "ridge": True,
            "elasticnet": True,
            "catboost": True,
            "xgboost": True,
            "ngboost": True,
            "lgbm": True,
            "tabnet": False,
            "vowpal_wabbit": True,
            "sklearn_ensemble": True,
        }
        # store chosen preprocessing settings
        if not preprocess_decisions:
            self.preprocess_decisions = {}
        else:
            self.preprocess_decisions = preprocess_decisions
        self.transformer_model_load_from_path = transformer_model_load_from_path
        self.transformer_model_save_states_path = transformer_model_save_states_path
        self.transformer_settings = {
            "train_batch_size": 32,
            "test_batch_size": 32,
            "pred_batch_size": 32,
            "num_workers": 4,
            "epochs": self.transformer_epochs,  # TODO: Change to 20 again
            "transformer_model_path": self.transformer_model_load_from_path,
            "model_save_states_path": {self.transformer_model_save_states_path},
            "keep_best_model_only": False,
        }

        # automatically determine batch sizes for Tabnet

        rec_batch_size = (len(self.dataframe.index) * 0.8) / 20
        if int(rec_batch_size) % 2 == 0:
            rec_batch_size = int(rec_batch_size)
        else:
            rec_batch_size = int(rec_batch_size) + 1
        if rec_batch_size > 16384:
            rec_batch_size = 16384
            virtual_batch_size = 4096
        else:

            virtual_batch_size = int(rec_batch_size / 4)

        self.tabnet_settings = {
            "batch_size": rec_batch_size,
            "virtual_batch_size": virtual_batch_size,
            "num_workers": 0,
            "max_epochs": 1000,
        }

        self.hyperparameter_tuning_rounds = {
            "xgboost": 100,
            "lgbm": 500,
            "lgbm_focal": 50,
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
            "final_pca_dimensionality_reduction": 50,
        }

        self.hyperparameter_tuning_max_runtime_secs = {
            "xgboost": 2 * 60 * 60,
            "lgbm": 2 * 60 * 60,
            "lgbm_focal": 2 * 60 * 60,
            "tabnet": 2 * 60 * 60,
            "ngboost": 2 * 60 * 60,
            "sklearn_ensemble": 2 * 60 * 60,
            "ridge": 2 * 60 * 60,
            "elasticnet": 2 * 60 * 60,
            "catboost": 2 * 60 * 60,
            "sgd": 2 * 60 * 60,
            "svm": 2 * 60 * 60,
            "svm_regression": 2 * 60 * 60,
            "ransac": 2 * 60 * 60,
            "multinomial_nb": 2 * 60 * 60,
            "bruteforce_random": 2 * 60 * 60,
            "synthetic_data_augmentation": 1 * 60 * 60,
            "autoencoder_based_oversampling": 2 * 60 * 60,
            "final_kernel_pca_dimensionality_reduction": 4 * 60 * 60,
            "final_pca_dimensionality_reduction": 2 * 60 * 60,
        }

        self.feature_selection_sample_size = 100000
        self.hyperparameter_tuning_sample_size = 10000
        self.brute_force_selection_sample_size = 15000
        self.final_pca_dimensionality_reduction_sample_size = 15000
        self.brute_force_selection_base_learner = (
            "double"  # 'lgbm', 'vowpal_wabbit', 'auto
        )

        if self.class_problem == "regression":
            skewness = datasource[self.target_variable].skew(axis=0, skipna=True)
            if skewness < -0.75 or skewness > 0.75:
                self.target_is_skewed = True
            else:
                self.target_is_skewed = False
        else:
            self.target_is_skewed = False
        self.selected_feats = selected_feats
        self.selected_shap_feats = None
        self.cat_encoded = cat_encoded
        self.cat_encoder_model = cat_encoder_model
        self.data_scaled = False
        if not trained_model:
            self.trained_models = {}
        else:
            self.trained_models = trained_model
        self.optuna_studies = {}
        self.predicted_classes = {}
        self.predicted_probs = {}
        self.predicted_values = {}
        self.evaluation_scores = {}
        self.xg_boost_regression = None
        self.xgboost_objective = None
        self.prediction_mode = prediction_mode
        self.best_model = None
        self.excluded = None
        self.detected_col_types = {}
        self.num_dtypes = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]
        if not num_columns:
            num_col_list = []
            for vartype in self.num_dtypes:
                num_cols = self.dataframe.select_dtypes(include=[vartype]).columns
                for col in num_cols:
                    num_col_list.append(col)
            self.num_columns = num_col_list
        else:
            self.num_columns = num_columns
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def __repr__(self):
        return (
            "Central data class holding all information like dataframes, "
            "columns of certain data types, saved models and predictions."
            f"Current target variable:'{self.target_variable}'"
        )

    def __str__(self):
        return f"Current target: {self.target_variable}"

    def get_current_timestamp(self, task=None):
        """
        Prints and return the current timestamp (not timezone aware)
        :param task: Expects a string. Can be used to inject the printed message with context.
        :return: Returns timestamp as string.
        """
        t = time.localtime()
        if task:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"Started {task} at {current_time}.")
        else:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"{current_time}")
        return current_time

    def runtime_warnings(self, warn_about="shap_cpu"):
        """
        This function returns custom warnings for a better user experience.
        :return: warning message
        """
        if warn_about == "shap_cpu":
            warning_message = """Calculating SHAP values for feature importance on CPU might run a long time. To disable
            the calculation set the parameter 'feat_importance' to False. Alternatively the LGBM and Xgboost
            blueprints can be used as well. These run on GPU by default and usually yield better
            classification results as well."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == "long runtime":
            warning_message = """This blueprint has long runtimes. GPU acceleration is only possible for LGBM and Xgboost
            as of now. Also Ngboost is relatively fast even though it can only run on CPU."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == "wrong null algorithm":
            warning_message = """The chosen option does not exist. Currently only "iterative_imputation" and "static"
            exist. Any other declared option will result in not-handling of NULLs and are likely to fail later in the
             pipeline."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == "future_architecture_change":
            warning_message = """The organization of blueprints will change in a future version to better separate NLP
            and non-NLP related preprocessing(!) blueprints. This change is likely to be live with e2eml version 2.0.0
            """
            return warnings.warn(warning_message, DeprecationWarning)
        elif warn_about == "no_nlp_transformer":
            warning_message = """No nlp_transformer_columns have been provided during class instantiation. Some
            NLP related functions only run with this information.."""
            return warnings.warn(warning_message, UserWarning)
        elif warn_about == "duplicate_column_names":
            warning_message = """Duplicate column names have been found and duplicate columns have been removed. Please
            make check, if these columns were fully duplicates or sharing an identical name only.
            """
            return warnings.warn(warning_message, UserWarning)
        elif warn_about == "not_enough_target_class_members":
            warning_message = """Some target classes have less members than allowed. You can ignore this message, if you
            are running a blueprint without NLP transformers.

            In order to create a strong model e2eml splits the data into several folds. Please provide data with at least
             6 class members for each target class. Otherwise the model is likely to fail to a CUDA error on runtime.
             You can use the following function on your dataframe before passing it to e2eml:

            def handle_rarity(all_data, threshold=6, mask_as='miscellaneous', rarity_cols=None, normalize=False):
                if isinstance(rarity_cols, list):
                    for col in rarity_cols:
                        frequencies = all_data[col].value_counts(normalize=normalize)
                        condition = frequencies < threshold
                        mask_obs = frequencies[condition].index
                        mask_dict = dict.fromkeys(mask_obs, mask_as)
                        all_data[col] = all_data[col].replace(mask_dict)
                    del rarity_cols
                else:
                    pass
                return all_data

            Example usage:
            train_df = handle_rarity(train_df, rarity_cols=["your_target_column_name"])

            Important:
            This function modifies the original data. It is recommended to create a copy of your data first.
            """
            return warnings.warn(warning_message, UserWarning)
        else:
            pass

    def check_gpu_support(self, algorithm=None):
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        try:
            if not self.preprocess_decisions["gpu_support"]:
                self.preprocess_decisions["gpu_support"] = {}
        except KeyError:
            self.preprocess_decisions["gpu_support"] = {}
        else:
            pass
        if algorithm == "lgbm":
            self.get_current_timestamp(task="Check LGBM for GPU acceleration.")
            train_data = lightgbm.Dataset(data, label=label)  # noqa: F841
            params = {"num_iterations": 1, "device": "gpu"}
            try:
                # gbm = lightgbm.train(params, train_set=train_data)
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "gpu"
                print("LGBM uses GPU.")
            except Exception:
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "cpu"
                print("LGBM uses CPU.")
        elif algorithm == "xgboost":
            self.get_current_timestamp(task="Check Xgboost for GPU acceleration.")
            D_train = xgb.DMatrix(data, label=label)
            params = {"tree_method": "gpu_hist", "steps": 2}
            try:
                model = xgb.train(params, D_train)
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "gpu_hist"
                print("Xgboost uses GPU.")
            except Exception:
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "exact"
                print("Xgboost uses CPU.")
        elif algorithm == "catboost":
            try:
                model = CatBoostClassifier(iterations=2, task_type="GPU", devices="0:1")
                model.fit(data, label)
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "GPU"
                print("Catboost uses GPU.")
            except Exception:
                self.preprocess_decisions["gpu_support"][f"{algorithm}"] = "CPU"
                print("Catboost uses CPU.")
        else:
            print("No algorithm has been checked for GPU acceleration.")

    def automatic_type_detection_casting(self):
        """
        Loops through the dataframe and detects column types and type casts them accordingly.
        :return: Returns casted dataframe
        """
        if self.prediction_mode:
            self.get_current_timestamp(task="Started column type detection and casting")
            logging.info("Started column type detection and casting.")
            for key in self.detected_col_types:
                if self.detected_col_types[key] == "datetime[ns]":
                    self.dataframe[key] = pd.to_datetime(
                        self.dataframe[key], yearfirst=True
                    )
                else:
                    self.dataframe[key] = self.dataframe[key].astype(
                        self.detected_col_types[key]
                    )
            logging.info("Finished column type detection and casting.")
            return self.dataframe
        else:
            self.get_current_timestamp(task="Started column type detection and casting")
            logging.info("Started column type detection and casting.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            # detect and cast boolean columns
            bool_cols = list(X_train.select_dtypes(["bool"]))
            for col in bool_cols:
                X_train[col] = X_train[col].astype(bool)
                X_test[col] = X_test[col].astype(bool)
                self.detected_col_types[col] = "bool"

            # detect and cast datetime columns
            try:
                no_bool_df = X_train.loc[:, ~X_train.columns.isin(bool_cols)]
                no_bool_cols = no_bool_df.columns.to_list()
            except Exception:
                no_bool_cols = X_train.columns.to_list()
            if not self.date_columns:
                date_columns = []
                # convert date columns from object to datetime type
                for col in no_bool_cols:
                    if col not in self.num_columns:
                        try:
                            X_train[col] = pd.to_datetime(X_train[col], yearfirst=True)
                            X_test[col] = pd.to_datetime(X_test[col], yearfirst=True)
                            date_columns.append(col)
                            self.detected_col_types[col] = "datetime[ns]"
                        except Exception:
                            pass
                self.date_columns = date_columns

            # detect and cast floats
            no_bool_dt_cols = bool_cols + self.date_columns
            no_bool_datetime_df = X_train.loc[:, ~X_train.columns.isin(no_bool_dt_cols)]
            no_bool_datetime_cols = no_bool_datetime_df.columns.to_list()
            for col in no_bool_datetime_cols:
                try:
                    X_train[col] = X_train[col].astype(float)
                    X_test[col] = X_test[col].astype(float)
                    self.detected_col_types[col] = "float"
                except Exception:
                    X_train[col] = X_train[col].astype(str)
                    X_test[col] = X_test[col].astype(str)
                    self.detected_col_types[col] = "object"
            logging.info("Finished column type detection and casting.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def binary_imbalance(self):
        """
        Measures the percentage of minority class. If the minority class consists of less than 3%, inbalance will be flagged.
        This will lead some algorithms to use class weights to adjust for that.
        :return: Updates class attribute binary_unbalanced
        """
        if self.prediction_mode:
            pass
        else:
            if self.class_problem == "binary":
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                total_labels = len(Y_train.index)
                majority_class = Y_train.value_counts().sum()
                minority_class = total_labels - majority_class
                if minority_class / total_labels < 0.03:
                    self.binary_unbalanced = True
                else:
                    self.binary_unbalanced = False
            else:
                pass

    def remove_duplicate_column_names(self):
        """
        Takes the dataframes in the class instance and checks, if column names are duplicate.
        If so, it will reduce the dataframe to non-duplicate column names and raise a warning to prevent the blueprint
        to break at later steps.
        :return: Updates class attributes
        """
        self.get_current_timestamp(task="Checking for duplicate columns")
        logging.info("Start checking for duplicate columns")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            all_columns = self.dataframe.columns.to_list()
            cols_no_duplicates = list(set(all_columns))
            self.dataframe = self.dataframe[cols_no_duplicates].copy()
            logging.info("Finished checking for duplicate columns")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            all_columns = X_train.columns.to_list()
            cols_no_duplicates = list(set(all_columns))
            X_train = X_train[cols_no_duplicates].copy()
            X_test = X_test[cols_no_duplicates].copy()
            if len(all_columns) != len(cols_no_duplicates):
                self.runtime_warnings(warn_about="duplicate_column_names")
            logging.info("Finished checking for duplicate columns")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def reset_dataframe_index(self):
        """
        All indices will be reset and indices will be dropped. This shall prevent a breaking blueprint by duplicates
        in the intex (i.e. after concatinating multiples dataframes before).
        :return: Updates class instance.
        """
        self.get_current_timestamp("Reset dataframe index.")
        logging.info("Started resetting dataframe.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            self.dataframe = self.dataframe.reset_index(drop=True)
            logging.info("Finished resetting dataframe.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            X_train[self.target_variable] = Y_train
            X_test[self.target_variable] = Y_test

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            Y_train = X_train[self.target_variable]
            Y_test = X_test[self.target_variable]

            X_train = X_train.drop(self.target_variable, axis=1)
            X_test = X_test.drop(self.target_variable, axis=1)
            logging.info("Finished resetting dataframe.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def wrap_test_train_to_dict(self, X_train, X_test, Y_train, Y_test):
        """
        Takes in X_train & X_test parts and updates the class instance dictionary.
        :param X_train: Dataframe
        :param X_test: Dataframe
        :param Y_train: Pandas Series
        :param Y_test: Pandas Series
        :return: Class dictionary
        """
        if self.prediction_mode:
            logging.info("Skipped wrapping dataframe dict due to prediction mode.")
            pass
        else:
            logging.info("Start wrapping dataframe dictionary")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.df_dict = {
                "X_train": X_train,
                "X_test": X_test,
                "Y_train": Y_train,
                "Y_test": Y_test,
            }
            logging.info("Finished wrapping dataframe dictionary")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            del (X_train,)
            del (X_test,)
            del (Y_train,)
            del Y_test
            _ = gc.collect()
            return self.df_dict

    def unpack_test_train_dict(self):
        """
        This function takes in the class dictionary holding test and train split and unpacks it.
        :return: X_train, X_test as dataframes. Y_train, Y_test as Pandas series.
        """
        logging.info("Start unpacking data dictionary")
        X_train, X_test, Y_train, Y_test = (
            self.df_dict["X_train"],
            self.df_dict["X_test"],
            self.df_dict["Y_train"],
            self.df_dict["Y_test"],
        )
        logging.info("Unpacking of data dictionary finished.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        return X_train, X_test, Y_train, Y_test

    def np_array_wrap_test_train_to_dict(self, Y_train, Y_test):
        """
        Takes in X_train & X_test parts and updates the class instance dictionary.
        :param Y_train: Numpy array
        :param Y_test: Numpy array
        :return: Class dictionary
        """
        if self.prediction_mode:
            logging.info("Wrapping Numpy dict skipped due to prediction mode.")
            pass
        else:
            logging.info("Start wrapping Numpy dict.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.df_dict = {"Y_train": Y_train, "Y_test": Y_test}
            logging.info("Finished wrapping Numpy dict.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.df_dict

    def np_array_unpack_test_train_dict(self):
        """
        This function takes in the class dictionary holding test and train split and unpacks it.
        :return: X_train, X_test as dataframes. Y_train, Y_test as numpy array.
        """
        logging.info("Start unpacking Numpy dict.")
        Y_train, Y_test = self.df_dict["Y_train"], self.df_dict["Y_test"]
        logging.info("Finished unpacking Numpy dict.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        return Y_train, Y_test

    def save_load_model_file(
        self,
        model_object=None,
        model_path=None,
        algorithm=None,
        algorithm_variant="none",
        file_type=".dat",
        action="save",
        clean=True,
    ):
        """
        Function to save and load class instances. This function shall be used to save whole blueprints and to
        reload them.
        :param model_object: The blueprint class instance to be saved.
        :param model_path: Expects a string. The path to save the model to or load from.
        :param algorithm: Expects a string. Used to name the final stored file.
        :param algorithm_variant: Expects a string. Used to name the final stored file.
        :param file_type: File type to be saved as. Default ".dat"
        :param action: Chose 'save' or 'load'.
        :param clean: Expects a boolean. If True, deletes the provided class instance instantly after saving.
        :return: When action is 'load', returns the loaded blueprint class instance.
        """
        if self.save_models_path:
            path = self.save_models_path
        elif model_path:
            path = model_path
        else:
            pass
        full_path = path + "_" + algorithm + "_" + algorithm_variant + "_" + file_type

        if action == "save":
            self.get_current_timestamp(task="Save blueprint instance.")
            filehandler = open(full_path, "wb")
            pickle.dump(model_object, filehandler)
            filehandler.close()
            if clean:
                del model_object
                _ = gc.collect()
        elif action == "load":
            self.get_current_timestamp(task="Load blueprint instance.")
            filehandler = open(full_path, "rb")
            model_object = pickle.load(filehandler)
            filehandler.close()
            return model_object
        else:
            pass

    def reduce_mem_usage(self, df):
        """
        Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        :param df: Expects a Pandas dataframe.
        :return: Returns downcasted dataframe.
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype("category")

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
        return df

    def reduce_memory_footprint(self):
        """
        Takes a dataframe and downcasts columns if possible.
        :return: Returns downcasted dataframe.
        """
        self.get_current_timestamp(task="Reduce memory footprint of dataframe")
        logging.info("Started reducing memory footprint.")
        if self.prediction_mode:
            self.dataframe = self.reduce_mem_usage(self.dataframe)
            logging.info("Finished reducing memory footprint.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = self.reduce_mem_usage(X_train)
            X_test = self.reduce_mem_usage(X_test)
            logging.info("Finished reducing memory footprint.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def sort_columns_alphabetically(self):
        """
        Takes a dataframe and sorts its columns alphabetically. This increases pipelines robustness in cases
        where the input data might have been changed in order.
        :return: Updates class instance. Returns dictionary.
        """
        self.get_current_timestamp(task="Sort columns alphabetically")
        if self.prediction_mode:
            logging.info("Started sorting columns alphabetically.")
            self.dataframe = self.dataframe.sort_index(axis=1)
            logging.info("Finished sorting columns alphabetically.")
        else:
            logging.info("Started sorting columns alphabetically.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = X_train.sort_index(axis=1)
            X_test = X_test.sort_index(axis=1)
            logging.info("Finished sorting columns alphabetically.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def calc_scale_pos_weight(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        try:
            pos_labels = Y_train.sum()
        except Exception:
            pos_labels = np.sum(Y_train)

        try:
            neg_labels = len(X_train.index) - pos_labels
        except Exception:
            neg_labels = len(Y_train) - np.sum(Y_train)

        scale_pos_weight = neg_labels / pos_labels
        return scale_pos_weight

    def label_encoder_decoder(self, target, mode="fit"):
        """
        Takes a Pandas series and encodes string-based labels to numeric values. Flags previously unseen
        values with -1.
        :param target: Expects Pandas Series.
        :param mode: 'Chose' fit to create label encoding dictionary and 'transform' the labels. Chose 'transform'
        to encode labels based on already created dictionary.
        :return: Returns Pandas Series.
        """
        self.get_current_timestamp(task="Execute label encoding")
        logging.info("Started label encoding.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        def label_encoder_fit(pandas_series):
            pandas_series = pandas_series.astype("category")
            col = pandas_series.name
            try:
                pandas_series = pandas_series.to_frame()
            except Exception:
                pass
            values = pandas_series[col].unique()
            cat_mapping = {}
            for label, cat in enumerate(values):
                cat_mapping[cat] = label
            return cat_mapping

        def label_encoder_transform(pandas_series, mapping):
            pandas_series = pandas_series.astype("category")
            col = pandas_series.name
            try:
                pandas_series = pandas_series.to_frame()
            except Exception:
                pass
            mapping = self.preprocess_decisions["label_encoder_mapping"]
            pandas_series[col] = pandas_series[col].apply(lambda x: mapping.get(x, 999))
            # pandas_series = pandas_series[col]
            return pandas_series

        if self.prediction_mode:
            target = label_encoder_transform(
                target, self.preprocess_decisions["label_encoder_mapping"]
            )
        else:
            if mode == "fit":
                cat_mapping = label_encoder_fit(target)
                self.preprocess_decisions["label_encoder_mapping"] = cat_mapping
            else:
                pass
            target = label_encoder_transform(
                target, self.preprocess_decisions["label_encoder_mapping"]
            )
        self.labels_encoded = True
        if self.class_problem == "binary" or self.class_problem == "multiclass":
            target = target[self.target_variable].astype(int)
        elif self.class_problem == "regression":
            target = target[self.target_variable].astype(float)
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        return target

    def label_encoder_reverse_transform(self, pandas_series):
        col = pandas_series.name
        try:
            pandas_series = pandas_series.to_frame()
        except Exception:
            pass
        reverse_mapping = {
            value: key
            for key, value in self.preprocess_decisions["label_encoder_mapping"].items()
        }
        pandas_series = pandas_series.replace({col: reverse_mapping})
        return pandas_series

    def check_max_sentence_length(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            text_columns = self.nlp_transformer_columns
            sentence_length = X_train[text_columns].apply(
                lambda x: np.max([len(w) for w in x.split()])
            )
            if "nlp_transformers" in self.preprocess_decisions:
                pass
            else:
                self.preprocess_decisions["nlp_transformers"] = {}
            self.preprocess_decisions["nlp_transformers"][
                "max_sentence_len"
            ] = sentence_length.max()

    def data_scaling(self, scaling="minmax"):
        """
        Scales the data using the chosen scaling algorithm.
        :param scaling: Chose 'minmax'.
        :return: Returns scaled dataframes
        """
        self.get_current_timestamp(task="Scale data")
        if self.prediction_mode:
            logging.info("Started data scaling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            dataframe_cols = self.dataframe.columns
            if scaling == "minmax":
                scaler = self.preprocess_decisions["scaling"]
                scaler.fit(self.dataframe)
                scaler.transform(self.dataframe)
            self.dataframe = pd.DataFrame(self.dataframe, columns=dataframe_cols)
            self.data_scaled = True
            logging.info("Finished data scaling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe, self.data_scaled
        else:
            logging.info("Started data scaling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train_cols = X_train.columns
            if scaling == "minmax":
                scaler = MinMaxScaler()
                scaler.fit(X_train)
                scaler.transform(X_train)
                scaler.transform(X_test)
                self.preprocess_decisions["scaling"] = scaler
            X_train = pd.DataFrame(X_train, columns=X_train_cols)
            X_test = pd.DataFrame(X_test, columns=X_train_cols)
            self.data_scaled = True
            logging.info("Finished data scaling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            del scaler
            _ = gc.collect()
            return (
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test),
                self.data_scaled,
                self.preprocess_decisions,
            )

    def skewness_removal(self, overwrite_orig_col=False):
        """
        Loops through the in-class stored dataframe columns and checks the skewness. If skewness exceeds a certain threshold,
        executes log transformation.
        :param overwrite_orig_col: If True, replace the original column with its unskewed counterpart. Otherwise append
        an unskewed counterpart as new column to the dataframe.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Remove skewness")
        if self.prediction_mode:
            logging.info("Started skewness removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            for col in self.preprocess_decisions["skewed_columns"]:
                log_array = np.log1p(self.dataframe[col])
                log_array[np.isfinite(log_array) == False] = 0  # noqa: E712
                if overwrite_orig_col:
                    self.dataframe[col] = log_array
                else:
                    self.dataframe[f"{col}_unskewed"] = log_array
            logging.info("Finished skewness removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            logging.info("Started skewness removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            skewness = X_train.skew(axis=0, skipna=True)
            left_skewed = skewness[skewness < -0.5].index.to_list()
            right_skewed = skewness[skewness > 0.5].index.to_list()
            skewed = left_skewed + right_skewed
            for col in X_train[skewed].columns:
                log_array = np.log1p(X_train[col])
                log_array[np.isfinite(log_array) == False] = 0  # noqa: E712
                if overwrite_orig_col:
                    X_train[col] = log_array
                else:
                    X_train[f"{col}_unskewed"] = log_array

                log_array = np.log1p(X_test[col])
                log_array[np.isfinite(log_array) == False] = 0  # noqa: E712
                if overwrite_orig_col:
                    X_test[col] = log_array
                else:
                    X_test[f"{col}_unskewed"] = log_array
            logging.info("Finished skewness removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.preprocess_decisions["skewed_columns"] = skewed
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def target_skewness_handling(self, mode="fit", preds_to_reconvert=None):
        """
        Loops through the in-class stored dataframe columns and checks the skewness. If skewness exceeds a certain threshold,
        executes log transformation.
        :param overwrite_orig_col: If True, replace the original column with its unskewed counterpart. Otherwise append
        an unskewed counterpart as new column to the dataframe.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Target skewness handling")
        logging.info("Started target skewness handling.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            if (
                self.target_is_skewed
                and self.blueprint_step_selection_non_nlp["handle_target_skewness"]
            ):
                if mode == "fit":
                    pass
                else:
                    preds_to_reconvert_reverted = np.expm1(preds_to_reconvert)
                    return preds_to_reconvert_reverted
            else:
                pass
            return preds_to_reconvert
        else:
            if (
                self.target_is_skewed
                and self.blueprint_step_selection_non_nlp["handle_target_skewness"]
            ):
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                if mode == "fit":
                    skewness = Y_train.skew(axis=0, skipna=True)
                    if skewness < -0.5 or skewness > 0.5:
                        Y_train = np.log1p(Y_train)
                        Y_train[np.isfinite(Y_train) == False] = 0  # noqa: E712
                        Y_test = np.log1p(Y_test)
                        Y_test[np.isfinite(Y_test) == False] = 0  # noqa: E712
                else:
                    Y_train = np.expm1(Y_train)
                    Y_test = np.expm1(Y_test)
                logging.info("Finished target skewness handling.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            else:
                pass

    def create_folds(self, data, target, num_splits=5, mode="advanced"):
        if self.prediction_mode:
            pass
        else:
            if mode == "simple":
                data["kfold"] = data.index % num_splits
            else:
                # we create a new column called kfold and fill it with -1
                data["kfold"] = -1

                # the next step is to randomize the rows of the data
                data = data.sample(frac=1).reset_index(drop=True)
                print(data.info())

                # calculate number of bins by Sturge's rule
                # I take the floor of the value, you can also
                # just round it
                num_bins = int(np.floor(1 + np.log2(len(data))))
                # bin targets
                data.loc[:, "bins"] = pd.cut(data[target], bins=num_bins, labels=False)
                print(data.info())
                # initiate the kfold class from model_selection module
                kf = model_selection.StratifiedKFold(n_splits=num_splits)
                # fill the new kfold column
                # note that, instead of targets, we use bins!
                for f, (_t, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
                    data.loc[v_, "kfold"] = f
                # drop the bins column
                data = data.drop("bins", axis=1)
                # return dataframe with folds
            return data

    def reset_test_train_index(self, drop_target=False):
        if self.prediction_mode:
            self.dataframe = self.dataframe.reset_index(drop=True)
        else:
            # index shuffling
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train[self.target_variable] = Y_train
            X_test[self.target_variable] = Y_test
            all_data = pd.concat([X_train, X_test])
            all_data = self.create_folds(all_data, self.target_variable)
            X_train = all_data[all_data["kfold"] != 0].reset_index(drop=True)
            X_test = all_data[all_data["kfold"] == 0].reset_index(drop=True)
            Y_train = X_train[self.target_variable]
            Y_test = X_test[self.target_variable]
            X_train.drop("kfold", axis=1)
            X_test.drop("kfold", axis=1)
            if drop_target:
                X_train.drop(self.target_variable, axis=1)
                X_test.drop(self.target_variable, axis=1)
            else:
                pass
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def train_test_split(
        self, how="cross", split_by_col=None, split_date=None, train_size=0.70
    ):
        """
        This method splits the dataframe either as a simple or as a time split.
        :param how: 'cross' for cross validation, 'time' for time validation.
        :param split_by_col: Chose column to be used for split. For time validation only.
        :param split_date: Chose exact date to split. Test dataframe is equal or greater than provided date.
        :param train_size: Chose how much percentage the train dataframe will have. For cross validation only.
        :return: X_train, X_test, Y_train, Y_test
        """
        self.get_current_timestamp(task="Execute test train split")
        if self.prediction_mode:
            logging.info("Skipped test train split due to prediction mode.")
        elif how == "cross":
            logging.info("Started test train split.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.check_target_class_distribution()

            try:
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                    self.dataframe,
                    self.dataframe[self.target_variable],
                    train_size=train_size,
                    random_state=42,
                    stratify=self.dataframe[self.target_variable],
                )
            except Exception:
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                    self.dataframe,
                    self.dataframe[self.target_variable],
                    train_size=train_size,
                    random_state=42,
                )
            try:
                Y_train = Y_train.astype(float)
                Y_test = Y_test.astype(float)
            except Exception:
                Y_train = self.label_encoder_decoder(Y_train, mode="fit")
                Y_test = self.label_encoder_decoder(Y_test, mode="transform")
            del X_train[self.target_variable]
            del X_test[self.target_variable]
            _ = gc.collect()
            logging.info("Finished test train split.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif how == "time":
            logging.info("Started test train split.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            if self.source_format == "numpy array":
                length = self.dataframe.size
                train_length = int(length * train_size)
                test_length = length - train_length
                Y_train, Y_test = (
                    self.dataframe[:train_length],
                    self.dataframe[:test_length],
                )
                logging.info("Finished test train split.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.np_array_wrap_test_train_to_dict(Y_train, Y_test)
            elif self.source_format == "Pandas dataframe":
                length = len(self.dataframe.index)
                train_length = int(length * 0.80)
                test_length = length - train_length
                if not split_by_col:
                    self.dataframe = self.dataframe.sort_index()
                elif split_by_col:
                    self.dataframe = self.dataframe.sort_values(by=[split_by_col])
                else:
                    pass
                if split_date:
                    X_train = self.dataframe[(self.dataframe.split_by_col < split_date)]
                    X_test = self.dataframe[(self.dataframe.split_by_col >= split_date)]
                else:
                    X_train = self.dataframe.head(train_length)
                    X_test = self.dataframe.tail(test_length)
                Y_train = X_train[self.target_variable]
                Y_test = X_test[self.target_variable]
                del X_train[self.target_variable]
                del X_test[self.target_variable]
                logging.info("Finished test train split.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        else:
            logging.warning("No split method provided.")
            raise Exception("Please provide a split method.")

    def check_target_class_distribution(self):
        if self.prediction_mode:
            pass
        else:
            min_target_train = self.dataframe[self.target_variable].value_counts().min()
            if min_target_train < 7:
                self.runtime_warnings(warn_about="not_enough_target_class_members")
            else:
                pass

    def set_random_seed(self, seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def apply_k_folds(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = self.create_folds(X_train, num_splits=4)
            X_test["kfold"] = 0
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def data_binning(self, nb_bins=10):
        """
        Takes numerical columns and splits them into desired number of bins. Bins will be attached as
        new columns to the dataframe.
        :param nb_bins: Takes a positive integer.
        :return: Updates class instance.
        """
        self.get_current_timestamp(task="Execute numerical binning")

        def random_noise(a_series, noise_reduction=1000000):
            return (
                np.random.random(len(a_series)) * a_series.std() / noise_reduction
            ) - (a_series.std() / (2 * noise_reduction))

        def binning_on_data(dataframe, cols_to_bin=None):
            num_columns = cols_to_bin.select_dtypes(  # noqa: F821
                include=[vartype]  # noqa: F821
            ).columns  # noqa: F821
            for col in num_columns:
                dataframe[str(col) + "_binned"] = pd.cut(
                    dataframe[col].replace(np.inf, np.nan).dropna(),
                    bins=nb_bins,
                    labels=False,
                )
                dataframe[str(col) + "_equal_binned"] = pd.cut(
                    dataframe[col] + random_noise(dataframe[col]), nb_bins, labels=False
                )
                self.new_sin_cos_col_names.append(str(col) + "_binned")
                self.new_sin_cos_col_names.append(str(col) + "_equal_binned")
            del num_columns
            _ = gc.collect()
            return dataframe

        logging.info("Start numerical binning.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            for _vartype in self.num_dtypes:
                filtered_columns = self.dataframe.loc[
                    :, ~self.dataframe.columns.isin(self.new_sin_cos_col_names)
                ]
                self.dataframe = binning_on_data(
                    self.dataframe, cols_to_bin=filtered_columns
                )
            logging.info("Finished numerical binning.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if len(self.num_dtypes) > 0:
                self.blueprint_step_selection_non_nlp["data_binning"] = False
            else:
                for _vartype in self.num_dtypes:
                    filtered_columns = X_train.loc[
                        :, ~X_train.columns.isin(self.new_sin_cos_col_names)
                    ]

                    X_train = binning_on_data(X_train, cols_to_bin=filtered_columns)
                    X_test = binning_on_data(X_test, cols_to_bin=filtered_columns)
                logging.info("Finished numerical binning.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def cardinality_remover(self, threshold=100):
        """
        Loops through all columns and delete columns with cardinality higher than defined threshold.
        :param threshold: integer of any size
        :return:Cleaned dataframe.
        """
        self.get_current_timestamp(task="Remove cardinality")

        def remove_high_cardinality(df, threshold=threshold, cols_to_delete=None):
            if not cols_to_delete:
                deleted_columns = []
                cat_columns = df.select_dtypes(include=["object"]).columns
                for col in cat_columns:
                    cardinality = df[col].nunique()
                    if cardinality >= threshold:
                        df = df.drop([col], axis=1)
                        deleted_columns.append(col)
                    else:
                        pass
            else:
                deleted_columns = cols_to_delete
                for col in deleted_columns:
                    df = df.drop([col], axis=1)
            return df, deleted_columns

        logging.info("Start cardinality removal.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            threshold = self.preprocess_decisions["cardinality_threshold"]
            (
                self.dataframe,
                self.preprocess_decisions["cardinality_deleted_columns"],
            ) = remove_high_cardinality(
                self.dataframe,
                cols_to_delete=self.preprocess_decisions["cardinality_deleted_columns"],
            )
            logging.info("Finished cardinality removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            (
                X_train,
                self.preprocess_decisions["cardinality_deleted_columns"],
            ) = remove_high_cardinality(X_train, threshold=threshold)
            (
                X_test,
                self.preprocess_decisions["cardinality_deleted_columns"],
            ) = remove_high_cardinality(
                df=X_test,
                cols_to_delete=self.preprocess_decisions["cardinality_deleted_columns"],
            )
            self.preprocess_decisions["cardinality_threshold"] = threshold
            logging.info("Finished cardinality removal.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def categorical_column_embeddings(self):
        self.get_current_timestamp(task="Create categorical column embeddings.")
        logging.info("Start categorical column embeddings.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Downloading language model for the spaCy POS tagger\n"
                "(don't worry, this will only happen once)"
            )
            from spacy.cli import download

            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        def get_spacy_embeddings(df, text_col):
            df[text_col] = df[text_col].astype(str)
            df["tokenized"] = df[text_col].apply(nlp)
            df["sent_vectors"] = df["tokenized"].apply(
                lambda sent: np.mean(
                    [token.vector for token in sent if not token.is_stop]
                )
            )
            return df["sent_vectors"]

        if self.prediction_mode:
            cat_columns = self.preprocess_decisions["spacy_embedding_cols"]
            for col in cat_columns:
                self.dataframe[col] = self.dataframe[col] + ". "
                self.dataframe[f"Spacy_embeddings_{col}"] = get_spacy_embeddings(
                    self.dataframe, col
                )

            self.dataframe["Spacy_embeddings"] = self.dataframe[cat_columns].sum(axis=1)
            self.dataframe["Spacy_embeddings"] = get_spacy_embeddings(
                self.dataframe, "Spacy_embeddings"
            )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            cat_columns = X_train.select_dtypes(include=["object"]).columns.to_list()
            for col in cat_columns:
                X_train[col] = X_train[col] + ". "
                X_test[col] = X_test[col] + ". "
                X_train[f"Spacy_embeddings_{col}"] = get_spacy_embeddings(X_train, col)
                X_test[f"Spacy_embeddings_{col}"] = get_spacy_embeddings(X_test, col)

            X_train["Spacy_embeddings"] = X_train[cat_columns].sum(axis=1)
            X_test["Spacy_embeddings"] = X_test[cat_columns].sum(axis=1)

            X_train["Spacy_embeddings"] = get_spacy_embeddings(
                X_train, "Spacy_embeddings"
            )
            X_test["Spacy_embeddings"] = get_spacy_embeddings(
                X_test, "Spacy_embeddings"
            )

            self.preprocess_decisions["spacy_embedding_cols"] = cat_columns
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            logging.info("Finished categorical column embeddings.")

        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def rare_feature_processor(
        self, threshold=0.005, mask_as="miscellaneous", rarity_cols=None, normalize=True
    ):
        """
        Loops through categorical columns and identifies categories, which occur less than the
        given threshold. These features will be grouped together as defined by mask_as parameter.
        :param threshold: Minimum share of categories to be not grouped as misc. Takes a float between 0 and 1.
        :param mask_as: Group name of grouped rare features.
        :return: Updates class attributes
        """
        self.get_current_timestamp("Handle rare features")

        def handle_rarity(
            all_data,
            threshold=threshold,
            mask_as=mask_as,
            rarity_cols=rarity_cols,
            normalize=normalize,
        ):
            if isinstance(rarity_cols, list):
                for col in rarity_cols:
                    frequencies = all_data[col].value_counts(normalize=normalize)
                    condition = frequencies < threshold
                    mask_obs = frequencies[condition].index
                    mask_dict = dict.fromkeys(mask_obs, mask_as)
                    all_data[col] = all_data[col].replace(
                        mask_dict
                    )  # or you could make a copy not to modify original data
                del rarity_cols
                _ = gc.collect()
            return all_data

        logging.info("Start rare feature processing.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            threshold = self.preprocess_decisions["rare_feature_threshold"]
            self.dataframe = handle_rarity(
                self.dataframe, threshold, mask_as, rarity_cols, normalize
            )
            logging.info("Finished rare feature processing.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = handle_rarity(X_train, threshold, mask_as, rarity_cols, normalize)
            X_test = handle_rarity(X_test, threshold, mask_as, rarity_cols, normalize)
            self.preprocess_decisions["rare_feature_threshold"] = threshold
            logging.info("Finished rare feature processing.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def auto_tuned_clustering(self):
        """
        Takes a dataframe and optimizes for best clustering hyperparameters and columns to be used.
        :return: Updates class attributes/dataframes.
        """
        self.get_current_timestamp("Start autotuned kmeans clustering.")

        algorithm = "autotuned_clustering"
        logging.info("Start adding autotuned clusters as additional features.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        if not self.data_scaled:
            self.data_scaling()

        if self.prediction_mode:
            chosen_cols = self.preprocess_decisions["autotuned_cluster_pcolumns"]
            kmeans_parameters = self.preprocess_decisions[
                "autotuned_cluster_hyperparameters"
            ]

            # cluster based on all data
            kmeans = KMeans(
                n_clusters=kmeans_parameters["clusters"],
                n_init=kmeans_parameters["n_init"],
                tol=kmeans_parameters["tol"],
                max_iter=kmeans_parameters["max_iter"],
            )
            kmeans.fit(self.dataframe[chosen_cols])
            y_kmeans = kmeans.predict(self.dataframe[chosen_cols])
            self.dataframe[algorithm] = y_kmeans
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            if self.hyperparameter_tuning_sample_size > len(X_train.index):
                cluster_sample_size = len(X_train.index)
            else:
                cluster_sample_size = self.hyperparameter_tuning_sample_size

            if 1000 > len(X_train.index):
                eval_sample_size = len(X_train.index)
            else:
                eval_sample_size = 2000

            X_train_cluster_sample = X_train.sample(
                cluster_sample_size, random_state=42
            )
            X_train_eval_sample = X_train.sample(eval_sample_size, random_state=42)

            try:
                del X_train[self.target_variable]
            except KeyError:
                pass

            def objective(trial):
                param = {}
                for col in X_train_cluster_sample.columns:
                    param[col] = trial.suggest_int(col, 0, 1)
                temp_features = []
                for k, v in param.items():
                    if v == 1:
                        temp_features.append(k)
                    else:
                        pass
                param["clusters"] = (trial.suggest_int("clusters", 2, 30),)
                param["max_iter"] = (trial.suggest_int("max_iter", 10, 20),)
                param["n_init"] = (trial.suggest_int("n_init", 10, 500),)
                param["tol"] = trial.suggest_loguniform("tol", 1e-5, 1e-1)
                kmeans = KMeans(
                    n_clusters=param["clusters"][0],
                    n_init=param["n_init"][0],
                    tol=param["tol"],
                    max_iter=param["max_iter"][0],
                )
                try:
                    kmeans.fit(X_train_cluster_sample[temp_features])
                    y_kmeans = kmeans.predict(X_train_eval_sample[temp_features])
                    s_score = silhouette_score(
                        X_train_eval_sample[temp_features], y_kmeans
                    )
                except Exception:
                    s_score = 0
                return s_score

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name="autotuned kmeans"
            )
            study.optimize(
                objective, n_trials=50, gc_after_trial=True, show_progress_bar=True
            )
            try:
                # fig = optuna.visualization.plot_optimization_history(study)
                # fig.show()
                fig = optuna.visualization.plot_param_importances(study)
                fig.show()
            except ZeroDivisionError:
                pass

            chosen_cols = []
            kmeans_parameters = {}
            for key, value in study.best_trial.params.items():
                if key in X_train.columns.to_list() and value == 1:
                    chosen_cols.append(key)
                elif key not in X_train.columns.to_list():
                    kmeans_parameters[key] = value
                else:
                    pass
            self.preprocess_decisions["autotuned_cluster_pcolumns"] = chosen_cols
            self.preprocess_decisions[
                "autotuned_cluster_hyperparameters"
            ] = kmeans_parameters

            # cluster based on all data
            kmeans = KMeans(
                n_clusters=kmeans_parameters["clusters"],
                n_init=kmeans_parameters["n_init"],
                tol=kmeans_parameters["tol"],
                max_iter=kmeans_parameters["max_iter"],
            )
            kmeans.fit(X_train[chosen_cols])
            y_kmeans = kmeans.predict(X_train[chosen_cols])
            X_train[algorithm] = y_kmeans
            kmeans.fit(X_test[chosen_cols])
            y_kmeans = kmeans.predict(X_test[chosen_cols])
            X_test[algorithm] = y_kmeans

            logging.info("Finished adding autotuned clusters as additional features.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            optuna.logging.set_verbosity(optuna.logging.INFO)
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def delete_low_variance_features(self):
        """
        Takes a dataframe removes columns with very low variance.
        :return: Updates class attributes/dataframes.
        """
        self.get_current_timestamp("Start deleting low variance features")
        logging.info("Start deleting low variance features.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if not self.data_scaled:
            self.data_scaling()

        if self.prediction_mode:
            variable = self.preprocess_decisions[
                "delete_low_variance_features_columns_left"
            ]
            self.dataframe = self.dataframe[variable]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            variance = X_train.var()
            columns = X_train.columns.to_list()
            print(f"Features before low variance deletion: {len(columns)}")
            variable = []

            for i in range(0, len(variance)):
                if variance[i] >= 0.006:  # setting the threshold as 1%
                    variable.append(columns[i])

            X_train = X_train[variable]
            X_test = X_test[variable]

            print(f"Features before low variance deletion: {len(variable)}")

            self.preprocess_decisions[
                "delete_low_variance_features_columns_left"
            ] = variable

            features_dropped = list(set(columns) - set(variable))
            print("The following features have been dropped....:")
            for col in features_dropped:
                print(col)
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            logging.info("Finished deleting low variance features.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def clustering_as_a_feature(  # noqa: C901
        self, algorithm="dbscan", nb_clusters=2, eps=0.3, n_jobs=-1, min_samples=50
    ):
        """
        Takes the numerical columns of a dataframe and performs clustering via the chosen algorithm.
        Appends the clusters as a new column.
        :param algorithm: Chose 'dbscan' or 'gaussian_clusters'
        :param nb_clusters: Takes an integer of 2 or higher.
        :param eps: Epsilon (only needed for DBSCAN). Defines the distance clusters can be apart from each other.
        :param n_jobs: How many cores to use. Chose -1 for all cores.
        :param min_samples: Minimum number of samples required to form a cluster.
        :return: Returns the modified dataframe.
        """
        self.get_current_timestamp("Execute clustering as a feature")

        if self.rapids_acceleration:
            import cudf
            from cuml import DBSCAN as RapidsDBSCAN
            from cuml import KMeans as RapidsKMeans

            def add_dbscan_clusters(
                dataframe, eps=eps, n_jobs=n_jobs, min_samples=min_samples
            ):
                dataframe_red = dataframe.loc[
                    :, dataframe.columns.isin(self.num_columns)
                ].copy()
                dataframe_red = cudf.from_pandas(dataframe_red)
                db = RapidsDBSCAN(eps=eps, min_samples=min_samples).fit(dataframe_red)
                labels = db.labels_
                dataframe[f"dbscan_cluster_{eps}"] = labels
                del db
                del labels
                _ = gc.collect()
                dataframe = dataframe_red.to_pandas()
                return dataframe

            def add_gaussian_mixture_clusters(dataframe, n_components=nb_clusters):
                dataframe = cudf.from_pandas(dataframe)
                kmeans = RapidsKMeans(
                    n_clusters=n_components, random_state=42, n_init=20, max_iter=500
                )
                kmeans.fit(dataframe)
                kmeans_clusters = kmeans.predict(dataframe)
                dataframe[f"gaussian_clusters_{n_components}"] = kmeans_clusters
                del kmeans
                del kmeans_clusters
                _ = gc.collect()
                dataframe = dataframe.to_pandas()
                return dataframe

            def add_kmeans_clusters(dataframe, n_components=nb_clusters):
                dataframe = cudf.from_pandas(dataframe)
                kmeans = RapidsKMeans(
                    n_clusters=n_components, random_state=42, n_init=20, max_iter=500
                )
                kmeans.fit(dataframe)
                kmeans_clusters = kmeans.predict(dataframe)
                dataframe[f"kmeans_clusters_{n_components}"] = kmeans_clusters
                del kmeans
                del kmeans_clusters
                _ = gc.collect()
                dataframe = dataframe.to_pandas()
                return dataframe

        else:

            def add_dbscan_clusters(
                dataframe, eps=eps, n_jobs=n_jobs, min_samples=min_samples
            ):
                dataframe_red = dataframe.loc[
                    :, dataframe.columns.isin(self.num_columns)
                ].copy()
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(dataframe_red)
                labels = db.labels_
                dataframe[f"dbscan_cluster_{eps}"] = labels
                del db
                del labels
                _ = gc.collect()
                return dataframe

            def add_gaussian_mixture_clusters(dataframe, n_components=nb_clusters):
                gaussian = GaussianMixture(n_components=n_components)
                gaussian.fit(dataframe)
                gaussian_clusters = gaussian.predict(dataframe)
                dataframe[f"gaussian_clusters_{n_components}"] = gaussian_clusters
                del gaussian
                del gaussian_clusters
                _ = gc.collect()
                return dataframe

            def add_kmeans_clusters(dataframe, n_components=nb_clusters):
                kmeans = KMeans(
                    n_clusters=n_components, random_state=42, n_init=20, max_iter=500
                )
                kmeans.fit(dataframe)
                kmeans_clusters = kmeans.predict(dataframe)
                dataframe[f"kmeans_clusters_{n_components}"] = kmeans_clusters
                del kmeans
                del kmeans_clusters
                _ = gc.collect()
                return dataframe

        logging.info("Start adding clusters as additional features.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if not self.data_scaled:
            self.data_scaling()
        if algorithm == "dbscan":
            if self.prediction_mode:
                try:
                    self.dataframe = add_dbscan_clusters(self.dataframe)
                except ValueError:
                    self.dataframe[f"dbscan_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                try:
                    X_train = add_dbscan_clusters(X_train)
                    X_test = add_dbscan_clusters(X_test)
                except ValueError:
                    X_train[f"dbscan_clusters_{nb_clusters}"] = 0
                    X_test[f"dbscan_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif algorithm == "gaussian":
            if self.prediction_mode:
                try:
                    self.dataframe = add_gaussian_mixture_clusters(self.dataframe)
                except ValueError:
                    self.dataframe[f"gaussian_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                try:
                    X_train = add_gaussian_mixture_clusters(X_train)
                    X_test = add_gaussian_mixture_clusters(X_test)
                except ValueError:
                    X_train[f"gaussian_clusters_{nb_clusters}"] = 0
                    X_test[f"gaussian_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif algorithm == "kmeans":
            if self.prediction_mode:
                try:
                    self.dataframe = add_kmeans_clusters(self.dataframe)
                except ValueError:
                    self.dataframe[f"kmeans_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                try:
                    X_train = add_kmeans_clusters(X_train)
                    X_test = add_kmeans_clusters(X_test)
                except ValueError:
                    X_train[f"kmeans_clusters_{nb_clusters}"] = 0
                    X_test[f"kmeans_clusters_{nb_clusters}"] = 0
                logging.info("Finished adding clusters as additional features.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def pca_clustering(self, df, mode="fit"):
        """
        Takes a Dataframe and searchs for features containing "clustering". Reduces these to two PCA dimensions, but keeps
        the original colum,ns as well.
        :param df: Pandas Dataframe
        :param mode: "fit" and "transform"
        :return: Returns extended Dataframe
        """
        if mode == "transform":
            all_cols = df.columns
            cluster_columns = [x for x in all_cols if "cluster" not in x]
            cluster_df = df[cluster_columns].copy()
            pca = self.preprocess_decisions["cluster_pca"]
            comps = pca.transform(cluster_df.values)
            self.preprocess_decisions["cluster_pca"] = pca
            cluster_pca_cols = ["Cluster PC-1", "Cluster PC-2"]
            pos_df = pd.DataFrame(comps, columns=cluster_pca_cols)
            tfidf_df_pca = pos_df[cluster_pca_cols]
            df = pd.merge(
                df, tfidf_df_pca, left_index=True, right_index=True, how="left"
            )
        elif mode == "fit":
            all_cols = df.columns
            cluster_columns = [x for x in all_cols if "cluster" not in x]
            cluster_df = df[cluster_columns].copy()
            pca = PCA(n_components=2)
            comps = pca.fit_transform(cluster_df.values)
            self.preprocess_decisions["cluster_pca"] = pca
            cluster_pca_cols = ["Cluster PC-1", "Cluster PC-2"]
            pos_df = pd.DataFrame(comps, columns=cluster_pca_cols)
            tfidf_df_pca = pos_df[cluster_pca_cols]
            df = pd.merge(
                df, tfidf_df_pca, left_index=True, right_index=True, how="left"
            )
        return df

    def pca_clustering_results(self):
        """
        Adds PCA of clusterings as part of the blueprint pipeline.
        :return: Modifies class attributes.
        """
        self.get_current_timestamp("PCA the clustering results.")
        logging.info("Started to PCA the clustering results.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            self.dataframe = self.pca_clustering(self.dataframe, mode="transform")
            logging.info("Finished to PCA the clustering results.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = self.pca_clustering(X_train, mode="fit")
            X_test = self.pca_clustering(X_test, mode="transform")
            logging.info("Finished to PCA the clustering results.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def delete_high_null_cols(self, threshold=0.05):
        """
        Takes in a dataframe and removes columns, which have more NULLs than the given threshold.
        :param threshold: Maximum percentage of NULLs in a column allowed.
        :return: Updates test and train class attributes.
        """
        self.get_current_timestamp("Delete columns with high share of NULLs")
        if self.prediction_mode:
            for high_null_col in self.preprocess_decisions["deleted_high_null_cols"]:
                del self.dataframe[high_null_col]
            logging.info("Finished deleting columns with many NULLs.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        else:
            logging.info("Started deleting columns with many NULLs.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            columns_before = X_train.columns.to_list()
            X_train.dropna(
                axis=1, thresh=int(threshold * len(X_train.index)), inplace=True
            )
            columns_after = X_train.columns.to_list()
            X_test = X_test[columns_after].copy()
            deleted_columns = set(columns_before).difference(columns_after)
            deleted = []
            for key in deleted_columns:
                deleted.append(key)
            self.preprocess_decisions["deleted_high_null_cols"] = deleted
            logging.info(f"Finished deleting columns with many NULLs: {deleted}.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def iterative_imputation(self, dataframe, imputer=None):
        dataframe_cols = dataframe.columns  # [dataframe.isna().any()].tolist()
        imp_mean = IterativeImputer(
            random_state=0,
            estimator=BayesianRidge(),
            imputation_order="ascending",
            max_iter=1000,
        )
        if not imputer:
            imp_mean.fit(dataframe)
        else:
            imp_mean = imputer
            imp_mean.fit(dataframe)
        dataframe = imp_mean.transform(dataframe)
        dataframe_final = pd.DataFrame(dataframe, columns=dataframe_cols)
        self.preprocess_decisions["fill_nulls_imputer"] = imp_mean
        del imp_mean
        _ = gc.collect()
        return dataframe_final

    def static_filling(
        self, dataframe, fill_with=0, fill_cat_col_with="No words have been found"
    ):
        """
        Loop through dataframe and fill categorical and numeric columns seperately with predefined values.
        :param dataframe: Pandas Dataframe
        :param fill_with: Numeric value to fill with
        :param fill_cat_col_with: String to fill categorical NULLs.
        :return:
        """
        cat_columns = dataframe.select_dtypes(include=["object"]).columns.to_list()
        for col in cat_columns:
            dataframe[col] = dataframe[col].fillna(fill_cat_col_with, inplace=False)

        for vartype in self.num_dtypes:
            try:
                filtered_columns = dataframe.select_dtypes(
                    include=[vartype]
                ).columns.to_list()
                for col in filtered_columns:
                    dataframe[col] = dataframe[col].fillna(fill_with, inplace=False)
            except ValueError:
                pass
        return dataframe

    def fill_nulls(
        self,
        how="iterative_imputation",
        fill_with=0,
        fill_cat_col_with="No words have been found",
    ):
        """
        Takes in a dataframe and fills all NULLs with chosen value.
        :param fill_with: Define value to replace NULLs with.
        :param how: Chose 'static' to define static fill values, 'iterative_imputation' for the sklearns iterative
        imputer.
        :return: Returns modified dataframe
        """
        self.get_current_timestamp("Fill nulls")

        logging.info("Started filling NULLs.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        algorithms = ["iterative_imputation", "static"]
        if how not in algorithms:
            self.runtime_warnings(warn_about="wrong null algorithm")
        else:
            pass

        if self.prediction_mode:
            if not how:
                how = self.preprocess_decisions["fill_nulls_how"]
            else:
                pass

            if how == "static":
                self.dataframe = self.static_filling(
                    self.dataframe,
                    fill_with=fill_with,
                    fill_cat_col_with=fill_cat_col_with,
                )
            elif how == "iterative_imputation":
                self.dataframe = self.iterative_imputation(
                    self.dataframe,
                    imputer=self.preprocess_decisions["fill_nulls_imputer"],
                )
            logging.info("Finished filling NULLs.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if how == "static":
                X_train = self.static_filling(
                    X_train, fill_with=fill_with, fill_cat_col_with=fill_cat_col_with
                )
                X_test = self.static_filling(
                    X_test, fill_with=fill_with, fill_cat_col_with=fill_cat_col_with
                )
                self.preprocess_decisions["fill_nulls_how"] = how
            elif how == "iterative_imputation":
                # TODO: Test, if it woks + revert LGBM + test model ensemble
                X_train = self.iterative_imputation(X_train)
                X_test = self.iterative_imputation(
                    X_test, imputer=self.preprocess_decisions["fill_nulls_imputer"]
                )
                self.preprocess_decisions["fill_nulls_how"] = how
            logging.info("Finished filling NULLs.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def holistic_null_filling(self, iterative=False):
        self.get_current_timestamp("Holistic NULL filling")
        logging.info("Started holistic NULL filling.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            for col in self.preprocess_decisions["holistically_filled_cols"]:
                if (
                    self.dataframe[col].dtype in self.num_dtypes
                ):  # checking if col is numeric
                    algorithm = "mean_filling"
                    imp = self.preprocess_decisions[
                        f"fill_nulls_{algorithm}_imputer_{col}"
                    ]
                    self.dataframe[col + "_" + algorithm] = imp.transform(
                        self.dataframe[col].values.reshape(-1, 1)
                    ).reshape(-1, 1)

                    algorithm = "static_filling"
                    self.dataframe[col + "_" + algorithm] = self.dataframe[col].fillna(
                        0, inplace=False
                    )

                    algorithm = "most_frequent"
                    imp = self.preprocess_decisions[
                        f"fill_nulls_{algorithm}_imputer_{col}"
                    ]
                    self.dataframe[col + "_" + algorithm] = imp.transform(
                        self.dataframe[col].values.reshape(-1, 1)
                    ).reshape(-1, 1)

                else:
                    algorithm = "most_frequent"
                    imp = self.preprocess_decisions[
                        f"fill_nulls_{algorithm}_imputer_{col}"
                    ]
                    self.dataframe[col + "_" + algorithm] = imp.transform(
                        self.dataframe[col].values.reshape(-1, 1)
                    ).reshape(-1, 1)

                    algorithm = "static_filling"
                    self.dataframe[col + "_" + algorithm] = self.dataframe[col].fillna(
                        "None", inplace=False
                    )

                    # fill original column as prep for iterative filling
                    self.dataframe[col] = self.dataframe[col].fillna(
                        "None", inplace=False
                    )

            if iterative:
                algorithm = "iterative_filling"
                imp = self.preprocess_decisions[
                    f"fill_nulls_{algorithm}_imputer_all_cols"
                ]
                cat_columns = self.dataframe.select_dtypes(
                    include=["object"]
                ).columns.to_list()
                no_cat_cols = self.dataframe.loc[
                    :, ~self.dataframe.columns.isin(cat_columns)
                ].columns.to_list()
                imp.transform(self.dataframe[no_cat_cols])
            else:
                pass
            logging.info("Finished holistic NULL filling.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            filled_cols = []
            # numeric vs categorical
            for col in X_train.columns.to_list():
                if X_train[col].isna().sum() > 0:
                    print(f"Impute column {col}...")
                    if (
                        X_train[col].dtype in self.num_dtypes
                    ):  # checking if col is numeric
                        algorithm = "mean_filling"
                        imp = SimpleImputer(
                            missing_values=np.nan, strategy="mean", copy=True
                        )
                        imp.fit(X_train[col].values.reshape(-1, 1))
                        X_train[col + "_" + algorithm] = imp.transform(
                            X_train[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        X_test[col + "_" + algorithm] = imp.transform(
                            X_test[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        self.preprocess_decisions[
                            f"fill_nulls_{algorithm}_imputer_{col}"
                        ] = imp

                        algorithm = "static_filling"
                        X_train[col + "_" + algorithm] = X_train[col].fillna(
                            0, inplace=False
                        )
                        X_test[col + "_" + algorithm] = X_test[col].fillna(
                            0, inplace=False
                        )

                        # most frequent filling
                        algorithm = "most_frequent"
                        imp = SimpleImputer(
                            missing_values=np.nan, strategy="most_frequent", copy=True
                        )
                        imp.fit(X_train[col].values.reshape(-1, 1))
                        X_train[col + "_" + algorithm] = imp.transform(
                            X_train[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        X_test[col + "_" + algorithm] = imp.transform(
                            X_test[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        self.preprocess_decisions[
                            f"fill_nulls_{algorithm}_imputer_{col}"
                        ] = imp
                    else:
                        # most frequent filling
                        algorithm = "most_frequent"
                        imp = SimpleImputer(
                            missing_values=np.nan, strategy="most_frequent", copy=True
                        )
                        imp.fit(X_train[col].values.reshape(-1, 1))
                        X_train[col + "_" + algorithm] = imp.transform(
                            X_train[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        X_test[col + "_" + algorithm] = imp.transform(
                            X_test[col].values.reshape(-1, 1)
                        ).reshape(-1, 1)
                        self.preprocess_decisions[
                            f"fill_nulls_{algorithm}_imputer_{col}"
                        ] = imp

                        algorithm = "static_filling"
                        X_train[col + "_" + algorithm] = X_train[col].fillna(
                            "None", inplace=False
                        )
                        X_test[col + "_" + algorithm] = X_test[col].fillna(
                            "None", inplace=False
                        )

                        # fill original column as prep for iterative filling
                        X_train[col] = X_train[col].fillna("None", inplace=False)
                        X_test[col] = X_test[col].fillna("None", inplace=False)

                    filled_cols.append(col)
            if iterative:
                algorithm = "iterative_filling"
                model = lgb.LGBMRegressor()
                cat_columns = X_train.select_dtypes(
                    include=["object"]
                ).columns.to_list()
                no_cat_cols = X_train.loc[
                    :, ~X_train.columns.isin(cat_columns)
                ].columns.to_list()
                imp = IterativeImputer(
                    random_state=0,
                    estimator=model,
                    imputation_order="ascending",
                    max_iter=1000,
                )
                imp.fit(X_train[no_cat_cols])
                imp.transform(X_test[no_cat_cols])
                self.preprocess_decisions[
                    f"fill_nulls_{algorithm}_imputer_all_cols"
                ] = imp
            else:
                pass

            self.preprocess_decisions["holistically_filled_cols"] = filled_cols
            logging.info("Finished holistic NULL filling.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def fill_infinite_values(self, fill_with_zero=True):
        if fill_with_zero:
            filler = 0
        else:
            filler = np.nan
        if self.prediction_mode:
            self.dataframe = self.dataframe.replace([np.inf, -np.inf], filler)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = X_train.replace([np.inf, -np.inf], filler)
            X_test = X_test.replace([np.inf, -np.inf], filler)
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def svm_outlier_detection(self, nu=0.1):
        """
        Uses SVM oneclass to append outlier scores to the DataFrame.
        :param nu: Float between >0 and <1. Indicates how much of the values in percent shall be flagged as outliers.
        :return: Updates class attributes.
        """
        self.get_current_timestamp("Started SVM outlier detection.")
        logging.info("Started outlier handling.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            onesvm = OneClassSVM(kernel="rbf", nu=nu)
            onesvm.fit(self.dataframe)
            outlier_flags = onesvm.predict(self.dataframe)
            self.dataframe[f"svm_outlier_score_nu_{nu}"] = outlier_flags
            logging.info("Started outlier handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.get_current_timestamp("Finished SVM outlier detection.")
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            onesvm = OneClassSVM(kernel="rbf", nu=nu)
            onesvm.fit(X_train)
            outlier_flags = onesvm.predict(X_train)
            X_train[f"svm_outlier_score_nu_{nu}"] = outlier_flags

            onesvm = OneClassSVM(kernel="rbf", nu=nu)
            onesvm.fit(X_test)
            outlier_flags = onesvm.predict(X_test)
            X_test[f"svm_outlier_score_nu_{nu}"] = outlier_flags

            logging.info("Started outlier handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.get_current_timestamp("Finished SVM outlier detection.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def isolation_forest_identifier(self, how="append", threshold=0):
        """
        Takes a dataframe and runs isolation forest to either flag or delete outliers.
        :param how: Chose if outlier scores shall be 'append' or 'delete'.
        :param threshold: Threshold responsible for outlier deletion. Samples under this threshold will be deleted.
        :return: Returns modified dataframe.
        """
        if self.prediction_mode:
            if (
                self.preprocess_decisions["isolation_forest"]["how"] == "append"
                and how == "append"
            ):
                outlier_detector = self.preprocess_decisions["isolation_forest"][
                    "model"
                ]
                outlier_predictions = outlier_detector.decision_function(self.dataframe)
                outlier_predictions_class = outlier_predictions * -1
                self.dataframe["isolation_probs"] = outlier_predictions
                self.dataframe["isolation_class"] = outlier_predictions_class
                return self.dataframe
            else:
                pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            outlier_detector = IsolationForest(contamination="auto")
            if how == "append":
                outlier_detector.fit(X_train)
                outlier_predictions_train = outlier_detector.decision_function(X_train)
                outlier_predictions_class_train = outlier_predictions_train * -1
                X_train["isolation_probs"] = outlier_predictions_train
                X_train["isolation_class"] = outlier_predictions_class_train
                outlier_predictions_test = outlier_detector.decision_function(X_test)
                outlier_predictions_class_test = outlier_predictions_test * -1
                X_test["isolation_probs"] = outlier_predictions_test
                X_test["isolation_class"] = outlier_predictions_class_test
                del outlier_predictions_train
                del outlier_predictions_test
                del outlier_predictions_class_train
                del outlier_predictions_class_test
                self.preprocess_decisions["isolation_forest"] = {}
                self.preprocess_decisions["isolation_forest"][
                    "model"
                ] = outlier_detector
                self.preprocess_decisions["isolation_forest"]["how"] = how
            elif how == "delete":
                original_len = len(X_train.index)
                outlier_detector.fit(X_train)
                outlier_predictions_train = outlier_detector.decision_function(X_train)
                X_train["isolation_probs_for_deletion"] = outlier_predictions_train

                X_train[self.target_variable] = Y_train
                X_train = X_train[
                    (X_train["isolation_probs_for_deletion"] > threshold)
                ].copy()
                X_train = X_train.reset_index(drop=True)
                Y_train = X_train[self.target_variable].copy()
                X_train = X_train.drop(self.target_variable, axis=1)
                X_train = X_train.drop("isolation_probs_for_deletion", axis=1)
                new_len = len(X_train.index)
                print(f"Training data size reduced from {original_len} to {new_len}.")
                del outlier_predictions_train
            del outlier_detector
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def iqr_remover(self, threshold=1.5):
        """
        Remove outliers from a dataframe by column, including optional whiskers, removing rows for which the column value
         are less than Q1-1.5IQR or greater than Q3+1.5IQR.
        :param threshold: whisker_width (float): Optional, loosen the IQR filter by a factor of `whisker_width` * IQR.
        Default is 1.5.
        :return: Updates class attributed.
        """
        if self.prediction_mode:
            return self.dataframe
        else:
            whisker_width = threshold
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            dataframe_red = X_train.loc[
                :, X_train.columns.isin(self.num_columns)
            ].copy()
            dataframe_red[self.target_variable] = Y_train
            for col in dataframe_red.columns:
                # Calculate Q1, Q2 and IQR
                q1 = dataframe_red[col].quantile(0.25)
                q3 = dataframe_red[col].quantile(0.75)
                iqr = q3 - q1
                # Apply filter with respect to IQR, including optional whiskers
                filter = (dataframe_red[col] > q1 - whisker_width * iqr) & (
                    dataframe_red[col] < q3 + whisker_width * iqr
                )
                dataframe_red = dataframe_red.loc[filter]
            X_train = dataframe_red
            Y_train = dataframe_red[self.target_variable]
            del dataframe_red[self.target_variable]
            del dataframe_red
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def outlier_care(self, method="isolation", how="append", threshold=None):
        """
        This method handles outliers isolation forest only currently.
        :param method: Chose the method of outlier detection. Either 'IQR', 'z_avg or 'iqr_avg'.
        :param how: Chose 'adjust' to correct outliers by adjusting them to IQR (for IQR only), 'delete' to delete all
        rows with outliers or 'append' to append outlier scores.
        :param threshold: Define by how many range an outlier has to be off to be interpreted as outlier.
        :return: Returns instantiated dataframe object.
        """
        self.get_current_timestamp("Handle outliers")
        logging.info("Started outlier handling.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if method == "isolation" and how == "append":
            logging.info("Finished outlier handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == "isolation" and how == "delete":
            logging.info("Finished outlier handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == "iqr":
            logging.info("Finished outlier handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.iqr_remover(threshold=1.5)

    def datetime_converter(  # noqa: C901
        self, datetime_handling="all", force_conversion=False
    ):
        """
        Takes in a dataframe and processes date and datetime columns by categorical and/or cyclic transformation.
        Tries to identify datetime columns automatically, if no date columns have been provided during class
        instantiation.
        :param datetime_handling: Chose '
        :return:
        """
        if self.prediction_mode:
            if not self.date_columns:
                logging.info("Started automatic datetime column detection.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                date_columns = []
                # convert date columns from object to datetime type
                for col in self.dataframe.columns:
                    if col not in self.num_columns:
                        try:
                            self.dataframe[col] = pd.to_datetime(
                                self.dataframe[col], yearfirst=True
                            )
                            date_columns.append(col)
                        except Exception:
                            if force_conversion:
                                self.dataframe[col] = pd.to_datetime(
                                    self.dataframe[col], yearfirst=True, errors="coerce"
                                )
                                date_columns.append(col)
                logging.info("Finished automatic datetime column detection.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    try:
                        self.dataframe[col] = pd.to_datetime(
                            self.dataframe[col], yearfirst=True
                        )
                    except KeyError:
                        pass

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if not self.date_columns:
                logging.info("Started automatic datetime column detection.")
                logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
                date_columns = []
                # convert date columns from object to datetime type
                for col in X_train.columns:
                    if col not in self.num_columns:
                        try:
                            X_train[col] = pd.to_datetime(X_train[col], yearfirst=True)
                            X_test[col] = pd.to_datetime(X_test[col], yearfirst=True)
                            date_columns.append(col)
                        except Exception:
                            if force_conversion:
                                X_train[col] = pd.to_datetime(
                                    X_train[col], yearfirst=True, errors="coerce"
                                )
                                X_test[col] = pd.to_datetime(
                                    X_test[col], yearfirst=True, errors="coerce"
                                )
                                date_columns.append(col)
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    try:
                        X_train[col] = pd.to_datetime(
                            X_train[col], yearfirst=True, errors="coerce"
                        )
                        X_test[col] = pd.to_datetime(
                            X_test[col], yearfirst=True, errors="coerce"
                        )
                    except KeyError:
                        # might happen if deleted due to high nulls
                        pass
            logging.info("Finished automatic datetime column detection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        self.date_columns_created = {}
        self.new_sin_cos_col_names = []  # used to filter out these columns from binning

        def date_converter(dataframe):
            """
            Takes in a dataframe and loops through datetime columns to and extracts the date parts month, day, dayofweek
            and hour and adds them as additional columns.
            :param dataframe:
            :return: Returns modified dataframe.
            """
            for c in date_columns:
                if c in dataframe.columns:
                    if dataframe[c].dt.month.nunique() > 0:
                        dataframe[c + "_month"] = dataframe[c].dt.month
                        self.date_columns_created[c + "_month"] = "month"
                    if dataframe[c].dt.day.nunique() > 0:
                        dataframe[c + "_day"] = dataframe[c].dt.day
                        self.date_columns_created[c + "_day"] = "day"
                    if dataframe[c].dt.dayofweek.nunique() > 0:
                        dataframe[c + "_dayofweek"] = dataframe[c].dt.dayofweek
                        self.date_columns_created[c + "_dayofweek"] = "dayofweek"
                    if dataframe[c].dt.hour.nunique() > 0:
                        dataframe[c + "_hour"] = dataframe[c].dt.hour
                        self.date_columns_created[c + "_hour"] = "hour"
            return dataframe

        def cos_sin_transformation(dataframe):
            """
            Takes in a dataframe and loops through date columns. Create sine and cosine features and appends them
            as new columns.
            :param dataframe:
            :return: Returns modified dataframe.
            """
            for c in self.date_columns_created:
                if c in dataframe.columns:
                    if self.date_columns_created[c] == "month":
                        dataframe[c + "_sin"] = np.sin(
                            dataframe[c] * (2.0 * np.pi / 12)
                        )
                        dataframe[c + "_cos"] = np.cos(
                            dataframe[c] * (2.0 * np.pi / 12)
                        )
                        self.new_sin_cos_col_names.append(c + "_sin")
                        self.new_sin_cos_col_names.append(c + "_cos")
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == "day":
                        dataframe[c + "_sin"] = np.sin(
                            dataframe[c] * (2.0 * np.pi / 31)
                        )
                        dataframe[c + "_cos"] = np.cos(
                            dataframe[c] * (2.0 * np.pi / 31)
                        )
                        self.new_sin_cos_col_names.append(c + "_sin")
                        self.new_sin_cos_col_names.append(c + "_cos")
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == "dayofweek":
                        dataframe[c + "_sin"] = np.sin(
                            (dataframe[c] + 1) * (2.0 * np.pi / 7)
                        )
                        dataframe[c + "_cos"] = np.cos(
                            (dataframe[c] + 1) * (2.0 * np.pi / 7)
                        )
                        self.new_sin_cos_col_names.append(c + "_sin")
                        self.new_sin_cos_col_names.append(c + "_cos")
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == "hour":
                        dataframe[c + "_sin"] = np.sin(
                            dataframe[c] * (2.0 * np.pi / 24)
                        )
                        dataframe[c + "_cos"] = np.cos(
                            dataframe[c] * (2.0 * np.pi / 24)
                        )
                        self.new_sin_cos_col_names.append(c + "_sin")
                        self.new_sin_cos_col_names.append(c + "_cos")
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == "dayofyear":
                        dataframe[c + "_sin"] = np.sin(
                            dataframe[c] * (2.0 * np.pi / 365)
                        )
                        dataframe[c + "_cos"] = np.cos(
                            dataframe[c] * (2.0 * np.pi / 365)
                        )
                        self.new_sin_cos_col_names.append(c + "_sin")
                        self.new_sin_cos_col_names.append(c + "_cos")
                        dataframe.drop(c, axis=1, inplace=True)
            return dataframe

        self.get_current_timestamp(task="Apply datetime transformation")
        logging.info("Started datetime column handling.")
        if self.prediction_mode:
            datetime_handling = self.preprocess_decisions["datetime_handling"]
            if datetime_handling == "cyclic":
                self.dataframe = cos_sin_transformation(self.dataframe)
            elif datetime_handling == "categorical":
                self.dataframe = date_converter(self.dataframe)
            elif datetime_handling == "all":
                self.dataframe = date_converter(self.dataframe)
                self.dataframe = cos_sin_transformation(self.dataframe)
            else:
                pass

        elif datetime_handling == "cyclic":
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = cos_sin_transformation(X_train)
            X_test = cos_sin_transformation(X_test)
        elif datetime_handling == "categorical":
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = date_converter(X_train)
            X_test = date_converter(X_test)
        elif datetime_handling == "all":
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = date_converter(X_train)
            X_train = cos_sin_transformation(X_train)
            X_test = date_converter(X_test)
            X_test = cos_sin_transformation(X_test)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

        if self.prediction_mode:
            # drop initial date columns
            for dates in date_columns:
                if dates in self.dataframe.columns:
                    # safe_copy = all_data[dates].copy()
                    self.dataframe.drop(dates, axis=1, inplace=True)
            logging.info("Finished datetime column handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            # drop initial date columns
            for dates in date_columns:
                if dates in X_train.columns:
                    # safe_copy = all_data[dates].copy()
                    X_train.drop(dates, axis=1, inplace=True)
                    X_test.drop(dates, axis=1, inplace=True)
            self.preprocess_decisions["datetime_handling"] = datetime_handling
            logging.info("Finished datetime column handling.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return (
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test),
                self.date_columns_created,
            )

    def onehot_pca(self):
        """
        Takes categorical columns, executes onehot encoding on them and reduces dimensionality with PCA.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Onehot + PCA categorical features")
        logging.info("Started Onehot + PCA categorical features.")
        if self.prediction_mode:
            if len(self.cat_columns_encoded) > 0:
                df_branch = self.dataframe[self.cat_columns_encoded].copy()
                enc = self.preprocess_decisions["onehot_pca"]["onehot_encoder"]
                df_branch = enc.transform(df_branch[self.cat_columns_encoded])
                df_branch.fillna(0, inplace=True)
                onehot_cols = df_branch.columns
                # pca = self.preprocess_decisions["onehot_pca"]["pca_encoder"]
                pca = PCA(n_components=2)
                pred_comps = pca.fit_transform(df_branch[onehot_cols])
                df_branch = pd.DataFrame(pred_comps, columns=["PC-1", "PC-2"])
                for col in df_branch.columns:
                    self.dataframe[f"{col}_pca"] = df_branch[col]
                del df_branch
                del pca
                del pred_comps
                del enc
                _ = gc.collect()
            else:
                pass
            logging.info("Finished Onehot + PCA categorical features.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions["onehot_pca"] = {}
            if self.cat_columns_encoded:
                cat_columns = self.cat_columns_encoded
            else:
                cat_columns = X_train.select_dtypes(
                    include=["object"]
                ).columns.to_list()
                self.cat_columns_encoded = cat_columns

            if len(self.cat_columns_encoded) > 0:
                enc = OneHotEncoder(handle_unknown="ignore")
                X_train_branch = X_train[cat_columns].copy()
                X_test_branch = X_test[cat_columns].copy()
                X_train_branch = enc.fit_transform(X_train_branch[cat_columns], Y_train)
                X_test_branch = enc.transform(X_test_branch[cat_columns])
                onehot_cols = X_train_branch.columns
                X_train_branch.fillna(0, inplace=True)
                X_test_branch.fillna(0, inplace=True)
                pca = PCA(n_components=2)
                train_comps = pca.fit_transform(X_train_branch[onehot_cols])
                X_train_branch = pd.DataFrame(train_comps, columns=["PC-1", "PC-2"])
                test_comps = pca.transform(X_test_branch[onehot_cols])
                X_test_branch = pd.DataFrame(test_comps, columns=["PC-1", "PC-2"])
                pca_cols = []
                for col in X_train_branch.columns:
                    X_train[f"{col}_pca"] = X_train_branch[col]
                    X_test[f"{col}_pca"] = X_test_branch[col]
                    pca_cols.append(f"{col}_pca")
                self.preprocess_decisions["onehot_pca"]["pca_cols"] = pca_cols
                self.preprocess_decisions["onehot_pca"]["onehot_encoder"] = enc
                self.preprocess_decisions["onehot_pca"]["pca_encoder"] = pca
                del X_train_branch
                del X_test_branch
                del pca
                del train_comps
                del test_comps
                _ = gc.collect()
            else:
                pass
            logging.info("Finished Onehot + PCA categorical features.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def numeric_binarizer_pca(self):
        self.get_current_timestamp(
            task="Binarize numeric columns + PCA binarized features"
        )
        logging.info("Started to binarize numeric columns + PCA binarized features.")
        if self.prediction_mode:
            if len(self.num_columns_encoded) > 0:
                num_cols_binarized_created = []
                for num_col in self.num_columns_encoded:
                    self.dataframe[num_col + "_binarized"] = self.dataframe[
                        num_col
                    ].apply(lambda x: 1 if x > 0 else 0)
                    num_cols_binarized_created.append(num_col + "_binarized")
                pca = PCA(n_components=2)
                df_branch = self.dataframe.copy()
                pred_comps = pca.fit_transform(df_branch[num_cols_binarized_created])
                df_branch = pd.DataFrame(pred_comps, columns=["Num_PC-1", "Num_PC-2"])
                for col in df_branch.columns:
                    self.dataframe[f"{col}_num_pca"] = df_branch[col]
                del df_branch
                del pred_comps
                del pca
                _ = gc.collect()
            else:
                pass
            logging.info(
                "Finished to binarize numeric columns + PCA binarized features."
            )
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions["numeric_binarizer_pca"] = {}

            encoded_num_cols = []
            for vartype in self.num_dtypes:
                try:
                    filtered_columns = X_train.select_dtypes(
                        include=[vartype]
                    ).columns.to_list()
                    for _pcas in filtered_columns:
                        try:
                            filtered_columns.remove("Num_PC-1_num_pca")
                            filtered_columns.remove("Num_PC-2_num_pca")
                        except Exception:
                            pass
                    for i in filtered_columns:
                        try:
                            encoded_num_cols.remove(i)
                        except Exception:
                            pass

                    filtered_columns = [
                        x for x in filtered_columns if "tfids_" not in x
                    ]
                    # filtered_columns = [ x for x in filtered_columns if "POS PC-" not in x]
                    # filtered_columns = [ x for x in filtered_columns if "textblob_sentiment_score" not in x]
                    # filtered_columns = [ x for x in filtered_columns if "TFIDF PC" not in x]
                    # filtered_columns = [ x for x in filtered_columns if "tfid_bayes_" not in x]

                    if len(filtered_columns) > 0:
                        num_cols_binarized_created = []
                        for num_col in filtered_columns:
                            X_train[num_col + "_binarized"] = X_train[num_col].apply(
                                lambda x: 1 if x > 0 else 0
                            )
                            X_test[num_col + "_binarized"] = X_test[num_col].apply(
                                lambda x: 1 if x > 0 else 0
                            )
                            num_cols_binarized_created.append(num_col + "_binarized")
                            encoded_num_cols.append(num_col)
                        pca = PCA(n_components=2)
                        X_train_branch = X_train.copy()
                        X_test_branch = X_test.copy()
                        train_comps = pca.fit_transform(
                            X_train_branch[num_cols_binarized_created]
                        )
                        test_comps = pca.fit_transform(
                            X_test_branch[num_cols_binarized_created]
                        )
                        X_train_branch = pd.DataFrame(
                            train_comps, columns=["Num_PC-1", "Num_PC-2"]
                        )
                        X_test_branch = pd.DataFrame(
                            test_comps, columns=["Num_PC-1", "Num_PC-2"]
                        )
                        pca_cols = []
                        for col in X_train_branch.columns:
                            X_train[f"{col}_num_pca"] = X_train_branch[col]
                            X_test[f"{col}_num_pca"] = X_test_branch[col]
                            pca_cols.append(f"{col}_num_pca")
                        self.preprocess_decisions["numeric_binarizer_pca"][
                            f"pca_cols_{vartype}"
                        ] = pca_cols
                        del X_train_branch
                        del X_test_branch
                        del train_comps
                        del test_comps
                        del pca
                        _ = gc.collect()
                    else:
                        pass
                except ValueError:
                    pass
            self.num_columns_encoded = encoded_num_cols
            logging.info(
                "Finished to binarize numeric columns + PCA binarized features."
            )
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def target_encode_multiclass(self, X, y=None, mode="fit"):
        algorithm = "multiclass_target_encoding_onehotter"
        if mode == "transform":
            enc = self.preprocess_decisions["category_encoders"][
                f"{algorithm}_all_cols"
            ]
            class_names = self.preprocess_decisions["category_encoders"]["seen_targets"]
        else:
            enc = OneHotEncoder()
            enc.fit(y)
            y_onehot = enc.transform(y)
            class_names = y_onehot.columns
            self.preprocess_decisions["category_encoders"]["seen_targets"] = class_names
        X_obj = X.select_dtypes("object").copy()
        X = X.select_dtypes(exclude="object")
        for class_ in class_names:
            if mode == "transform":
                target_enc = self.preprocess_decisions["category_encoders"][
                    f"multiclass_target_encoder_all_cols_{class_}"
                ]
            else:
                target_enc = TargetEncoder()
                target_enc.fit(X_obj, y_onehot[class_])
                self.preprocess_decisions["category_encoders"][
                    f"multiclass_target_encoder_all_cols_{class_}"
                ] = target_enc
            temp = target_enc.transform(X_obj)
            temp.columns = [str(x) + "_" + str(class_) for x in temp.columns]
            X = pd.concat([X, temp], axis=1)
        self.preprocess_decisions["category_encoders"][f"{algorithm}_all_cols"] = enc
        return X

    def category_encoding(self, algorithm="target"):
        """
        Takes in a dataframe and applies the chosen category encoding algorithm to categorical columns.
        :param algorithm: Chose type of encoding as 'target' (default), 'onehot', 'woee', 'ordinal', 'leaveoneout' and 'GLMM'.
        :return: Returns modified dataframe.
        """
        self.get_current_timestamp("Execute categorical encoding")
        logging.info("Started category encoding.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            cat_columns = self.cat_columns_encoded
            if algorithm == "target" and self.class_problem == "multiclass":
                self.dataframe[cat_columns] = self.target_encode_multiclass(
                    self.dataframe[cat_columns], mode="transform"
                )
            else:
                enc = self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ]
                self.dataframe[cat_columns] = enc.transform(self.dataframe[cat_columns])
            logging.info("Finished category encoding.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            cat_columns = X_train.select_dtypes(include=["object"]).columns.to_list()
            self.cat_columns_encoded = cat_columns
            self.preprocess_decisions["category_encoders"] = {}
            if algorithm == "target":
                if self.class_problem in ["binary", "regression"]:
                    enc = TargetEncoder(cols=cat_columns)
                    X_train[cat_columns] = enc.fit_transform(
                        X_train[cat_columns], Y_train
                    )
                    X_test[cat_columns] = enc.transform(X_test[cat_columns])
                    self.preprocess_decisions["category_encoders"][
                        f"{algorithm}_all_cols"
                    ] = enc
                else:
                    X_train[cat_columns] = self.target_encode_multiclass(
                        X_train[cat_columns], Y_train, mode="fit"
                    )
                    X_test[cat_columns] = self.target_encode_multiclass(
                        X_test[cat_columns], mode="transform"
                    )
            elif algorithm == "onehot":
                enc = OneHotEncoder(handle_unknown="ignore")
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ] = enc
            elif algorithm == "woee":
                enc = WOEEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ] = enc
            elif algorithm == "GLMM":
                enc = GLMMEncoder(cols=cat_columns)
                # enc = NestedCVWrapper(enc_enc, random_state=42)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ] = enc
            elif algorithm == "ordinal":
                enc = OrdinalEncoder(cols=cat_columns)
                X_train = enc.fit_transform(X_train, Y_train)
                X_test = enc.transform(X_test)
                self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ] = enc
            elif algorithm == "leaveoneout":
                enc = LeaveOneOutEncoder(cols=cat_columns)
                # enc = NestedCVWrapper(enc_enc, random_state=42)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions["category_encoders"][
                    f"{algorithm}_all_cols"
                ] = enc
            logging.info("Finished category encoding.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train.drop(cat_columns, axis=1)
            X_test.drop(cat_columns, axis=1)
            try:
                del enc
                _ = gc.collect()
            except UnboundLocalError:
                pass
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def remove_collinearity(self, threshold=0.8):
        """
        Loops through all columns and checks, if features are highly positively correlated.
        If correlation is above given threshold, then only one column is kept.
        :param threshold: Maximum allowed correlation. Expects a float from -1 to +1.
        :return: Returns modified dataframe.
        """
        self.get_current_timestamp("Remove collinearity")

        def correlation(dataset, threshold=threshold):
            col_corr = set()  # Set of all the names of deleted columns
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= threshold) and (
                        corr_matrix.columns[j] not in col_corr
                    ):
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
                        del_corr.append(colname)
                        if colname in dataset.columns:
                            del dataset[colname]  # deleting the column from the dataset
            del corr_matrix
            _ = gc.collect()
            return dataset

        logging.info("Started removing collinearity.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            threshold = self.preprocess_decisions["remove_collinearity_threshold"]
            self.dataframe = self.dataframe.drop(self.excluded, axis=1)
            logging.info("Finished removing collinearity.")
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            del_corr = []
            X_train = correlation(X_train, 0.8)
            X_test = X_test.drop(del_corr, axis=1)
            self.excluded = del_corr
            self.preprocess_decisions["remove_collinearity_threshold"] = threshold
            logging.info("Finished removing collinearity.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return (
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test),
                self.excluded,
            )

    def smote_data(self):
        """
        Applies vanilla form of Synthetical Minority Over-Sampling Technique.
        :return: Returns modified dataframe.
        """
        self.get_current_timestamp("Smote data")
        if self.prediction_mode:
            logging.info("Skipped SMOTE due to prediction mode.")
            pass
        else:
            logging.info("Started SMOTE.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            oversample = SMOTE(n_jobs=-1)
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train_cols = X_train.columns
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)
            X_train = pd.DataFrame(X_train, columns=X_train_cols)
            logging.info("Finished SMOTE.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            del oversample
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def naive_undersampling(self, df, target_name):
        """
        Takes a dataframe and the name of the target variable to undersample all classes other than the minority class.
        This is a naive undersampling technique and should only be done on the training dataset.
        :param df: Expects a Pandas dataframe.
        :param target_name: Expects a string with the name of the target column.
        :return: Returns the modified Pandas dataframe.
        """
        classes = df[target_name].value_counts().to_dict()
        least_class_amount = min(classes.values())
        classes_list = []
        for key in classes:
            classes_list.append(df[df[target_name] == key])
        classes_sample = []
        for i in range(0, len(classes_list) - 1):
            classes_sample.append(
                classes_list[i].sample(least_class_amount, random_state=50)
            )
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([df_maybe, classes_list[-1]], axis=0)
        final_df = final_df.reset_index(drop=True)
        return final_df

    def naive_oversampling(self, df, target_name):
        classes = df[target_name].value_counts().to_dict()
        most = max(classes.values())
        classes_list = []
        for key in classes:
            classes_list.append(df[df[target_name] == key])
        classes_sample = []
        for i in range(1, len(classes_list)):
            classes_sample.append(
                classes_list[i].sample(most, replace=True, random_state=50)
            )
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
        final_df = final_df.reset_index(drop=True)
        return final_df

    def undersample_train_data(self):
        if self.prediction_mode:
            pass
        else:
            if self.class_problem == "regression":
                pass
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train[self.target_variable] = Y_train
                X_train = self.naive_undersampling(X_train, self.target_variable)
                Y_train = X_train[self.target_variable]
                X_train.drop(self.target_variable, axis=1)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def oversample_train_data(self):
        if self.prediction_mode:
            pass
        else:
            if self.class_problem == "regression":
                pass
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train[self.target_variable] = Y_train
                X_train = self.naive_oversampling(X_train, self.target_variable)
                # X_train = X_train.sample(frac=0.50)
                Y_train = X_train[self.target_variable]
                X_train.drop(self.target_variable, axis=1)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def create_trainers(self):
        if self.class_problem == "binary" or self.class_problem == "multiclass":
            if self.rapids_acceleration:
                from cuml.ensemble import RandomForestClassifier

                model_2 = RandomForestClassifier(
                    max_features=1.0,
                    max_depth=8,
                    output_type="numpy",
                    random_state=self.preprocess_decisions["random_state_counter"],
                )
            else:
                # model_1 = VWClassifier()
                model_2 = lgb.LGBMClassifier(
                    random_state=self.preprocess_decisions["random_state_counter"]
                )
        else:
            if self.rapids_acceleration:
                from cuml.ensemble import RandomForestRegressor

                model_2 = RandomForestRegressor(
                    max_features=1.0,
                    max_depth=8,
                    output_type="numpy",
                    random_state=self.preprocess_decisions["random_state_counter"],
                )
            else:
                # model_1 = VWRegressor()
                model_2 = lgb.LGBMRegressor(
                    random_state=self.preprocess_decisions["random_state_counter"]
                )
        return model_2

    def meissner_cv_score(self, matthew_scores, penality_is_deducted=True):
        """
        Takes in a list of scores from a crossvalidation and returns the Meißner CV score.
        The Meißner CV score will penalize, if the cross validation scores have higher variance. It scales from minus
        infinity to 100. The Meißner CV score is intended to be used with matthew correlation scores, but could also be
        used for other metrics.
        :param matthew_scores: List of cross validation scores
        :param penality_is_deducted: If true, penality of higher variance shall be deducted from the cross validation scores.
        :return: Returns the Meißner CV score.
        """
        mean_matthew_corr = np.array(matthew_scores) * 100
        if penality_is_deducted:
            meissner_cv = np.power(
                np.mean(mean_matthew_corr) ** 3
                - (
                    np.sum(
                        abs(
                            mean_matthew_corr
                            - mean_matthew_corr
                            - np.std(mean_matthew_corr)
                        )
                    )
                )
                ** 3,
                1 / 3,
            )
        else:
            meissner_cv = np.power(
                np.mean(mean_matthew_corr) ** 3
                + (
                    np.sum(
                        abs(
                            mean_matthew_corr
                            - mean_matthew_corr
                            - np.std(mean_matthew_corr)
                        )
                    )
                )
                ** 3,
                1 / 3,
            )
        return meissner_cv

    def synthetic_floating_data_generator(  # noqa: C901
        self, column_name=None, metric=None, sample_size=None
    ):
        self.get_current_timestamp("Synthetic data augmentation")

        if self.prediction_mode:
            pass
        else:
            logging.info("Start creating synthetic data.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            float_cols = [
                x
                for x, y in self.detected_col_types.items()
                if y in ["float", "int", "bool"]
                if x in X_train.columns.to_list()
            ]
            X_train = X_train[float_cols].copy()
            X_test = X_test[float_cols].copy()

            X_train_sample = X_train.copy()
            X_train_sample[self.target_variable] = Y_train
            X_train_sample = X_train_sample.sample(
                sample_size,
                random_state=self.preprocess_decisions["random_state_counter"],
            )
            Y_train_sample = X_train_sample[self.target_variable]
            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            if self.rapids_acceleration:
                X_train = X_train.astype("float32")
                X_train_sample = X_train_sample.astype("float32")
                X_test = X_test.astype("float32")

            if metric:
                metric = metric
            elif self.class_problem == "binary":
                metric = make_scorer(matthews_corrcoef)
                problem = "binary"
                class_cats = Y_train_sample.unique()
            elif self.class_problem == "multiclass":
                metric = make_scorer(matthews_corrcoef)
                problem = "multiclass"
                class_cats = Y_train_sample.unique()
            elif self.class_problem == "regression":
                metric = "neg_mean_squared_error"
                problem = "regression"

            model_2 = self.create_trainers()
            model_2_copy = model_2
            model_3_copy = model_2

            if (
                self.preprocess_decisions["synthetic_augmentation_parameters_benchmark"]
                == 0
            ):
                # get benchmark
                try:
                    # train scores
                    scores_2 = cross_val_score(
                        model_2, X_train, Y_train, cv=10, scoring=metric
                    )
                    mae_2 = np.mean(scores_2)
                    train_mae = mae_2 * 100
                    # test scores
                    model_2.fit(X_train, Y_train)
                    scores_2_test = model_2.predict(X_test)
                    try:
                        matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
                    except Exception:
                        matthew_2 = 0
                    test_mae = matthew_2 * 100
                    self.preprocess_decisions[
                        "synthetic_augmentation_parameters_benchmark"
                    ] = (train_mae + test_mae) / 2 - (abs(train_mae - test_mae)) ** 3
                except Exception:
                    self.preprocess_decisions[
                        "synthetic_augmentation_parameters_benchmark"
                    ] = 0
                else:
                    pass

            print(
                f"The benchmark score is {self.preprocess_decisions['synthetic_augmentation_parameters_benchmark']}."
            )

            # get core characteristics
            dist_max = int(X_train_sample[column_name].max() * 1.2)
            dist_min = int(X_train_sample[column_name].min() / 1.2)
            # dist_median = X_train_sample[column_name].median()
            dist_median_lowq = X_train_sample[column_name].quantile(0.25)
            dist_median_high = X_train_sample[column_name].quantile(0.75)

            if dist_max - dist_min <= 1:
                dist_max += 2
            if dist_median_high - dist_median_lowq <= 1:
                dist_median_high += 2

            if dist_median_lowq < 0 and dist_median_high <= 0:
                dist_median_high_inv = abs(dist_median_lowq)
                dist_median_lowq_inv = abs(dist_median_high)
            elif dist_median_lowq < 0 and dist_median_high > 0:
                if abs(dist_median_lowq) < abs(dist_median_high):
                    dist_median_high_inv = abs(dist_median_high)
                    dist_median_lowq_inv = abs(dist_median_lowq)
                else:
                    dist_median_high_inv = abs(dist_median_lowq)
                    dist_median_lowq_inv = abs(dist_median_high)
            else:
                dist_median_lowq_inv = dist_median_lowq
                dist_median_high_inv = dist_median_high

            if dist_min < 0 and dist_max <= 0:
                dist_max_inv = abs(dist_min)
                dist_min_inv = abs(dist_max)
            elif dist_min < 0 and dist_max > 0:
                if abs(dist_min) < abs(dist_max):
                    dist_max_inv = abs(dist_max)
                    dist_min_inv = abs(dist_min)
                else:
                    dist_max_inv = abs(dist_min)
                    dist_min_inv = abs(dist_max)
            else:
                dist_min_inv = dist_min
                dist_max_inv = dist_max

            if dist_max_inv - dist_min_inv <= 1:
                dist_max_inv += 2
            if dist_median_high_inv - dist_median_lowq_inv <= 1:
                dist_median_high_inv += 2

            if self.class_problem == "binary" or self.class_problem == "multiclass":
                pass
            else:
                X_train_sample[self.target_variable] = Y_train
                # sort on A
                X_train_sample.sort_values(self.target_variable, inplace=True)
                # create bins
                X_train_sample["bin"] = pd.cut(
                    X_train_sample[self.target_variable], 10, include_lowest=True
                )
                # group on bin
                group = X_train_sample.groupby("bin")
                # list comprehension to split groups into list of dataframes
                dfs = [group.get_group(x) for x in group.groups]

                Y_train_sample = X_train_sample[self.target_variable]
                X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            def objective(trial):
                param = {}

                sample_distribution = trial.suggest_categorical(
                    "sample_distribution",
                    [
                        "Uniform",
                        "Binomial",
                        "Poisson",
                        "Exponential",
                        "Gamma",
                        "Normal",
                        "Uniform",
                        "Pareto",
                        "Levy",
                        "dweibull",
                        "halfcauchy",
                        "halfnorm",
                        "powernorm",
                        "semicircular",
                        "tukeylambda",
                        "rdist",
                    ],
                )
                random_or_control_factor = trial.suggest_categorical(
                    "random_or_control_factor", ["Random", "Random pos", "Controlled"]
                )
                p_value = trial.suggest_loguniform("p_value", 0.05, 0.95)
                mu = trial.suggest_uniform(
                    "mu", dist_median_lowq_inv, dist_median_high_inv
                )
                scale = trial.suggest_int("scale", dist_min_inv, dist_max_inv)
                parteo_b = trial.suggest_uniform("parteo_b", dist_min_inv, dist_max_inv)
                uniformity = trial.suggest_uniform(
                    "uniformity", dist_min_inv, dist_max_inv
                )
                location = trial.suggest_int(
                    "location", dist_median_lowq_inv, dist_median_high_inv
                )
                lambda_value = trial.suggest_uniform("lambda_value", 1e-3, 10)
                c_value = trial.suggest_uniform("c_value", 1e-3, 10)
                pos_only_location = trial.suggest_uniform("pos_only_location", 0, 100)

                random_factor = trial.suggest_int("random_factor", dist_min, dist_max)
                if random_factor < 0:
                    random_factor_pos = random_factor + abs(dist_min)
                elif random_factor == 0:
                    random_factor_pos = random_factor + 1
                else:
                    random_factor_pos = random_factor

                param["sample_distribution"] = sample_distribution
                param["random_or_control_factor"] = random_or_control_factor
                param["p_value"] = p_value
                param["mu"] = mu
                param["scale"] = scale
                param["parteo_b"] = parteo_b
                param["random_factor"] = random_factor
                param["location"] = location
                param["lambda_value"] = lambda_value
                param["c_value"] = c_value
                param["pos_only_location"] = pos_only_location
                param["random_factor_pos"] = random_factor_pos

                temp_df_list = []
                X_train_sample[self.target_variable] = Y_train_sample
                if problem == "binary" or problem == "multiclass":
                    for class_inst in class_cats:
                        X_train_sample_class = X_train_sample[
                            (X_train_sample[self.target_variable] == class_inst)
                        ]
                        size = len(X_train_sample_class.index)
                        if sample_distribution == "Uniform":
                            gen_data = np.full((size,), uniformity)
                        elif sample_distribution == "Binomial":
                            gen_data = binom.rvs(
                                n=random_factor_pos, p=p_value, size=size
                            )
                        elif sample_distribution == "Poisson":
                            gen_data = poisson.rvs(mu=mu, size=size)
                        elif sample_distribution == "Exponential":
                            gen_data = expon.rvs(scale=scale, loc=location, size=size)
                        elif sample_distribution == "Gamma":
                            gen_data = gamma.rvs(a=mu, size=size)
                        elif sample_distribution == "Uniform":
                            gen_data = class_inst
                        elif sample_distribution == "Normal":
                            gen_data = norm.rvs(size=size, loc=location, scale=scale)
                        elif sample_distribution == "Pareto":
                            gen_data = pareto.rvs(parteo_b, size=size)
                        elif sample_distribution == "Levy":
                            gen_data = levy.rvs(size=size)
                            if random_or_control_factor == "Random":
                                gen_data = gen_data * random_factor
                            elif random_or_control_factor == "Random pos":
                                gen_data = gen_data * random_factor_pos
                            else:
                                gen_data += class_inst * 2
                        elif sample_distribution == "dweibull":
                            gen_data = dweibull.rvs(c=pos_only_location, size=size)
                        elif sample_distribution == "halfcauchy":
                            gen_data = halfcauchy.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "halfnorm":
                            gen_data = halfnorm.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "powernorm":
                            gen_data = powernorm.rvs(
                                c_value, loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "semicircular":
                            gen_data = semicircular.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "tukeylambda":
                            gen_data = tukeylambda.rvs(
                                lambda_value, loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "rdist":
                            gen_data = rdist.rvs(
                                c_value, loc=location, scale=scale, size=size
                            )
                        else:
                            gen_data = random_factor
                        try:
                            X_train_sample_class[column_name] = gen_data
                        except UnboundLocalError:
                            X_train_sample_class[column_name] = 0
                        temp_df_list.append(X_train_sample_class)

                else:
                    bin_encoder = 1
                    for X_train_sample_class in dfs:
                        size = len(X_train_sample_class.index)
                        class_inst = bin_encoder
                        bin_encoder += 1
                        if sample_distribution == "Uniform":
                            gen_data = np.full((size,), uniformity)
                        elif sample_distribution == "Binomial":
                            gen_data = binom.rvs(n=random_factor, p=p_value, size=size)
                        elif sample_distribution == "Poisson":
                            gen_data = poisson.rvs(mu=mu, size=size)
                        elif sample_distribution == "Exponential":
                            gen_data = expon.rvs(scale=scale, loc=location, size=size)
                        elif sample_distribution == "Gamma":
                            gen_data = gamma.rvs(a=mu, size=size)
                        elif sample_distribution == "Uniform":
                            gen_data = class_inst
                        elif sample_distribution == "Normal":
                            gen_data = norm.rvs(size=size, loc=location, scale=scale)
                        elif sample_distribution == "Pareto":
                            gen_data = pareto.rvs(parteo_b, size=size)
                        elif sample_distribution == "Levy":
                            gen_data = levy.rvs(size=size)
                            if random_or_control_factor == "Random":
                                gen_data = gen_data * random_factor
                            elif random_or_control_factor == "Random pos":
                                gen_data = gen_data * random_factor_pos
                            else:
                                gen_data += class_inst * 2
                        elif sample_distribution == "dweibull":
                            gen_data = dweibull.rvs(c=pos_only_location, size=size)
                        elif sample_distribution == "halfcauchy":
                            gen_data = halfcauchy.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "halfnorm":
                            gen_data = halfnorm.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "powernorm":
                            gen_data = powernorm.rvs(
                                c=c_value, loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "semicircular":
                            gen_data = semicircular.rvs(
                                loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "tukeylambda":
                            gen_data = tukeylambda.rvs(
                                lambda_value, loc=location, scale=scale, size=size
                            )
                        elif sample_distribution == "rdist":
                            gen_data = rdist.rvs(
                                c_value, loc=location, scale=scale, size=size
                            )
                        else:
                            gen_data = random_factor
                        try:
                            X_train_sample_class[column_name] = gen_data
                        except UnboundLocalError:
                            X_train_sample_class[column_name] = 0
                        X_train_sample_class = X_train_sample_class.drop("bin", axis=1)
                        temp_df_list.append(X_train_sample_class)

                temp_df = pd.concat(temp_df_list, ignore_index=False)
                Y_temp = temp_df[self.target_variable]
                temp_df = temp_df.drop(self.target_variable, axis=1)
                if self.rapids_acceleration:
                    temp_df = temp_df.astype("float32")

                # get train scores
                scores_2 = cross_val_score(
                    model_2_copy, temp_df, Y_temp, cv=10, scoring=metric
                )
                mae_2 = np.mean(scores_2)
                train_mae = mae_2 * 100
                # test scores
                model_2_copy.fit(temp_df, Y_temp)
                scores_2_test = model_2_copy.predict(X_test)
                try:
                    matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
                except Exception:
                    matthew_2 = 0

                test_mae = matthew_2 * 100
                mae = (train_mae + test_mae) / 2 - (abs(train_mae - test_mae)) ** 3

                return mae

            algorithm = "synthetic_data_augmentation"

            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.preprocess_decisions["random_state_counter"],
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
            )

            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds[
                    "synthetic_data_augmentation"
                ],
                timeout=self.hyperparameter_tuning_max_runtime_secs[
                    "synthetic_data_augmentation"
                ],
                show_progress_bar=True,
            )
            self.optuna_studies[f"{algorithm}"] = {}

            # get best logic
            best_parameters = study.best_trial.params
            temp_df_list = []
            X_train[self.target_variable] = Y_train

            if best_parameters["random_factor"] < 0:
                random_factor_pos = best_parameters["random_factor"] + abs(dist_min)
            elif best_parameters["random_factor"] == 0:
                random_factor_pos = best_parameters["random_factor"] + 1
            else:
                random_factor_pos = best_parameters["random_factor"]

            if self.class_problem == "binary" or self.class_problem == "multiclass":
                for class_inst in class_cats:
                    X_train_sample_class = X_train[
                        (X_train[self.target_variable] == class_inst)
                    ]
                    size = len(X_train_sample_class.index)
                    if best_parameters["sample_distribution"] == "Uniform":
                        gen_data = np.full((size,), best_parameters["uniformity"])
                    elif best_parameters["sample_distribution"] == "Binomial":
                        gen_data = binom.rvs(
                            n=random_factor_pos, p=best_parameters["p_value"], size=size
                        )
                    elif best_parameters["sample_distribution"] == "Poisson":
                        gen_data = poisson.rvs(mu=best_parameters["mu"], size=size)
                    elif best_parameters["sample_distribution"] == "Exponential":
                        gen_data = expon.rvs(
                            scale=best_parameters["scale"],
                            loc=best_parameters["location"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "Gamma":
                        gen_data = gamma.rvs(a=best_parameters["mu"], size=size)
                    elif best_parameters["sample_distribution"] == "Uniform":
                        gen_data = class_inst
                    elif best_parameters["sample_distribution"] == "Normal":
                        gen_data = norm.rvs(
                            size=size,
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                        )
                    elif best_parameters["sample_distribution"] == "Pareto":
                        gen_data = pareto.rvs(best_parameters["parteo_b"], size=size)
                    elif best_parameters["sample_distribution"] == "Levy":
                        gen_data = levy.rvs(size=size)
                        if best_parameters["random_or_control_factor"] == "Random":
                            gen_data = gen_data * best_parameters["random_factor"]
                        elif (
                            best_parameters["random_or_control_factor"] == "Random pos"
                        ):
                            gen_data = gen_data * random_factor_pos
                        else:
                            gen_data += class_inst * 2
                    elif best_parameters["sample_distribution"] == "dweibull":
                        gen_data = dweibull.rvs(
                            best_parameters["pos_only_location"], size=size
                        )
                    elif best_parameters["sample_distribution"] == "halfcauchy":
                        gen_data = halfcauchy.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "halfnorm":
                        gen_data = halfnorm.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "powernorm":
                        gen_data = powernorm.rvs(
                            best_parameters["c_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "semicircular":
                        gen_data = semicircular.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "tukeylambda":
                        gen_data = tukeylambda.rvs(
                            best_parameters["lambda_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "rdist":
                        gen_data = rdist.rvs(
                            best_parameters["c_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    else:
                        gen_data = best_parameters["random_factor"]
                    try:
                        X_train_sample_class[column_name] = gen_data
                    except UnboundLocalError:
                        X_train_sample_class[column_name] = 0
                    temp_df_list.append(X_train_sample_class)

            else:
                bin_encoder = 1
                for X_train_sample_class in dfs:
                    size = len(X_train_sample_class.index)
                    class_inst = bin_encoder
                    bin_encoder += 1
                    if best_parameters["sample_distribution"] == "Uniform":
                        gen_data = np.full((size,), best_parameters["uniformity"])
                    elif best_parameters["sample_distribution"] == "Binomial":
                        gen_data = binom.rvs(
                            n=random_factor_pos, p=best_parameters["p_value"], size=size
                        )
                    elif best_parameters["sample_distribution"] == "Poisson":
                        gen_data = poisson.rvs(mu=best_parameters["mu"], size=size)
                    elif best_parameters["sample_distribution"] == "Exponential":
                        gen_data = expon.rvs(
                            scale=best_parameters["scale"],
                            loc=best_parameters["location"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "Gamma":
                        best_parameters["sample_distribution"] = gamma.rvs(
                            a=best_parameters["mu"], size=size
                        )
                    elif best_parameters["sample_distribution"] == "Uniform":
                        gen_data = class_inst
                    elif best_parameters["sample_distribution"] == "Normal":
                        gen_data = norm.rvs(
                            size=size,
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                        )
                    elif best_parameters["sample_distribution"] == "Pareto":
                        gen_data = pareto.rvs(best_parameters["parteo_b"], size=size)
                    elif best_parameters["sample_distribution"] == "Levy":
                        gen_data = levy.rvs(size=size)
                        if best_parameters["random_or_control_factor"] == "Random":
                            gen_data = gen_data * random_factor_pos
                        else:
                            gen_data += class_inst * 2
                    elif best_parameters["sample_distribution"] == "dweibull":
                        gen_data = dweibull.rvs(
                            best_parameters["pos_only_location"], size=size
                        )
                    elif best_parameters["sample_distribution"] == "halfcauchy":
                        gen_data = halfcauchy.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "halfnorm":
                        gen_data = halfnorm.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "powernorm":
                        gen_data = powernorm.rvs(
                            best_parameters["c_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "semicircular":
                        gen_data = semicircular.rvs(
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "tukeylambda":
                        gen_data = tukeylambda.rvs(
                            best_parameters["lambda_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    elif best_parameters["sample_distribution"] == "rdist":
                        gen_data = rdist.rvs(
                            best_parameters["c_value"],
                            loc=best_parameters["location"],
                            scale=best_parameters["scale"],
                            size=size,
                        )
                    else:
                        gen_data = best_parameters["random_factor"]

                    try:
                        X_train_sample_class[column_name] = gen_data
                    except UnboundLocalError:
                        X_train_sample_class[column_name] = 0
                    temp_df_list.append(X_train_sample_class)

            temp_df = pd.concat(temp_df_list, ignore_index=False)
            temp_df = temp_df.drop(self.target_variable, axis=1)

            # save copy of ol column
            original_col = X_train[column_name].copy()
            X_train[column_name] = temp_df[column_name]
            Y_train = X_train[self.target_variable]
            X_train = X_train.drop(self.target_variable, axis=1)
            if self.rapids_acceleration:
                X_train = X_train.astype("float32")

            try:
                # get train scores
                scores_2 = cross_val_score(
                    model_3_copy, X_train, Y_train, cv=10, scoring=metric
                )
                mae_2 = np.mean(scores_2)
                train_mae = mae_2 * 100
            except Exception:
                train_mae = 0

            # test scores
            model_3_copy.fit(X_train, Y_train)
            scores_2_test = model_3_copy.predict(X_test)
            try:
                matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
            except Exception:
                matthew_2 = 0
            test_mae = matthew_2 * 100
            synthetic_mae = (train_mae + test_mae) / 2 - (
                abs(train_mae - test_mae)
            ) ** 3

            print(f"The synthetic score is {synthetic_mae}.")

            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

            # sort by index to return in correct order
            X_train = X_train.sort_index()

            # original data or synthetic data?
            if (
                synthetic_mae
                > self.preprocess_decisions[
                    "synthetic_augmentation_parameters_benchmark"
                ]
            ):
                print("Keep synthetic column.")
                self.preprocess_decisions["synthetic_augmentation_parameters"][
                    column_name
                ] = best_parameters
                self.preprocess_decisions[
                    "synthetic_augmentation_parameters_benchmark"
                ] = synthetic_mae
                return X_train[column_name]
            else:
                print("Keep original column. No improvement found.")
                return original_col

    def synthetic_data_augmentation(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if self.prediction_mode:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            logging.info("Start creating synthetic data.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.preprocess_decisions["synthetic_augmentation_parameters"] = {}
            self.preprocess_decisions["synthetic_augmentation_parameters_benchmark"] = 0
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            data_size = len(X_train.index)

            # get sample size to run brute force feature selection against
            if self.brute_force_selection_sample_size > data_size:
                sample_size = len(X_train.index)
            else:
                sample_size = self.brute_force_selection_sample_size

            # get copy of the dataframe
            self.preprocess_decisions["random_state_counter"] = 0

            # sort by index so returned column can match original dataframe
            Y_train = Y_train.sort_index()
            X_train = X_train.sort_index()
            # get columns which are floats and from the original dataset
            float_cols = [
                x
                for x, y in self.detected_col_types.items()
                if y in ["float", "int", "bool"]
                if x in X_train.columns.to_list()
            ]
            print(
                f"Synthetic augmentation will be executed on the following columns: {float_cols}"
            )
            num_float_cols = len(float_cols)

            for col in X_train[float_cols].columns.to_list():
                if self.detected_col_types[col] == "float":
                    print(
                        f"Started augmenting column {col}. Progress: {round(((self.preprocess_decisions['random_state_counter']+1)/num_float_cols)*100, 2)}%"
                    )
                    self.preprocess_decisions["random_state_counter"] += 1
                    self.set_random_seed()
                    binom.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    poisson.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    expon.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    gamma.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    norm.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    pareto.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    levy.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    dweibull.random_state = np.random.RandomState(
                        seed=self.preprocess_decisions["random_state_counter"]
                    )
                    X_train[col] = self.synthetic_floating_data_generator(
                        column_name=col, sample_size=sample_size
                    )
                    self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
                    print(f"Finished augmenting column {col}")
                else:
                    print(
                        f"Skipped augmentation for column {col}, because {col} is not of type float."
                    )
            print("Export training data with synthetic optimized features.")
            optuna.logging.set_verbosity(optuna.logging.INFO)
            # shuffle dataframe for Tabnet
            X_train[self.target_variable] = Y_train
            X_train = X_train.sample(frac=1.0, random_state=42)
            X_train = X_train.reset_index(drop=True)
            Y_train = X_train[self.target_variable].copy()
            X_train = X_train.drop(self.target_variable, axis=1)
            try:
                del X_train[self.target_variable]
            except KeyError:
                pass
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def automated_feature_selection(  # noqa: C901
        self, model=None, metric=None, numeric_only=False
    ):
        """
        Uses boostaroota algorithm to automatically chose best features. boostaroota choses XGboost under
        the hood.
        :param metric: Metric to evaluate strength of features.
        :param float_only: If True, the feature selection will consider integer, floating and boolean columns only.
        :return: Returns reduced dataframe.
        """
        self.get_current_timestamp("Select best features")
        if self.prediction_mode:
            logging.info("Start filtering for preselected columns.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            if numeric_only:
                self.dataframe = self.dataframe[
                    self.preprocess_decisions["early_selected_features"]
                ]
            else:
                self.dataframe = self.dataframe[self.selected_feats]
            logging.info("Finished filtering preselected columns.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            logging.info("Start automated feature selection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            # get sample size to run brute force feature selection against
            if self.feature_selection_sample_size > len(X_train.index):
                sample_size = len(X_train.index)
            else:
                sample_size = self.feature_selection_sample_size

            X_train_sample = X_train.copy()
            X_train_sample[self.target_variable] = Y_train
            X_train_sample = X_train_sample.sample(sample_size, random_state=42)
            Y_train_sample = X_train_sample[self.target_variable]
            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            for col in X_train.columns:
                print(f"Features before selection are...{col}")
            if model:
                model = model
                metric = metric
                br = BoostARoota(clf=model)
            elif self.class_problem == "binary":
                if self.feature_selection_backend == "lgbm":
                    model = lgb.LGBMClassifier(random_state=42, objective="binary")
                    br = BoostARoota(clf=model)
                elif self.feature_selection_backend == "xgboost":
                    br = BoostARoota(metric="logloss")
            elif self.class_problem == "multiclass":
                if self.feature_selection_backend == "lgbm":
                    model = lgb.LGBMClassifier(random_state=42, objective="multiclass")
                    br = BoostARoota(clf=model)
                elif self.feature_selection_backend == "xgboost":
                    br = BoostARoota(metric="mlogloss")
            elif self.class_problem == "regression":
                if self.feature_selection_backend == "lgbm":
                    model = lgb.LGBMRegressor(random_state=42, objective="regression")
                    br = BoostARoota(clf=model)
                elif self.feature_selection_backend == "xgboost":
                    br = BoostARoota(metric="mae")

            if numeric_only:
                # get columns which are floats and from the original dataset
                float_cols = [
                    x
                    for x, y in self.detected_col_types.items()
                    if y in ["float", "int", "bool"]
                    if x in X_train.columns.to_list()
                ]
                other_cols = [
                    x for x in X_train.columns.to_list() if x not in float_cols
                ]
                X_train_temp = X_train_sample[float_cols]
                br.fit(X_train_temp, Y_train_sample)
                selected = br.keep_vars_
                all_cols = selected.values.tolist() + other_cols
                X_train = X_train[all_cols]
                X_test = X_test[all_cols]
                self.preprocess_decisions["early_selected_features"] = all_cols
                for i in selected:
                    print(f" Selected features are... {i}.")
            else:
                br.fit(X_train_sample, Y_train_sample)
                selected = br.keep_vars_
                X_train = X_train[selected]
                X_test = X_test[selected]
                self.selected_feats = selected
                for i in selected:
                    print(f" Selected features are... {i}.")
            logging.info("Finished automated feature selection.")
            del br
            del X_train_sample
            del Y_train_sample
            try:
                del X_train_temp
            except Exception:
                pass
            _ = gc.collect()
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return (
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test),
                self.selected_feats,
            )

    def bruteforce_random_feature_selection(self, metric=None):  # noqa: C901
        """
        Takes a dataframe or a sample of it. Select randomly features and runs Vowpal Wabbit on it with 10-fold cross
        validation. Evaluates performance and optimizes incrementally feature selection by learning from previous attempts.
        :param metric: Scoring metric for cross validation. Must be compatible for Sklearn's cross_val_score.
        :return: Updates class attributes
        """
        self.get_current_timestamp("Select best features")
        if self.prediction_mode:
            logging.info("Start filtering for final preselected columns.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.dataframe = self.dataframe[self.selected_feats]
            logging.info("Finished filtering final preselected columns.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.dataframe
        else:
            logging.info("Start Vowpal bruteforce feature selection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            for col in X_train.columns:
                print(f"Features before selection are...{col}")

            problem = self.class_problem
            if metric:
                metric = metric
            elif self.class_problem == "binary":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "multiclass":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "regression":
                metric = "neg_mean_squared_error"

            data_size = len(X_train.index)

            # get sample size to run brute force feature selection against
            if self.brute_force_selection_sample_size > data_size:
                sample_size = len(X_train.index)
            else:
                sample_size = self.brute_force_selection_sample_size

            X_train_sample = X_train.copy()
            X_train_sample[self.target_variable] = Y_train
            X_train_sample = X_train_sample.sample(sample_size, random_state=42)
            Y_train_sample = X_train_sample[self.target_variable]
            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            all_cols = X_train.columns.to_list()

            brute_force_selection_base_learner = self.brute_force_selection_base_learner

            def objective(trial):
                param = {}
                for col in all_cols:
                    param[col] = trial.suggest_int(col, 0, 1)

                if brute_force_selection_base_learner == "auto":
                    base_learner = trial.suggest_categorical(
                        "base_learner", ["lgbm", "vowal_wobbit"]
                    )
                elif brute_force_selection_base_learner == "double":
                    base_learner = None
                else:
                    base_learner = brute_force_selection_base_learner

                if base_learner == "lgbm":
                    if problem == "binary" or problem == "multiclass":
                        model = lgb.LGBMClassifier(random_state=1000)
                    else:
                        model = lgb.LGBMRegressor(random_state=1000)
                elif base_learner == "vowal_wobbit":
                    if problem == "binary" or problem == "multiclass":
                        model = VWClassifier()
                    else:
                        model = VWRegressor()
                elif brute_force_selection_base_learner == "double":
                    if problem == "binary" or problem == "multiclass":
                        model_1 = VWClassifier()
                        model_2 = lgb.LGBMClassifier(random_state=1000)
                    else:
                        model_1 = VWRegressor()
                        model_2 = lgb.LGBMRegressor(random_state=1000)
                else:
                    if problem == "binary" or problem == "multiclass":
                        model = VWClassifier()
                    else:
                        model = VWRegressor()

                temp_features = []
                for k, v in param.items():
                    if v == 1:
                        temp_features.append(k)
                    else:
                        pass

                if brute_force_selection_base_learner == "double":
                    try:
                        scores_1 = cross_val_score(
                            model_1,
                            X_train_sample[temp_features],
                            Y_train_sample,
                            cv=10,
                            scoring=metric,
                        )
                        scores_2 = cross_val_score(
                            model_2,
                            X_train_sample[temp_features],
                            Y_train_sample,
                            cv=10,
                            scoring=metric,
                        )
                        mae_1 = np.mean(scores_1)
                        mae_2 = np.mean(scores_2)
                        mae = (mae_1 + mae_2) / 2
                    except Exception:
                        mae = 0
                    return mae
                else:
                    try:
                        scores = cross_val_score(
                            model,
                            X_train_sample[temp_features],
                            Y_train_sample,
                            cv=10,
                            scoring=metric,
                        )
                        mae = np.mean(scores)
                    except Exception:
                        mae = 0
                    return mae

            algorithm = "bruteforce_random"

            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=42, consider_endpoints=True
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
            )
            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds[algorithm],
                timeout=self.hyperparameter_tuning_max_runtime_secs[algorithm],
                gc_after_trial=True,
                show_progress_bar=True,
            )
            self.optuna_studies[f"{algorithm}"] = {}
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                fig.show()
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            if len(X_train.columns) <= 1000:
                fig = optuna.visualization.plot_param_importances(study)
                self.optuna_studies[f"{algorithm}_param_importance"] = fig
                fig.show()
            else:
                print(
                    "Skipped calculating feature importance due to expected runtime being too long."
                )

            best_feature_combination = study.best_trial.params
            final_features = []
            for k, v in best_feature_combination.items():
                if v == 1:
                    final_features.append(k)
                else:
                    pass

            X_train = X_train[final_features].copy()
            X_test = X_test[final_features].copy()
            _ = gc.collect()
            self.selected_feats = final_features
            for i in final_features:
                print(f" Final features are... {i}.")
            logging.info("Finished bruteforce random feature selection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            optuna.logging.set_verbosity(optuna.logging.INFO)
            return (
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test),
                self.selected_feats,
            )

    def delete_unpredictable_training_rows(self):
        self.get_current_timestamp("Delete bad training rows")
        logging.info("Started deleting bad training rows.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        def get_test_result(Y_test, scores_2_test):
            try:
                if self.class_problem in ["binary", "multiclass"]:
                    matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
                    test_mae = matthew_2 * 100
                else:
                    test_mae = mean_squared_error(Y_test, scores_2_test, squared=True)
                    test_mae *= -1
            except Exception:
                test_mae = 0
            return test_mae

        if self.prediction_mode:
            pass
        else:
            # from sklearn.linear_model import Ridge

            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            columns = X_train.columns.to_list()
            if self.class_problem == "binary" or self.class_problem == "multiclass":
                model = lgb.LGBMClassifier()
                # model_3 = RidgeClassifier()
            else:
                model = lgb.LGBMRegressor()
                # model_3 = Ridge()

            total_rounds = 300
            fold_cols_created = []
            for sample in range(total_rounds):
                temp_results = []
                try:
                    X_train_sample = X_train.sample(frac=0.3, random_state=sample)
                    X_train_sample = X_train_sample[columns]
                    y_train_sample = Y_train.iloc[X_train_sample.index]
                    model.fit(X_train_sample, y_train_sample)
                    preds = model.predict(X_test)
                    temp_results.append(get_test_result(Y_test, preds))
                    # if self.class_problem in ['regression']:
                    #    model_3.fit(X_train_sample, y_train_sample)
                    #    preds = model_3.predict(X_test)
                    #    temp_results.append(get_test_result(Y_test, preds))
                    mean_results = np.mean(np.array(temp_results))
                    X_train_sample[f"fold_result_{sample}"] = mean_results
                    X_train = X_train.merge(
                        X_train_sample[f"fold_result_{sample}"],
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    fold_cols_created.append(f"fold_result_{sample}")
                    print(
                        f"Started epoch {sample} from {total_rounds} with score of {mean_results}."
                    )
                except Exception:
                    pass

            if self.class_problem in ["binary", "multiclass"]:
                temp_dfs = []
                X_train[self.target_variable] = Y_train
                for one_class in X_train[self.target_variable].unique():
                    class_df = X_train[
                        (X_train[self.target_variable] == one_class)
                    ].copy()
                    original_len = len(class_df.index)
                    class_df["all_sample_mean"] = class_df[fold_cols_created].mean(
                        axis=1, skipna=True
                    )
                    std = class_df["all_sample_mean"].std()
                    class_df = class_df[
                        (
                            class_df["all_sample_mean"]
                            > (class_df["all_sample_mean"].mean() - 1.96 * std)
                        )
                    ]
                    temp_dfs.append(class_df)
                    new_len = len(class_df.index)
                    print(
                        f"Class {one_class} reduced from {original_len} to {new_len} samples."
                    )
                X_train = pd.concat(temp_dfs)
                X_train = X_train.sample(frac=1.0, random_state=42)
                X_train = X_train.reset_index(drop=True)
                Y_train = X_train[self.target_variable].copy()
                X_train = X_train.drop(self.target_variable, axis=1)
                try:
                    del X_train[self.target_variable]
                except KeyError:
                    pass
                X_train = X_train[columns].copy()
            else:
                X_train["all_sample_mean"] = X_train[fold_cols_created].mean(
                    axis=1, skipna=True
                )
                quantile = X_train["all_sample_mean"].quantile(0.20)
                X_train = X_train[(X_train["all_sample_mean"] > quantile)][columns]
                X_train[self.target_variable] = Y_train
                X_train = X_train.reset_index(drop=True)
                Y_train = X_train[self.target_variable]
                X_train = X_train.drop(self.target_variable, axis=1)
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            logging.info("Finished deleting bad training rows.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def get_hyperparameter_tuning_sample_df(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if self.hyperparameter_tuning_sample_size > len(X_train.index):
                sample_size = len(X_train.index)
            else:
                sample_size = self.hyperparameter_tuning_sample_size

            X_train[self.target_variable] = Y_train
            X_train_sample = X_train.sample(sample_size, random_state=42).copy()
            X_train_sample = X_train_sample.reset_index(drop=True)
            Y_train_sample = X_train_sample[self.target_variable]  # .copy()

            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)
            # Y_train_sample = Y_train_sample.reset_index(drop=True)

            try:
                del X_train[self.target_variable]
            except KeyError:
                pass
            return X_train_sample, Y_train_sample

    def autoencoder_based_oversampling(self):  # noqa: C901
        if self.prediction_mode:
            pass
        else:

            class DataBuilder(Dataset):
                def __init__(self, dataset):
                    self.x = dataset.values
                    self.x = torch.from_numpy(self.x).to(torch.float)
                    self.len = self.x.shape[0]

                def __getitem__(self, index):
                    return self.x[index]

                def __len__(self):
                    return self.len

            class Autoencoder(nn.Module):
                def __init__(self, D_in, H=50, H2=12, latent_dim=3):

                    # Encoder
                    super(Autoencoder, self).__init__()
                    self.linear1 = nn.Linear(D_in, H)
                    self.lin_bn1 = nn.BatchNorm1d(num_features=H)
                    self.linear2 = nn.Linear(H, H2)
                    self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
                    self.linear3 = nn.Linear(H2, H2)
                    self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

                    # Latent vectors mu and sigma
                    self.fc1 = nn.Linear(H2, latent_dim)
                    self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
                    self.fc21 = nn.Linear(latent_dim, latent_dim)
                    self.fc22 = nn.Linear(latent_dim, latent_dim)

                    # Sampling vector
                    self.fc3 = nn.Linear(latent_dim, latent_dim)
                    self.fc_bn3 = nn.BatchNorm1d(latent_dim)
                    self.fc4 = nn.Linear(latent_dim, H2)
                    self.fc_bn4 = nn.BatchNorm1d(H2)

                    # Decoder
                    self.linear4 = nn.Linear(H2, H2)
                    self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
                    self.linear5 = nn.Linear(H2, H)
                    self.lin_bn5 = nn.BatchNorm1d(num_features=H)
                    self.linear6 = nn.Linear(H, D_in)
                    self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

                    self.relu = nn.ReLU()

                def encode(self, x):
                    lin1 = self.relu(self.lin_bn1(self.linear1(x)))
                    lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
                    lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

                    fc1 = F.relu(self.bn1(self.fc1(lin3)))

                    r1 = self.fc21(fc1)
                    r2 = self.fc22(fc1)

                    return r1, r2

                def reparameterize(self, mu, logvar):
                    if self.training:
                        std = logvar.mul(0.5).exp_()
                        eps = Variable(std.data.new(std.size()).normal_())
                        return eps.mul(std).add_(mu)
                    else:
                        return mu

                def decode(self, z):
                    fc3 = self.relu(self.fc_bn3(self.fc3(z)))
                    fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

                    lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
                    lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
                    return self.lin_bn6(self.linear6(lin5))

                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    return self.decode(z), mu, logvar

            class customLoss(nn.Module):
                def __init__(self):
                    super(customLoss, self).__init__()
                    self.mse_loss = nn.MSELoss(reduction="sum")

                def forward(self, x_recon, x, mu, logvar):
                    loss_MSE = self.mse_loss(x_recon, x)
                    loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    return loss_MSE + loss_KLD

            def random_noise(a_series, noise_reduction=1000000):
                return (
                    np.random.random(len(a_series)) * a_series.std() / noise_reduction
                ) - (a_series.std() / (2 * noise_reduction))

            def handle_rarity(
                all_data, threshold=10, mask_as=99, rarity_cols=None, normalize=False
            ):
                if isinstance(rarity_cols, list):
                    for col in rarity_cols:
                        frequencies = all_data[col].value_counts(normalize=normalize)
                        condition = frequencies < threshold
                        mask_obs = frequencies[condition].index
                        mask_dict = dict.fromkeys(mask_obs, mask_as)
                        all_data[col] = all_data[col].replace(
                            mask_dict
                        )  # or you could make a copy not to modify original data
                    del rarity_cols
                else:
                    pass
                return all_data

            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            cols = X_train.columns

            if self.class_problem == "binary" or self.class_problem == "multiclass":
                # in this part we count how much less rows than the main class we have for all other classes
                # get unique values
                unique_classes = np.unique(Y_train.values)
                # get counts
                unique, counts = np.unique(Y_train.values, return_counts=True)
                results = np.asarray((unique, counts))
                # get highest count
                results[1].max()
                # get array with delta of each element count compared to max count
                max_count = results[1].max()
                deltas = max_count - results[1]
                # class_deltas = np.vstack(
                #     (results[0], deltas)
                # )  # contains classes and how much they miss until max count
            else:
                X_train[self.target_variable] = Y_train
                # sort on A
                X_train.sort_values(self.target_variable, inplace=True)

                for bins in [10, 9, 8, 7, 6, 5, 4, 3, 2]:
                    # create bins
                    X_train["bin"] = pd.cut(
                        X_train[self.target_variable], bins, labels=False
                    )
                    bins_dist = X_train["bin"].value_counts()
                    print(
                        f"For {bins} bins the smallest bin contains {bins_dist.min()} samples."
                    )
                    if bins_dist.min() > 5:
                        print(f"Enough minimum samples found with {bins} bins.")
                        break
                    else:
                        print(
                            f"Not enough minimum samples found with {bins} bins. Reducing no. of bins..."
                        )

                if bins_dist.min() == 2:
                    X_train = handle_rarity(
                        X_train, rarity_cols=["bin"], threshold=5, mask_as=99
                    )

                Y_train_original = X_train[self.target_variable]
                Y_train = X_train["bin"]  # .astype('float')

                # Y_train_original = X_train[self.target_variable]
                X_train = X_train.drop(self.target_variable, axis=1)

                unique_classes = X_train["bin"].unique()
                print("Unique classes are...")
                print(X_train["bin"].value_counts())
                # get counts
                unique, counts = np.unique(X_train["bin"].values, return_counts=True)
                results = np.asarray((unique, counts))
                # get highest count
                results[1].max()
                # get array with delta of each element count compared to max count
                max_count = results[1].max()
                deltas = max_count - results[1]
                # class_deltas = np.vstack(
                #     (results[0], deltas)
                # )  # contains classes and how much they miss until max count
                X_train = X_train.drop("bin", axis=1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # torch.cuda.set_device(0)

            executed_classes = 0

            for i in range(len(unique_classes)):
                executed_classes += 1
                print(
                    f"""Starting oversampling for class {i}.
                Progress after this step is {round((executed_classes/len(unique_classes))*100, 2)}%."""
                )
                target_class = unique_classes[i]
                target_delta = deltas[i]
                if target_delta == 0:
                    pass
                else:
                    X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

                    if self.class_problem == "regression":
                        X_train[self.target_variable] = Y_train
                        bins_needed = 0
                        for bins in [10, 9, 8, 7, 6, 5, 4, 3, 2]:
                            # create bins
                            X_train["bin"] = pd.cut(
                                X_train[self.target_variable], bins, labels=False
                            )
                            bins_dist = X_train["bin"].value_counts()
                            print(
                                f"For {bins} bins the smallest bin contains {bins_dist.min()} samples."
                            )
                            if bins_dist.min() > 5:
                                print(f"Enough minimum samples found with {bins} bins.")
                                bins_needed = bins
                                break
                            else:
                                print(
                                    f"Not enough minimum samples found with {bins} bins. Reducing no. of bins..."
                                )

                            if bins_dist.min() == 2:
                                X_train = handle_rarity(
                                    X_train,
                                    rarity_cols=["bin"],
                                    threshold=5,
                                    mask_as=99,
                                )
                                bins_needed = 2

                        Y_train_original = X_train[self.target_variable]
                        Y_train = X_train["bin"]  # .astype('float')
                        X_train = X_train.drop(self.target_variable, axis=1)

                        unique_classes = X_train["bin"].unique().tolist()
                        # get counts
                        unique, counts = np.unique(
                            X_train["bin"].values, return_counts=True
                        )
                        X_train = X_train.drop("bin", axis=1)
                        results = np.asarray((unique, counts))
                        # get highest count
                        results[1].max()
                        # get array with delta of each element count compared to max count
                        max_count = results[1].max()
                        deltas = max_count - results[1]
                        # class_deltas = np.vstack(
                        #     (results[0], deltas)
                        # )  # contains classes and how much they miss until max count

                        X_test[self.target_variable] = Y_test

                        if bins_dist.min() == 2:
                            X_test = handle_rarity(
                                X_test, rarity_cols=["bin"], threshold=5, mask_as=99
                            )
                        else:
                            X_test["bin"] = X_test["bin"] = pd.cut(
                                X_test[self.target_variable]
                                + random_noise(X_test[self.target_variable]),
                                bins_needed,
                                labels=False,
                            )
                        Y_test_original = X_test[self.target_variable]
                        Y_test = X_test["bin"]  # .astype('float')
                        X_test = X_test.drop(self.target_variable, axis=1)
                        X_test = X_test.drop("bin", axis=1)

                    X_train_class_only = X_train.iloc[
                        np.where(Y_train == target_class)[0]
                    ]
                    Y_train_class_only = Y_train.iloc[
                        np.where(Y_train == target_class)[0]
                    ]

                    X_train_other_classes = X_train.iloc[
                        np.where(Y_train != target_class)[0]
                    ]
                    Y_train_other_classes = Y_train.iloc[
                        np.where(Y_train != target_class)[0]
                    ]

                    D_in = X_train_class_only.shape[1]
                    X_test_class_only = X_test.iloc[np.where(Y_test == target_class)[0]]

                    traindata_set = DataBuilder(X_train_class_only)
                    testdata_set = DataBuilder(X_test_class_only)

                    trainloader = DataLoader(dataset=traindata_set, batch_size=256)
                    testloader = DataLoader(dataset=testdata_set, batch_size=256)

                    # D_in = X_train_class_only.shape[1]
                    # HYPERPARAMETER OPTIMIZATION
                    def objective(trial):
                        optimizer_choice = trial.suggest_categorical(
                            "optimizer_choice", ["Adam", "AdamW", "RMSprop"]
                        )
                        param = {
                            "nb_epochs": trial.suggest_int("nb_epochs", 2, 5000),
                            "h": trial.suggest_int("h", 20, 50),
                            "h2": trial.suggest_int("h2", 2, 19),
                            "latent_dim": trial.suggest_int("latent_dim", 1, 3),
                            "optim_weight_decay": trial.suggest_uniform(
                                "optim_weight_decay", 0.0, 0.9
                            ),
                            "optim_learning_rate": trial.suggest_loguniform(
                                "optim_learning_rate", 1e-5, 0.3
                            ),
                        }
                        model = Autoencoder(
                            D_in, param["h"], param["h2"], param["latent_dim"]
                        ).to(device)
                        if optimizer_choice == "Adam":
                            optimizer = optim.Adam(
                                model.parameters(),
                                lr=param["optim_learning_rate"],
                                weight_decay=param["optim_weight_decay"],
                            )
                        elif optimizer_choice == "RMSprop":
                            optimizer = optim.RMSprop(
                                model.parameters(),
                                lr=param["optim_learning_rate"],
                                weight_decay=param["optim_weight_decay"],
                            )
                        elif optimizer_choice == "LBFGS":
                            optimizer = optim.LBFGS(
                                model.parameters(), lr=param["optim_learning_rate"]
                            )
                        elif optimizer_choice == "SGD":
                            optimizer = optim.SGD(
                                model.parameters(), lr=param["optim_learning_rate"]
                            )
                        elif optimizer_choice == "SparseAdam":
                            optimizer = optim.SparseAdam(
                                model.parameters(), lr=param["optim_learning_rate"]
                            )
                        elif optimizer_choice == "AdamW":
                            optimizer = optim.AdamW(
                                model.parameters(), lr=param["optim_learning_rate"]
                            )
                        else:
                            optimizer = optim.Adam(
                                model.parameters(), lr=param["optim_learning_rate"]
                            )

                        model = Autoencoder(
                            D_in, param["h"], param["h2"], param["latent_dim"]
                        ).to(device)
                        loss_mse = customLoss()

                        # train model
                        # log_interval = 1000
                        # val_losses = []
                        train_losses = []
                        test_losses = []

                        def train(epoch):
                            model.train()
                            train_loss = 0
                            for data in trainloader:
                                data = data.to(device)
                                optimizer.zero_grad()
                                recon_batch, mu, logvar = model(data)
                                loss = loss_mse(recon_batch, data, mu, logvar)
                                loss.backward()
                                train_loss += loss.item()
                                optimizer.step()
                                if epoch % 200 == 0:
                                    print(
                                        "====> Epoch: {} Average training loss: {:.4f}".format(
                                            epoch, train_loss / len(trainloader.dataset)
                                        )
                                    )
                                train_losses.append(
                                    train_loss / len(trainloader.dataset)
                                )
                                torch.cuda.empty_cache()

                        def test(epoch):
                            with torch.no_grad():
                                test_loss = 0
                                for data in testloader:
                                    data = data.to(device)
                                    optimizer.zero_grad()
                                    recon_batch, mu, logvar = model(data)
                                    loss = loss_mse(recon_batch, data, mu, logvar)
                                    test_loss += loss.item()
                                    if epoch % 200 == 0:
                                        print(
                                            "====> Epoch: {} Average test loss: {:.4f}".format(
                                                epoch,
                                                test_loss / len(testloader.dataset),
                                            )
                                        )
                                    trial.report(test_loss, epoch)
                                    if trial.should_prune():
                                        raise optuna.exceptions.TrialPruned()
                                    test_losses.append(
                                        test_loss / len(testloader.dataset)
                                    )
                                    torch.cuda.empty_cache()

                        epochs = param["nb_epochs"]
                        for epoch in range(1, epochs + 1):
                            train(epoch)
                            test(epoch)

                        with torch.no_grad():
                            for data in testloader:
                                data = data.to(device)
                                optimizer.zero_grad()
                                recon_batch, mu, logvar = model(data)

                            try:
                                loss_score = test_losses[
                                    -1
                                ]  # np.sum(test_losses[-1] + abs(np.sum([test_losses[-1], train_losses[-1]]))**2)
                            except IndexError:
                                loss_score = 10000000000

                            return loss_score  # test_losses[-1]

                    algorithm = "autoencoder_based_oversampling"

                    sampler = optuna.samplers.TPESampler(
                        multivariate=True, seed=42, consider_endpoints=True
                    )
                    study = optuna.create_study(
                        direction="minimize", sampler=sampler, study_name=f"{algorithm}"
                    )
                    study.optimize(
                        objective,
                        n_trials=self.hyperparameter_tuning_rounds[algorithm],
                        timeout=self.hyperparameter_tuning_max_runtime_secs[algorithm],
                        gc_after_trial=True,
                        show_progress_bar=True,
                    )
                    self.optuna_studies[f"{algorithm}"] = {}
                    try:
                        fig = optuna.visualization.plot_optimization_history(study)
                        self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                        fig.show()
                    except ZeroDivisionError:
                        print(
                            "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                        )

                    # FINAL TRAINING
                    best_parameters = study.best_trial.params
                    H = best_parameters["h"]
                    H2 = best_parameters["h2"]
                    latent_dim = best_parameters["latent_dim"]

                    model = Autoencoder(D_in, H, H2, latent_dim).to(device)
                    if best_parameters["optimizer_choice"] == "Adam":
                        optimizer = optim.Adam(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                            weight_decay=best_parameters["optim_weight_decay"],
                        )
                    elif best_parameters["optimizer_choice"] == "RMSprop":
                        optimizer = optim.RMSprop(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                            weight_decay=best_parameters["optim_weight_decay"],
                        )
                    elif best_parameters["optimizer_choice"] == "LBFGS":
                        optimizer = optim.LBFGS(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                        )
                    elif best_parameters["optimizer_choice"] == "SGD":
                        optimizer = optim.SGD(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                        )
                    elif best_parameters["optimizer_choice"] == "SparseAdam":
                        optimizer = optim.SparseAdam(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                        )
                    elif best_parameters["optimizer_choice"] == "AdamW":
                        optimizer = optim.AdamW(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                        )
                    else:
                        optimizer = optim.Adam(
                            model.parameters(),
                            lr=best_parameters["optim_learning_rate"],
                        )

                    loss_mse = customLoss()

                    # train model
                    # log_interval = 50
                    # val_losses = []
                    train_losses = []
                    test_losses = []

                    def train(epoch):
                        model.train()
                        train_loss = 0
                        for data in trainloader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            recon_batch, mu, logvar = model(data)
                            loss = loss_mse(recon_batch, data, mu, logvar)
                            loss.backward()
                            train_loss += loss.item()
                            optimizer.step()
                        if epoch % 200 == 0:
                            print(
                                "====> Epoch: {} Average training loss: {:.4f}".format(
                                    epoch, train_loss / len(trainloader.dataset)
                                )
                            )
                            train_losses.append(train_loss / len(trainloader.dataset))

                    def test(epoch):
                        with torch.no_grad():
                            test_loss = 0
                            for data in testloader:
                                data = data.to(device)
                                optimizer.zero_grad()
                                recon_batch, mu, logvar = model(data)
                                loss = loss_mse(recon_batch, data, mu, logvar)
                                test_loss += loss.item()
                                if epoch % 200 == 0:
                                    print(
                                        "====> Epoch: {} Average test loss: {:.4f}".format(
                                            epoch, test_loss / len(testloader.dataset)
                                        )
                                    )
                                test_losses.append(test_loss / len(testloader.dataset))

                    epochs = best_parameters["nb_epochs"]
                    for epoch in range(1, epochs + 1):
                        train(epoch)
                        test(epoch)

                    with torch.no_grad():
                        for data in testloader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            recon_batch, mu, logvar = model(data)

                    recon_row = recon_batch[0].cpu().numpy()
                    recon_row = np.append(recon_row, [1])
                    real_row = testloader.dataset.x[0].cpu().numpy()
                    real_row = np.append(real_row, [1])

                    sigma = torch.exp(logvar / 2)

                    # sample z from q
                    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
                    z = q.rsample(sample_shape=torch.Size([int(target_delta)]))

                    with torch.no_grad():
                        pred = model.decode(z).cpu().numpy()

                    if self.class_problem == "regression":
                        Y_train_class_only = Y_train_original.iloc[
                            np.where(Y_train == target_class)[0]
                        ]
                        Y_train_other_classes = Y_train_original.iloc[
                            np.where(Y_train != target_class)[0]
                        ]
                        Y_test = Y_test_original
                    else:
                        pass

                    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
                    z = q.rsample(sample_shape=torch.Size([int(target_delta)]))

                    with torch.no_grad():
                        pred = model.decode(z).cpu().numpy()

                    X_train = np.vstack((X_train_class_only.values, pred))
                    X_train = np.vstack((X_train, X_train_other_classes))
                    X_train = pd.DataFrame(X_train, columns=cols)

                    Y_train = np.append(
                        Y_train_class_only.values, np.repeat(target_class, target_delta)
                    )
                    Y_train = np.append(Y_train, Y_train_other_classes)
                    Y_train = pd.Series(Y_train)

                    X_train[self.target_variable] = Y_train
                    X_train = X_train.sample(frac=1.0, random_state=42)
                    X_train = X_train.reset_index(drop=True)
                    Y_train = X_train[self.target_variable].copy()
                    X_train = X_train.drop(self.target_variable, axis=1)
                    try:
                        del X_train[self.target_variable]
                    except KeyError:
                        pass

                    self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def final_kernel_pca_dimensionality_reduction(self):  # noqa: C901
        logging.info("Start final PCA dimensionality reduction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            best_parameters = self.preprocess_decisions[
                "final_kernel_pca_dimensionality_reduction_reduction_parameters"
            ]
            pca = self.preprocess_decisions[
                "final_kernel_pca_dimensionality_reduction_model"
            ]
            dataframe_comps = pca.transform(self.dataframe)
            new_cols = [f"PCA_{i}" for i in range(dataframe_comps.shape[1])]
            self.dataframe = pd.DataFrame(dataframe_comps, columns=new_cols)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if self.final_pca_dimensionality_reduction_sample_size > len(X_train.index):
                sample_size = len(X_train.index)
            else:
                sample_size = self.final_pca_dimensionality_reduction_sample_size

            print(
                f"Number of columns before dimensionality reduction is: {len(X_train.columns.to_list())}"
            )

            X_train_sample = X_train.copy()
            X_train_sample[self.target_variable] = Y_train
            X_train_sample = X_train_sample.sample(sample_size, random_state=42)
            Y_train_sample = X_train_sample[self.target_variable]
            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            if self.class_problem == "binary":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "multiclass":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "regression":
                metric = "neg_mean_squared_error"

            def objective(trial):
                param = {
                    "coef0": trial.suggest_loguniform("coef0", 1e-5, 1),
                    "degree": trial.suggest_int("degree", 2, 10),
                    "kernel": trial.suggest_categorical(
                        "kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"]
                    ),
                }
                pca = KernelPCA(
                    coef0=param["coef0"],
                    degree=param["degree"],
                    kernel=param["kernel"],
                    random_state=1000,
                )
                try:
                    train_comps = pca.fit_transform(X_train_sample)
                    test_comps = pca.transform(X_test)
                    new_cols = [f"PCA_{i}" for i in range(train_comps.shape[1])]
                    X_train_branch = pd.DataFrame(train_comps, columns=new_cols)
                    X_test_branch = pd.DataFrame(test_comps, columns=new_cols)
                except ValueError:
                    X_train_branch = X_train
                    X_test_branch = X_test

                if self.class_problem == "binary" or self.class_problem == "multiclass":
                    model = lgb.LGBMClassifier(random_state=42)
                else:
                    model = lgb.LGBMRegressor(random_state=42)

                try:
                    scores = cross_val_score(
                        model, X_train_branch, Y_train_sample, cv=10, scoring=metric
                    )
                    train_mae = np.mean(scores)
                    if self.class_problem in ["binary", "multiclass"]:
                        train_mae *= 100
                except Exception:
                    train_mae = 0

                print(train_mae)

                model.fit(X_train_branch, Y_train_sample)
                scores_2_test = model.predict(X_test_branch)

                try:
                    if self.class_problem in ["binary", "multiclass"]:
                        matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
                        test_mae = matthew_2 * 100
                    else:
                        test_mae = mean_squared_error(
                            Y_test, scores_2_test, squared=True
                        )
                        test_mae *= -1
                except Exception:
                    test_mae = 0

                print(test_mae)

                meissner_score = (train_mae + test_mae) / 2 - (
                    abs(train_mae - test_mae)
                ) ** 3
                return meissner_score

            algorithm = "final_kernel_pca_dimensionality_reduction"

            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=42, consider_endpoints=True
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
            )
            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds[algorithm],
                timeout=self.hyperparameter_tuning_max_runtime_secs[algorithm],
                gc_after_trial=True,
                show_progress_bar=True,
            )
            self.optuna_studies[f"{algorithm}"] = {}
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                fig.show()
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            best_parameters = study.best_trial.params
            pca = KernelPCA(
                coef0=best_parameters["coef0"],
                degree=best_parameters["degree"],
                kernel=best_parameters["kernel"],
                random_state=1000,
            )
            train_comps = pca.fit_transform(X_train)
            test_comps = pca.transform(X_test)
            new_cols = [f"PCA_{i}" for i in range(train_comps.shape[1])]
            X_train = pd.DataFrame(train_comps, columns=new_cols)
            X_test = pd.DataFrame(test_comps, columns=new_cols)
            self.preprocess_decisions[
                "final_kernel_pca_dimensionality_reduction_reduction_parameters"
            ] = best_parameters
            self.preprocess_decisions[
                "final_kernel_pca_dimensionality_reduction_reduction_model"
            ] = pca

            print(
                f"Number of columns after dimensionality reduction is: {len(X_train.columns.to_list())}"
            )

            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            try:
                del pca
                del train_comps
                del test_comps
                del study
                del sampler
            except Exception:
                pass

        logging.info("Finished final PCA dimensionality reduction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def final_pca_dimensionality_reduction(self):
        logging.info("Start final PCA dimensionality reduction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            best_parameters = self.preprocess_decisions[
                "final_pca_dimensionality_reduction_parameters"
            ]
            pca = self.preprocess_decisions["final_pca_dimensionality_reduction_model"]
            dataframe_comps = pca.transform(self.dataframe)
            new_cols = [f"PCA_{i}" for i in range(best_parameters["n_components"])]
            self.dataframe = pd.DataFrame(dataframe_comps, columns=new_cols)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if self.final_pca_dimensionality_reduction_sample_size > len(X_train.index):
                sample_size = len(X_train.index)
            else:
                sample_size = self.final_pca_dimensionality_reduction_sample_size

            max_pca_columns = len(X_train.columns.to_list()) - 1

            print(
                f"Number of columns before dimensionality reduction is: {len(X_train.columns.to_list())}"
            )

            X_train_sample = X_train.copy()
            X_train_sample[self.target_variable] = Y_train
            X_train_sample = X_train_sample.sample(sample_size, random_state=42)
            Y_train_sample = X_train_sample[self.target_variable]
            X_train_sample = X_train_sample.drop(self.target_variable, axis=1)

            if self.class_problem == "binary":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "multiclass":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "regression":
                metric = "neg_mean_squared_error"

            def objective(trial):
                param = {
                    "n_components": trial.suggest_int(
                        "n_components", 2, max_pca_columns
                    ),
                    "whiten": trial.suggest_categorical("whiten", [True, False]),
                }
                pca = PCA(
                    n_components=param["n_components"],
                    whiten=param["whiten"],
                    random_state=1000,
                )
                train_comps = pca.fit_transform(X_train_sample)
                test_comps = pca.transform(X_test)
                new_cols = [f"PCA_{i}" for i in range(param["n_components"])]
                X_train_branch = pd.DataFrame(train_comps, columns=new_cols)
                X_test_branch = pd.DataFrame(test_comps, columns=new_cols)

                if self.class_problem == "binary" or self.class_problem == "multiclass":
                    model = lgb.LGBMClassifier(random_state=42)
                else:
                    model = lgb.LGBMRegressor(random_state=42)

                try:
                    scores = cross_val_score(
                        model, X_train_branch, Y_train_sample, cv=10, scoring=metric
                    )
                    train_mae = np.mean(scores)
                    if self.class_problem in ["binary", "multiclass"]:
                        train_mae *= 100
                except Exception:
                    train_mae = 0

                print(train_mae)

                model.fit(X_train_branch, Y_train_sample)
                scores_2_test = model.predict(X_test_branch)

                try:
                    if self.class_problem in ["binary", "multiclass"]:
                        matthew_2 = matthews_corrcoef(Y_test, scores_2_test)
                        test_mae = matthew_2 * 100
                    else:
                        test_mae = mean_squared_error(
                            Y_test, scores_2_test, squared=True
                        )
                        test_mae *= -1
                except Exception:
                    test_mae = 0

                print(test_mae)

                meissner_score = (train_mae + test_mae) / 2 - (
                    abs(train_mae - test_mae)
                ) ** 3
                return meissner_score

            algorithm = "final_pca_dimensionality_reduction"

            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=42, consider_endpoints=True
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
            )
            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds[algorithm],
                timeout=self.hyperparameter_tuning_max_runtime_secs[algorithm],
                gc_after_trial=True,
                show_progress_bar=True,
            )
            self.optuna_studies[f"{algorithm}"] = {}
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                fig.show()
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            best_parameters = study.best_trial.params
            pca = PCA(
                n_components=best_parameters["n_components"],
                whiten=best_parameters["whiten"],
                random_state=1000,
            )
            train_comps = pca.fit_transform(X_train)
            test_comps = pca.transform(X_test)
            new_cols = [f"PCA_{i}" for i in range(best_parameters["n_components"])]
            X_train = pd.DataFrame(train_comps, columns=new_cols)
            X_test = pd.DataFrame(test_comps, columns=new_cols)
            self.preprocess_decisions[
                "final_pca_dimensionality_reduction_parameters"
            ] = best_parameters
            self.preprocess_decisions["final_pca_dimensionality_reduction_model"] = pca

            print(
                f"Number of columns after dimensionality reduction is: {len(X_train.columns.to_list())}"
            )

            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
            try:
                del pca
                del train_comps
                del test_comps
                del study
                del sampler
            except Exception:
                pass

        logging.info("Finished final PCA dimensionality reduction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

    def shap_based_feature_selection(self):
        """
        Takes the training data and trains an LGBM model on it. Calculates SHAP values on 10-fold CV and eliminates
        features, which have a very high SHAP standard deviation or very low SHAP values overall.
        :return: Updates class attribute
        """
        self.get_current_timestamp("SHAP based feature selection")
        logging.info("Started SHAP based feature selection.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            final_features = self.selected_shap_feats
            self.dataframe = self.dataframe[final_features].copy()
            logging.info("Finished SHAP based feature selection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            print(
                f"Number of columns before SHAP feature selection is: {len(X_train.columns.to_list())}"
            )
            original_features = X_train.columns.to_list()

            def global_shap_importance(model, X):
                """Return a dataframe containing the features sorted by Shap importance
                Parameters
                ----------
                model : The tree-based model
                X : pd.Dataframe
                     training set/test set/the whole dataset ... (without the label)
                Returns
                -------
                pd.Dataframe
                    A dataframe containing the features sorted by Shap importance
                """
                explainer = shap.Explainer(model)
                shap_values = explainer(X)
                cohorts = {"": shap_values}
                cohort_exps = list(cohorts.values())
                for i in range(len(cohort_exps)):
                    if len(cohort_exps[i].shape) == 2:
                        cohort_exps[i] = cohort_exps[i].abs.mean(0)
                feature_names = cohort_exps[0].feature_names
                values = np.array(
                    [cohort_exps[i].values for i in range(len(cohort_exps))]
                )
                feature_importance = pd.DataFrame(
                    list(zip(feature_names, sum(values))),
                    columns=["features", "importance"],
                )

                return feature_importance

            def get_final_features(quantile=0.05, outlier_factor=2.0):
                all_shap_values = []
                # start training, prediction & Shap loop
                if self.class_problem in ["binary", "multiclass"]:
                    least_rep_class = Y_test.value_counts().min()
                    if least_rep_class < 20:
                        n_folds = least_rep_class
                    else:
                        n_folds = 20
                else:
                    n_folds = 20

                if self.class_problem in ["binary", "multiclass"]:
                    skf = StratifiedKFold(
                        n_splits=n_folds, random_state=42, shuffle=True
                    )
                else:
                    skf = KFold(n_splits=n_folds, random_state=42, shuffle=True)

                for train_index, test_index in skf.split(X_train, Y_train):
                    x_train, x_test = (
                        X_train.iloc[train_index],
                        X_train.iloc[test_index],
                    )
                    y_train = Y_train.iloc[train_index]

                    model = lgb.LGBMClassifier(random_state=42)
                    model.fit(x_train, y_train)

                    all_shap_values.append(global_shap_importance(model, x_test))

                shaps_all_its = pd.concat(all_shap_values)

                def column_sum(lst):
                    arr = np.array(lst)
                    res = [np.sum(abs(i)) for i in arr]
                    return res

                shaps_all_its = shaps_all_its.assign(
                    Product=lambda x: column_sum(x["importance"])
                )

                shaps_all_its_all_folds = shaps_all_its.groupby("features").sum()
                # get standard deviation and sums
                shap_stds = shaps_all_its_all_folds["Product"].std()

                # get 5th percentile thresholds
                shap_5th = shaps_all_its_all_folds["Product"].quantile(quantile)

                # filter features for each category
                shap_stds_cols = shaps_all_its_all_folds[
                    (
                        shaps_all_its_all_folds["Product"]
                        > (
                            shaps_all_its_all_folds["Product"].mean()
                            - outlier_factor * shap_stds
                        )
                    )
                ].index.to_list()
                shap_sums_cols = shaps_all_its_all_folds[
                    (shaps_all_its_all_folds["Product"] > shap_5th)
                ].index.to_list()

                # final feature list
                final_features = set(shap_stds_cols).intersection(shap_sums_cols)
                return final_features

            final_features = get_final_features()

            self.selected_shap_feats = final_features
            features_dropped = list(set(original_features) - set(final_features))
            print("The following features have been dropped....:")
            for col in features_dropped:
                print(col)
            X_train = X_train[final_features].copy()
            X_test = X_test[final_features].copy()
            print(
                f"Number of columns after SHAP feature selection is: {len(X_train.columns.to_list())}"
            )

            logging.info("Finished SHAP based feature selection.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def random_trees_embedding(self):
        """
        Creates random forest tree embeddings adn appends them as features on original dataframe.
        :return:
        """
        self.get_current_timestamp("Random trees embedding")
        logging.info("Started Random trees embedding.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            random_trees = self.preprocess_decisions["random_tres_embedder"]
            X_sparse_embedding = random_trees.transform(self.dataframe.toarray())
            new_cols = [
                f"Forest_embedding_{i}" for i in range(X_sparse_embedding.shape[1])
            ]
            pred_df = pd.DataFrame(X_sparse_embedding, columns=new_cols)
            self.dataframe = self.dataframe.merge(
                pred_df, left_index=True, right_index=True
            )

            logging.info("Finished Random trees embedding.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            random_trees = RandomTreesEmbedding(
                n_estimators=100, random_state=5, max_depth=1, n_jobs=-1
            ).fit(X_train)
            X_sparse_embedding = random_trees.transform(X_train)
            new_cols = [
                f"Forest_embedding_{i}" for i in range(X_sparse_embedding.shape[1])
            ]
            X_train_df = pd.DataFrame(X_sparse_embedding.toarray(), columns=new_cols)
            X_train = X_train.merge(X_train_df, left_index=True, right_index=True)

            X_sparse_embedding = random_trees.transform(X_test)
            X_test_df = pd.DataFrame(X_sparse_embedding.toarray(), columns=new_cols)
            X_test = X_test.merge(X_test_df, left_index=True, right_index=True)

            self.preprocess_decisions["random_tres_embedder"] = random_trees
            logging.info("Finished Random trees embedding.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def automated_feature_transformation(self):
        """
        Tries different feature space transformation algorithms and transforms the dataset.
        :return: Updates class attribute.
        """
        self.get_current_timestamp("Automated feature transformation")
        logging.info("Started automated feature transformation.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")

        if self.prediction_mode:
            best_parameters = self.preprocess_decisions["scaler_param"]
            scaler = get_scaler(best_parameters)
            if best_parameters["transformer"] == "no_scaling":
                pass
            else:
                columns = self.dataframe.columns
                self.dataframe = pd.DataFrame(
                    scaler.fit_transform(self.dataframe), columns=columns
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            columns = X_train.columns
            if self.class_problem == "binary":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "multiclass":
                metric = make_scorer(matthews_corrcoef)
            elif self.class_problem == "regression":
                metric = "neg_mean_squared_error"

            def objective(trial):
                param = {
                    "transformer": trial.suggest_categorical(
                        "transformer",
                        [
                            "quantile",
                            "maxabs",
                            "robust",
                            "minmax",
                            "yeo-johnson",
                            # "box_cox",
                            "l1",
                            "l2",
                            "no_scaling",
                        ],
                    ),
                    "n_quantiles": trial.suggest_uniform("n_quantiles", 10, 1000),
                }
                scaler = get_scaler(param)

                if param["transformer"] == "no_scaling":
                    X_train_scaled = X_train

                else:
                    X_train_scaled = pd.DataFrame(
                        scaler.fit_transform(X_train), columns=columns
                    )

                if self.class_problem == "binary" or self.class_problem == "multiclass":
                    model = lgb.LGBMClassifier(random_state=42)
                else:
                    model = lgb.LGBMRegressor(random_state=42)

                try:
                    scores = cross_val_score(
                        model, X_train_scaled, Y_train, cv=10, scoring=metric
                    )
                    train_mae = np.mean(scores)
                    if self.class_problem in ["binary", "multiclass"]:
                        train_mae *= 100
                except Exception:
                    train_mae = 0

                return train_mae

            algorithm = "automated_feature_transformation"

            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=42, consider_endpoints=True
            )
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
            )
            study.optimize(
                objective,
                n_trials=20,
                timeout=1 * 60 * 60,
                gc_after_trial=True,
                show_progress_bar=True,
            )
            self.optuna_studies[f"{algorithm}"] = {}
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                fig.show()
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            best_parameters = study.best_trial.params
            scaler = get_scaler(best_parameters)
            if best_parameters["transformer"] == "no_scaling":
                pass
            else:
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
                X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=columns)
                self.data_scaled = True
            self.preprocess_decisions["scaler_param"] = best_parameters
            logging.info("Finished automated feature transformation.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
