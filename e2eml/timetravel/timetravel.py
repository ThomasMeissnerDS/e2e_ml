import copy
import gc
import logging
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)

from e2eml.full_processing import postprocessing

pd.set_option("display.max_colwidth", None)


class TimeTravel:
    def call_preprocessing_functions_mapping(self, class_instance):
        """
        A static function, that stores preprocessing step names, function objects and default arguments in a dictionbary.
        The order of items determines the order of preprocessing steps. We do not guarantee 100% success rate, if
        the order gets exchanged.

        :param class_instance:
        :return: Adds a class attribute.
        """
        class_instance.preprocessing_funcs = {
            "automatic_type_detection_casting": {
                "func": class_instance.automatic_type_detection_casting,
                "args": None,
            },
            "remove_duplicate_column_names": {
                "func": class_instance.remove_duplicate_column_names,
                "args": None,
            },
            "reset_dataframe_index": {
                "func": class_instance.reset_dataframe_index,
                "args": None,
            },
            "fill_infinite_values": {
                "func": class_instance.fill_infinite_values,
                "args": None,
            },
            "early_numeric_only_feature_selection": {
                "func": class_instance.automated_feature_selection,
                "args": (None, None, True),
            },
            "delete_high_null_cols": {
                "func": class_instance.delete_high_null_cols,
                "args": (0.05),
            },
            "data_binning": {"func": class_instance.data_binning, "args": None},
            "regex_clean_text_data": {
                "func": class_instance.regex_clean_text_data,
                "args": None,
            },
            "handle_target_skewness": {
                "func": class_instance.target_skewness_handling,
                "args": ("fit"),
            },
            "datetime_converter": {
                "func": class_instance.datetime_converter,
                "args": ("all"),
            },
            "pos_tagging_pca": {
                "func": class_instance.pos_tagging_pca,
                "args": (True),
            },  # slow with many categories
            "append_text_sentiment_score": {
                "func": class_instance.append_text_sentiment_score,
                "args": None,
            },
            "tfidf_vectorizer_to_pca": {
                "func": class_instance.tfidf_vectorizer_to_pca,
                "args": (True),
            },  # slow with many categories
            "tfidf_vectorizer": {
                "func": class_instance.tfidf_vectorizer_to_pca,
                "args": (False),
            },
            "rare_feature_processing": {
                "func": class_instance.rare_feature_processor,
                "args": (0.005, "miscellaneous", class_instance.rarity_cols),
            },
            "cardinality_remover": {
                "func": class_instance.cardinality_remover,
                "args": (100),
            },
            "holistic_null_filling": {
                "func": class_instance.holistic_null_filling,
                "args": (False),
            },  # slow
            "numeric_binarizer_pca": {
                "func": class_instance.numeric_binarizer_pca,
                "args": None,
            },
            "onehot_pca": {"func": class_instance.onehot_pca, "args": None},
            "category_encoding": {
                "func": class_instance.category_encoding,
                "args": ("target"),
            },
            "fill_nulls_static": {
                "func": class_instance.fill_nulls,
                "args": ("static"),
            },
            "outlier_care": {
                "func": class_instance.outlier_care,
                "args": ("isolation", "append"),
            },
            "delete_outliers": {
                "func": class_instance.outlier_care,
                "args": ("isolation", "delete", -0.5),
            },
            "remove_collinearity": {
                "func": class_instance.remove_collinearity,
                "args": (0.8),
            },
            "skewness_removal": {
                "func": class_instance.skewness_removal,
                "args": (False),
            },
            "automated_feature_transformation": {
                "func": class_instance.automated_feature_transformation,
                "args": None,
            },
            "clustering_as_a_feature_dbscan": {
                "func": class_instance.dbscan_clustering,
                "args": None,
            },
            "clustering_as_a_feature_kmeans_loop": {
                "func": class_instance.kmeans_clustering_loop,
                "args": None,
            },
            "clustering_as_a_feature_gaussian_mixture_loop": {
                "func": class_instance.gaussian_mixture_clustering_loop,
                "args": None,
            },
            "pca_clustering_results": {
                "func": class_instance.pca_clustering_results,
                "args": None,
            },
            "autotuned_clustering": {
                "func": class_instance.auto_tuned_clustering,
                "args": None,
            },
            "svm_outlier_detection_loop": {
                "func": class_instance.svm_outlier_detection_loop,
                "args": None,
            },
            "reduce_memory_footprint": {
                "func": class_instance.reduce_memory_footprint,
                "args": None,
            },
            "scale_data": {"func": class_instance.data_scaling, "args": None},
            "smote": {"func": class_instance.smote_binary_multiclass, "args": None},
            "automated_feature_selection": {
                "func": class_instance.automated_feature_selection,
                "args": (None, None, False),
            },
            "bruteforce_random_feature_selection": {
                "func": class_instance.bruteforce_random_feature_selection,
                "args": None,
            },
            "synthetic_data_augmentation": {
                "func": class_instance.synthetic_data_augmentation,
                "args": None,
            },
            "final_pca_dimensionality_reduction": {
                "func": class_instance.final_pca_dimensionality_reduction,
                "args": None,
            },
            "final_kernel_pca_dimensionality_reduction": {
                "func": class_instance.final_kernel_pca_dimensionality_reduction,
                "args": None,
            },
            "delete_low_variance_features": {
                "func": class_instance.delete_low_variance_features,
                "args": None,
            },
            "shap_based_feature_selection": {
                "func": class_instance.shap_based_feature_selection,
                "args": None,
            },
            "autoencoder_based_oversampling": {
                "func": class_instance.autoencoder_based_oversampling,
                "args": None,
            },
            "random_trees_embedding": {
                "func": class_instance.random_trees_embedding,
                "args": None,
            },  # slow
            "delete_unpredictable_training_rows": {
                "func": class_instance.delete_unpredictable_training_rows,
                "args": None,
            },
            "sort_columns_alphabetically": {
                "func": class_instance.sort_columns_alphabetically,
                "args": None,
            },
        }

    def call_classification_algorithm_mapping(self, class_instance):
        """
        A static function, that stores ml algorithms and their function objects..
        The order of items does not have any impact.

        :param class_instance: e2eml ClassificationBlueprint or RegressionBlueprint class instance.
        :return: Adds a class attribute.
        """
        class_instance.classification_algorithms_functions = {
            "logistic_regression": class_instance.ml_bp00_train_test_binary_full_processing_log_reg_prob,
            "ridge": class_instance.ml_bp08_multiclass_full_processing_ridge,
            "catboost": class_instance.ml_bp09_multiclass_full_processing_catboost,
            "xgboost": class_instance.ml_bp01_multiclass_full_processing_xgb_prob,
            "ngboost": class_instance.ml_bp04_multiclass_full_processing_ngboost,
            "lgbm": class_instance.ml_bp02_multiclass_full_processing_lgbm_prob,
            "lgbm_focal": class_instance.ml_bp14_multiclass_full_processing_lgbm_focal,
            "tabnet": class_instance.ml_bp07_multiclass_full_processing_tabnet,
            "vowpal_wabbit": class_instance.ml_bp05_multiclass_full_processing_vowpal_wabbit,
            "sklearn_ensemble": class_instance.ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble,
            "sgd": class_instance.ml_bp10_multiclass_full_processing_sgd,
            "quadratic_discriminant_analysis": class_instance.ml_bp11_multiclass_full_processing_quadratic_discriminant_analysis,
            "svm": class_instance.ml_bp12_multiclass_full_processing_svm,
            "multinomial_nb": class_instance.ml_bp13_multiclass_full_processing_multinomial_nb,
            "deesc": class_instance.ml_bp15_multiclass_full_processing_deesc,
        }

    def call_regression_algorithm_mapping(self, class_instance):
        class_instance.regression_algorithms_functions = {
            "linear_regression": class_instance.ml_bp10_train_test_regression_full_processing_linear_reg,
            "elasticnet": class_instance.ml_bp19_regression_full_processing_elasticnet_reg,
            "ridge": class_instance.ml_bp18_regression_full_processing_ridge_reg,
            "catboost": class_instance.ml_bp20_regression_full_processing_catboost,
            "xgboost": class_instance.ml_bp11_regression_full_processing_xgboost,
            "ngboost": class_instance.ml_bp14_regressions_full_processing_ngboost,
            "lgbm": class_instance.ml_bp12_regressions_full_processing_lgbm,
            "tabnet": class_instance.ml_bp17_regression_full_processing_tabnet_reg,
            "vowpal_wabbit": class_instance.ml_bp15_regression_full_processing_vowpal_wabbit_reg,
            "sklearn_ensemble": class_instance.ml_bp13_regression_full_processing_sklearn_stacking_ensemble,
            "sgd": class_instance.ml_bp20_regression_full_processing_sgd,
            "svm_regression": class_instance.ml_bp22_regression_full_processing_svm
            # "ransac": class_instance.ml_bp21_regression_full_processing_ransac
        }

    def create_time_travel_checkpoints(
        self, class_instance, checkpoint_file_path=None, df=None, reload_instance=False
    ):
        """
        Runs a preprocessing blueprint only. Saves blueprints after certain checkpoints, which can be defined in

        :param class_instance: Accepts a an e2eml Classification or Regression class instance. This does not support
            NLP transformers.
        :param checkpoint_file_path: (Optional). Takes a file path to store the saved class instance checkpoints.
            On default will save in current location.
        :param df: Accepts a dataframe to make predictions on new data.
        :return: Saves the checkpoints locally.
        """
        if not reload_instance:
            logging.info("Start blueprint.")
            class_instance.runtime_warnings(warn_about="future_architecture_change")
            class_instance.check_prediction_mode(df)
            class_instance.train_test_split(how=class_instance.train_split_type)
            self.last_checkpoint_reached = "train_test_split"
        else:
            pass

        self.call_preprocessing_functions_mapping(class_instance=class_instance)
        if class_instance.class_problem in ["binary", "multiclass"]:
            self.call_classification_algorithm_mapping(class_instance=class_instance)
        else:
            self.call_regression_algorithm_mapping(class_instance=class_instance)

        for key in class_instance.blueprint_step_selection_non_nlp.keys():
            if class_instance.blueprint_step_selection_non_nlp[key] and (
                not class_instance.checkpoint_reached[key]
                or class_instance.prediction_mode
            ):
                if (
                    (
                        key == "regex_clean_text_data"
                        and len(class_instance.nlp_transformer_columns) > 0
                    )
                    or (
                        key == "tfidf_vectorizer"
                        and len(class_instance.nlp_transformer_columns) > 0
                    )
                    or (
                        key == "append_text_sentiment_score"
                        and len(class_instance.nlp_transformer_columns) > 0
                    )
                    or (
                        key
                        not in [
                            "regex_clean_text_data",
                            "tfidf_vectorizer",
                            "append_text_sentiment_score",
                            "train_test_split",
                        ]
                    )
                ):
                    if class_instance.preprocessing_funcs[key]["args"]:
                        try:
                            if (
                                len(
                                    np.array(
                                        class_instance.preprocessing_funcs[key]["args"]
                                    )
                                )
                                == 1
                            ):
                                class_instance.preprocessing_funcs[key]["func"](
                                    class_instance.preprocessing_funcs[key]["args"]
                                )
                            else:
                                class_instance.preprocessing_funcs[key]["func"](
                                    *class_instance.preprocessing_funcs[key]["args"]
                                )
                        except TypeError:
                            class_instance.preprocessing_funcs[key]["func"](
                                class_instance.preprocessing_funcs[key]["args"]
                            )
                    else:
                        class_instance.preprocessing_funcs[key]["func"]()
                else:
                    print(
                        f"Skipped preprocessing step {key} as it has not been selected by user."
                    )

                # save checkpoints, if not in predict
                if class_instance.prediction_mode:
                    pass
                else:
                    class_instance.checkpoint_reached[key] = True
                    self.last_checkpoint_reached = key
                    postprocessing.save_to_production(
                        class_instance,
                        file_name=f"blueprint_checkpoint_{key}",
                        clean=False,
                        file_path=checkpoint_file_path,
                    )
            else:
                pass

    def load_checkpoint(self, checkpoint_to_load=None, checkpoint_file_path=None):
        """
        This function loads saved checkpoints on demand. If no checkpoint is specified, it loads the most recent
        checkpoint executed in the pipeline.

        :param checkpoint_to_load: Takes a string, specifying the checkpoint to load. All strings can be looked up
            in blueprint_step_selection_non_nlp class attribute. Only checkpoints, which have been saved explicitely can be
            loaded. If no checkpoint is specified, it loads the most recent
            checkpoint executed in the pipeline.
        :param checkpoint_file_path: (Optional) Takes a string. On default loads checkpoint from current path. If specified,
            loads the checkpoint from this path.
        :return: Returns loaded checkpoint.
        """
        if not checkpoint_to_load:
            class_instance = postprocessing.load_for_production(
                file_name=f"blueprint_checkpoint_{self.last_checkpoint_reached}",
                file_path=checkpoint_file_path,
            )
        else:
            class_instance = postprocessing.load_for_production(
                file_name=f"blueprint_checkpoint_{checkpoint_to_load}",
                file_path=checkpoint_file_path,
            )
        return class_instance

    def timetravel_model_training(self, class_instance, algorithm="lgbm"):
        def overwrite_normal_preprocess_func(df=None):
            return "overwritten"

        # overwriting the preprocessing function as we have done this already
        class_instance.std_preprocessing_pipeline = overwrite_normal_preprocess_func

        if class_instance.class_problem in ["binary", "multiclass"]:
            class_instance.classification_algorithms_functions[algorithm]()
        else:
            class_instance.regression_algorithms_functions[algorithm]()

        logging.info("Finished blueprint.")


def timewalk_auto_exploration(  # noqa: C901
    class_instance,
    holdout_df,
    holdout_target,
    is_imbalanced=False,
    algs_to_test=None,
    preprocess_checkpoints=None,
    speed_up_model_tuning=True,
    name_of_exist_experiment=None,
    experiment_name="timewalk.pkl",
    experiment_comment=f'Experiment run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
):
    """
    Timewalk is an extension to TimeTravel. It executes different preprocessing steps an short model training to explore
    the best combination. It returns a Pandas DataFrame with all results. Timewalk is meant to explore and is not suitable
    for final training for various reasons. Timewalk will result in long runtimes.

    :param class_instance: Expects a freshly instantiated e2eml ClassificationBlueprint or RegressionBlueprint class instance.
    :param holdout_df: Expects a Pandas Dataframe, which will be only used for final evaluation. Is not alolowed to
        include the target as it simulates prediction on new data. It is recommended to use the most recent datapoints as holdout.
    :param holdout_target: Expects a Pandas Series with holdout targets.
    :param speed_up_model_tuning: If True, timewalk will run with reduced rounds of hyprparameter tuning. If False,
        timewalk will use hyperparameter tuning rounds and maximum runtime from the imported e2eml class. If False, users can
        control these parameter by adjusting them after e2eml class instantiation and before importing the class to timewalk.
    :param name_of_exist_experiment: Expects a string. Name of a locally saved file with results from a past experiment
        can be provided. In this case timewalk will load the file as a dataframe and concatenate old and new results into
        one dataframe.
    :param algs_to_test: (Optional). Expects a list object with algorithms to test. Will test on default:
        ["ridge", "xgboost", "lgbm", "tabnet", "ngboost", "vowpal_wabbit", "logistic_regression",
        "linear_regression", "elasticnet", "sgd", "quadratic_discriminant_analysis", "svm"]
    :param experiment_name: Expects string. Will determine the name of the exported results dataframe (as pickle file).
    :param experiment_comment: Expects a string. This will add a comment of choice to the results dataframe. On default
        it will add a string stating when the experiment has been started.
    :return: Pandas DataFrame with results.
    """
    class_instance = copy.copy(class_instance)

    # define algorithms to consider
    if isinstance(algs_to_test, list):
        algorithms = algs_to_test
    else:

        algorithms = [
            "ridge",
            "xgboost",
            "lgbm",
            "lgbm_focal",
            "tabnet",
            "ngboost",
            "vowpal_wabbit",
            "logistic_regression",
            "linear_regression",
            "elasticnet",
            "sgd",
            "quadratic_discriminant_analysis",
            "svm",
            "svm_regression",
        ]

        if len(class_instance.dataframe.index) > 10000:
            algorithms.remove("ngboost")
            algorithms.remove("xgboost")
        else:
            pass

    # removing algorithms, that are not suitable for classification or regression tasks
    if class_instance.class_problem in ["binary", "multiclass"]:
        try:
            algorithms.remove("linear_regression")
        except Exception:
            pass

        try:
            algorithms.remove("elasticnet")
        except Exception:
            pass

        try:
            algorithms.remove("ransac")
        except Exception:
            pass

        try:
            algorithms.remove("svm_regression")
        except Exception:
            pass

    else:
        try:
            algorithms.remove("logistic_regression")
        except Exception:
            pass

        try:
            algorithms.remove("quadratic_discriminant_analysis")
        except Exception:
            pass

        try:
            algorithms.remove("svm")
        except Exception:
            pass

        try:
            algorithms.remove("multinomial_nb")
        except Exception:
            pass

        try:
            algorithms.remove("lgbm_focal")
        except Exception:
            pass

    if speed_up_model_tuning:
        # we reduce the tuning rounds for all algorithms
        class_instance.hyperparameter_tuning_rounds["xgboost"] = 10
        class_instance.hyperparameter_tuning_rounds["lgbm"] = 10
        class_instance.hyperparameter_tuning_rounds["sgd"] = 100
        class_instance.hyperparameter_tuning_rounds["svm"] = 25
        class_instance.hyperparameter_tuning_rounds["svm_regression"] = 10
        class_instance.hyperparameter_tuning_rounds["multinomial_nb"] = 10
        class_instance.hyperparameter_tuning_rounds["tabnet"] = 10
        class_instance.hyperparameter_tuning_rounds["ngboost"] = 3
        class_instance.hyperparameter_tuning_rounds["sklearn_ensemble"] = 10
        class_instance.hyperparameter_tuning_rounds["ridge"] = 10
        class_instance.hyperparameter_tuning_rounds["elasticnet"] = 10
        class_instance.hyperparameter_tuning_rounds["catboost"] = 3
        class_instance.hyperparameter_tuning_rounds["bruteforce_random"] = 50
        class_instance.hyperparameter_tuning_rounds[
            "autoencoder_based_oversampling"
        ] = 10
        class_instance.hyperparameter_tuning_rounds[
            "final_pca_dimensionality_reduction"
        ] = 50

        # we also limit the time for hyperparameter tuning
        class_instance.hyperparameter_tuning_max_runtime_secs["xgboost"] = 4 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["sgd"] = 3 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["lgbm"] = 3 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["tabnet"] = 2 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["ngboost"] = 3 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["sklearn_ensemble"] = (
            3 * 60 * 60
        )
        class_instance.hyperparameter_tuning_max_runtime_secs["ridge"] = 3 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["elasticnet"] = (
            3 * 60 * 60
        )
        class_instance.hyperparameter_tuning_max_runtime_secs["catboost"] = 4 * 60 * 60
        class_instance.hyperparameter_tuning_max_runtime_secs["bruteforce_random"] = (
            3 * 60 * 60
        )
        class_instance.hyperparameter_tuning_max_runtime_secs[
            "autoencoder_based_oversampling"
        ] = (1 * 60 * 60)
        class_instance.hyperparameter_tuning_max_runtime_secs[
            "final_pca_dimensionality_reduction"
        ] = (1 * 60 * 60)
    else:
        pass

    # we adjust default preprocessing
    class_instance.blueprint_step_selection_non_nlp["autotuned_clustering"] = True
    class_instance.blueprint_step_selection_non_nlp["scale_data"] = True

    # we want to store our results
    scoring_results = []
    scoring_2_results = []
    scoring_3_results = []
    # scoring_4_results = []
    algorithms_used = []
    preprocessing_steps_used = []
    elapsed_times = []
    unique_indices = []

    # define checkpoints to load
    if isinstance(preprocess_checkpoints, list):
        checkpoints = preprocess_checkpoints
    else:
        checkpoints = [
            "default",
            "delete_low_variance_features",
            "automated_feature_selection",
            "autotuned_clustering",
            "cardinality_remover",
            "delete_high_null_cols",
            "early_numeric_only_feature_selection",
            "fill_infinite_values",
        ]

    if not is_imbalanced:
        if "shap_based_feature_selection" in checkpoints:
            checkpoints.remove("shap_based_feature_selection")

    # define the type of scoring
    if class_instance.class_problem in ["binary", "multiclass"]:
        metric = "Matthews"
        metric_2 = "Accuracy"
        metric_3 = "Recall"
        # ascending = False
    else:
        metric = "Mean absolute error"
        metric_2 = "R2 score"
        metric_3 = "RMSE"
        # ascending = True

    # creating checkpoints and training the model
    automl_travel = TimeTravel()

    unique_indices_counter = 0
    for checkpoint in checkpoints:
        preprocess_runtime = 0
        preprocess_start = time.time()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Start iteration for checkpoint {checkpoint} at {preprocess_start}.")
        try:
            if len(scoring_results) == 0:
                automl_travel.create_time_travel_checkpoints(
                    class_instance, reload_instance=False
                )
            else:
                class_instance = automl_travel.load_checkpoint(
                    checkpoint_to_load=checkpoint
                )
                if checkpoint == "delete_low_variance_features":
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = False
                elif checkpoint == "automated_feature_selection":
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = False
                elif checkpoint == "autotuned_clustering":
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "delete_unpredictable_training_rows"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = True
                elif checkpoint == "cardinality_remover":
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "svm_outlier_detection_loop"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "delete_outliers"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "automated_feature_transformation"
                    ] = True
                elif checkpoint == "delete_high_null_cols":
                    class_instance.blueprint_step_selection_non_nlp[
                        "autotuned_clustering"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "tfidf_vectorizer_to_pca"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "data_binning"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "svm_outlier_detection_loop"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "delete_outliers"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "automated_feature_transformation"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = False
                elif checkpoint == "early_numeric_only_feature_selection":
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "tfidf_vectorizer_to_pca"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "data_binning"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "delete_outliers"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "automated_feature_transformation"
                    ] = True
                elif checkpoint == "fill_infinite_values":
                    class_instance.blueprint_step_selection_non_nlp[
                        "tfidf_vectorizer_to_pca"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "numeric_binarizer_pca"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "data_binning"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "svm_outlier_detection_loop"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "autoencoder_based_oversampling"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "clustering_as_a_feature_dbscan"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "clustering_as_a_feature_kmeans_loop"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "clustering_as_a_feature_gaussian_mixture_loop"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "pca_clustering_results"
                    ] = False
                    class_instance.blueprint_step_selection_non_nlp[
                        "delete_outliers"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "automated_feature_transformation"
                    ] = True
                    class_instance.blueprint_step_selection_non_nlp[
                        "shap_based_feature_selection"
                    ] = True
            automl_travel.create_time_travel_checkpoints(
                class_instance, reload_instance=True
            )

        except Exception:
            print(traceback.format_exc())
        preprocess_end = time.time()
        preprocess_runtime = preprocess_end - preprocess_start
        print(f"All algorithms to be tested are: {algorithms}.")
        try:
            for alg in algorithms:
                unique_indices_counter += 1
                start = time.time()
                print(f"Start iteration for algorithm {alg} at {start}.")
                try:
                    class_instance = automl_travel.load_checkpoint(
                        checkpoint_to_load="sort_columns_alphabetically"
                    )  # gets latest checkpoint
                    print("Successfully loaded checkpoint last checkpoint.")
                    automl_travel.create_time_travel_checkpoints(
                        class_instance, reload_instance=True
                    )
                    automl_travel.timetravel_model_training(class_instance, alg)
                    automl_travel.create_time_travel_checkpoints(
                        class_instance, df=holdout_df
                    )
                    automl_travel.timetravel_model_training(class_instance, alg)
                    if class_instance.labels_encoded:
                        hold_df_target = class_instance.label_encoder_decoder(
                            holdout_target, mode="transform"
                        )
                    else:
                        hold_df_target = holdout_target
                except Exception:
                    print(traceback.format_exc())

                try:
                    if class_instance.class_problem in ["binary", "multiclass"]:
                        scoring_2 = accuracy_score(
                            hold_df_target, class_instance.predicted_classes[alg]
                        )
                        scoring_3 = recall_score(
                            hold_df_target,
                            class_instance.predicted_classes[alg],
                            average="weighted",
                        )
                        scoring = matthews_corrcoef(
                            hold_df_target, class_instance.predicted_classes[alg]
                        )
                        full_classification_report = classification_report(
                            hold_df_target, class_instance.predicted_classes[alg]
                        )
                        print(full_classification_report)
                    else:
                        scoring = mean_absolute_error(
                            hold_df_target, class_instance.predicted_values[alg]
                        )
                        scoring_2 = r2_score(
                            hold_df_target, class_instance.predicted_values[alg]
                        )
                        scoring_3 = mean_squared_error(
                            hold_df_target,
                            class_instance.predicted_values[alg],
                            squared=True,
                        )
                except Exception:
                    print(traceback.format_exc())
                    try:
                        if class_instance.class_problem in ["binary", "multiclass"]:
                            scoring = matthews_corrcoef(
                                pd.Series(hold_df_target).astype(bool),
                                class_instance.predicted_classes[alg],
                            )
                            scoring_2 = accuracy_score(
                                pd.Series(hold_df_target).astype(bool),
                                class_instance.predicted_classes[alg].astype(bool),
                            )
                            scoring_3 = recall_score(
                                pd.Series(hold_df_target).astype(bool),
                                class_instance.predicted_classes[alg].astype(bool),
                                average="weighted",
                            )
                            full_classification_report = classification_report(
                                pd.Series(hold_df_target).astype(bool),
                                class_instance.predicted_classes[alg],
                            )
                            print(full_classification_report)
                        else:
                            scoring = mean_absolute_error(
                                pd.Series(hold_df_target).astype(float),
                                class_instance.predicted_values[alg],
                            )
                            scoring_2 = r2_score(
                                pd.Series(hold_df_target).astype(float),
                                class_instance.predicted_values[alg],
                            )
                            scoring_3 = mean_squared_error(
                                pd.Series(hold_df_target).astype(float),
                                class_instance.predicted_values[alg],
                                squared=True,
                            )
                    except Exception:
                        print(traceback.format_exc())
                        print("Evaluation on holdout failed.")
                        if class_instance.class_problem in ["binary", "multiclass"]:
                            scoring = 0
                            scoring_2 = 0
                            scoring_3 = 0
                        else:
                            scoring = 999999999
                            scoring_2 = -1
                            scoring_3 = 99999999

                print(f"Score achieved on holdout dataset is: {scoring}.")

                end = time.time()
                elapsed_time = end - start
                elapsed_times.append(elapsed_time)
                scoring_results.append(scoring)
                scoring_2_results.append(scoring_2)
                scoring_3_results.append(scoring_3)
                algorithms_used.append(alg)
                unique_indices.append(unique_indices_counter)
                preprocessing_steps_used.append(
                    class_instance.blueprint_step_selection_non_nlp
                )
                results_dict = {
                    "Trial number": unique_indices,
                    "Algorithm": algorithms_used,
                    metric: scoring_results,
                    metric_2: scoring_2_results,
                    metric_3: scoring_3_results,
                    "Preprocessing applied": preprocessing_steps_used,
                    "ML model runtime in seconds": elapsed_times,
                }

                results_df = pd.DataFrame(results_dict)
                results_df["experiment_comment"] = experiment_comment
                results_df["preprocessing_runtime_sec"] = preprocess_runtime
                results_df["Total runtime"] = (
                    results_df["ML model runtime in seconds"]
                    + results_df["preprocessing_runtime_sec"]
                )
                if isinstance(name_of_exist_experiment, str):
                    try:
                        print("Try to load former experiment data.")
                        loaded_experiment = pd.read_pickle(name_of_exist_experiment)
                        results_df = pd.concat([results_df, loaded_experiment])
                    except Exception:
                        print(traceback.format_exc())
                        print(
                            "Loading former experiment data failed. Will export running experiment only."
                        )
                else:
                    pass
                results_df.to_pickle(experiment_name)
                print(f"End iteration for algorithm {alg} at {end}.")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                try:
                    del class_instance.trained_models[alg]
                    del class_instance
                    _ = gc.collect()
                except KeyError:
                    pass
        except Exception:
            print(traceback.format_exc())
            if class_instance.class_problem in ["binary", "multiclass"]:  # noqa: F821
                scoring = 0
                scoring_2 = 0
                scoring_3 = 0
            else:
                scoring = 999999999
                scoring_2 = -1
                scoring_3 = 99999999
            end = time.time()
            elapsed_time = end - start
            elapsed_times.append(elapsed_time)
            scoring_results.append(scoring)
            scoring_2_results.append(scoring_2)
            scoring_3_results.append(scoring_3)
            algorithms_used.append(alg)
            unique_indices.append(unique_indices_counter)
            preprocessing_steps_used.append(  # noqa: F821
                class_instance.blueprint_step_selection_non_nlp  # noqa: F821
            )  # noqa: F821
            results_dict = {
                "Trial number": unique_indices,
                "Algorithm": algorithms_used,
                metric: scoring_results,
                metric_2: scoring_2_results,
                metric_3: scoring_3_results,
                "Preprocessing applied": preprocessing_steps_used,
                "ML model runtime in seconds": elapsed_times,
            }

            results_df = pd.DataFrame(results_dict)
            results_df["experiment_comment"] = experiment_comment
            results_df["preprocessing_runtime_sec"] = preprocess_runtime
            results_df["Total runtime"] = (
                results_df["ML model runtime in seconds"]
                + results_df["preprocessing_runtime_sec"]
            )
            if isinstance(name_of_exist_experiment, str):
                try:
                    print("Try to load former experiment data.")
                    loaded_experiment = pd.read_pickle(name_of_exist_experiment)
                    results_df = pd.concat([results_df, loaded_experiment])
                except Exception:
                    print(traceback.format_exc())
                    print(
                        "Loading former experiment data failed. Will export running experiment only."
                    )
            else:
                pass
            results_df.to_pickle(experiment_name)
            print(f"End iteration for algorithm {alg} at {end}.")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            try:
                del class_instance.trained_models[alg]  # noqa: F821
                del class_instance  # noqa: F821
                _ = gc.collect()
            except KeyError:
                pass

    results_dict = {
        "Trial number": unique_indices,
        "Algorithm": algorithms_used,
        metric: scoring_results,
        metric_2: scoring_2_results,
        metric_3: scoring_3_results,
        "Preprocessing applied": preprocessing_steps_used,
        "ML model runtime in seconds": elapsed_times,
    }

    results_df = pd.DataFrame(results_dict)
    results_df["experiment_comment"] = experiment_comment
    results_df["preprocessing_runtime_sec"] = preprocess_runtime
    results_df["Total runtime"] = (
        results_df["ML model runtime in seconds"]
        + results_df["preprocessing_runtime_sec"]
    )
    if isinstance(name_of_exist_experiment, str):
        try:
            print("Try to load former experiment data.")
            loaded_experiment = pd.read_pickle(name_of_exist_experiment)
            results_df = pd.concat([results_df, loaded_experiment])
        except Exception:
            print(traceback.format_exc())
            print(
                "Loading former experiment data failed. Will export running experiment only."
            )
    else:
        pass
    results_df.to_pickle(experiment_name)

    try:
        fig = px.line(
            results_df,
            x="Total runtime",
            y=metric,
            color="Algorithm",
            text="Trial number",
        )
        fig.update_traces(textposition="bottom right")
        fig.show()
    except Exception:
        print(traceback.format_exc())
    return results_df
