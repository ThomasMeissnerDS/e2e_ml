import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import optuna
import warnings
import logging
import dill as pickle
import gc
from e2eml.classification.classification_blueprints import ClassificationBluePrint
from e2eml.regression.regression_blueprints import RegressionBluePrint
from e2eml.full_processing import postprocessing
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
import gc
import time
import plotly.express as px


class TimeTravel():

    def call_preprocessing_functions_mapping(self, class_instance):
        """
        A static function, that stores preprocessing step names, function objects and default arguments in a dictionbary.
        The order of items determines the order of preprocessing steps. We do not guarantee 100% success rate, if
        the order gets exchanged.
        :param class_instance:
        :return: Adds a class attribute.
        """
        class_instance.preprocessing_funcs = {
            "automatic_type_detection_casting": {"func": class_instance.automatic_type_detection_casting, "args": None},
            "remove_duplicate_column_names": {"func": class_instance.remove_duplicate_column_names, "args": None},
            "reset_dataframe_index": {"func": class_instance.reset_dataframe_index, "args": None},
            "fill_infinite_values": {"func": class_instance.fill_infinite_values, "args": None},
            "early_numeric_only_feature_selection": {"func": class_instance.automated_feature_selection, "args": (None, None, True)},
            "delete_high_null_cols": {"func": class_instance.delete_high_null_cols, "args": (0.05)},
            "data_binning": {"func": class_instance.data_binning, "args": None},
            "regex_clean_text_data": {"func": class_instance.regex_clean_text_data, "args": None},
            "handle_target_skewness": {"func": class_instance.target_skewness_handling, "args": ("fit")},
            "datetime_converter": {"func": class_instance.datetime_converter, "args": ("all")},
            "pos_tagging_pca": {"func": class_instance.pos_tagging_pca, "args": (True)},# slow with many categories
            "append_text_sentiment_score": {"func": class_instance.append_text_sentiment_score, "args": None},
            "tfidf_vectorizer_to_pca": {"func": class_instance.tfidf_vectorizer_to_pca, "args": (True)}, # slow with many categories
            "tfidf_vectorizer": {"func": class_instance.tfidf_vectorizer_to_pca, "args": (False)},
            "rare_feature_processing": {"func": class_instance.rare_feature_processor, "args": (0.005, 'miscellaneous', class_instance.rarity_cols)},
            "cardinality_remover": {"func": class_instance.cardinality_remover, "args": (100)},
            "holistic_null_filling": {"func": class_instance.holistic_null_filling, "args": (False)}, # slow
            "numeric_binarizer_pca": {"func": class_instance.numeric_binarizer_pca, "args": None},
            "onehot_pca": {"func": class_instance.onehot_pca, "args": None},
            "category_encoding": {"func": class_instance.category_encoding, "args": ("target")},
            "fill_nulls_static": {"func": class_instance.fill_nulls, "args": ("static")},
            "outlier_care": {"func": class_instance.outlier_care, "args": ('isolation', 'append')},
            "remove_collinearity": {"func": class_instance.remove_collinearity, "args": (0.8)},
            "skewness_removal": {"func": class_instance.skewness_removal, "args": (False)},
            "clustering_as_a_feature_dbscan": {"func": class_instance.dbscan_clustering, "args": None},
            "clustering_as_a_feature_kmeans_loop": {"func": class_instance.kmeans_clustering_loop, "args": None},
            "clustering_as_a_feature_gaussian_mixture_loop": {"func": class_instance.gaussian_mixture_clustering_loop, "args": None},
            "pca_clustering_results": {"func": class_instance.pca_clustering_results, "args": None},
            "autotuned_clustering": {"func": class_instance.auto_tuned_clustering, "args": None},
            "reduce_memory_footprint": {"func": class_instance.reduce_memory_footprint, "args": None},
            "scale_data": {"func": class_instance.data_scaling, "args": None},
            "smote": {"func": class_instance.smote_binary_multiclass, "args": None},
            "automated_feature_selection": {"func": class_instance.automated_feature_selection, "args": (None, None, False)},
            "bruteforce_random_feature_selection": {"func": class_instance.bruteforce_random_feature_selection, "args": None}, # slow
            "delete_unpredictable_training_rows": {"func": class_instance.delete_unpredictable_training_rows, "args": None},
            "autoencoder_based_oversampling": {"func": class_instance.autoencoder_based_oversampling, "args": None},
            "synthetic_data_augmentation": {"func": class_instance.synthetic_data_augmentation, "args": None},
            "final_pca_dimensionality_reduction": {"func": class_instance.final_pca_dimensionality_reduction, "args": None},
            "final_kernel_pca_dimensionality_reduction": {"func": class_instance.final_kernel_pca_dimensionality_reduction, "args": None},
            "sort_columns_alphabetically": {"func": class_instance.sort_columns_alphabetically, "args": None},
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
                        "ridge":  class_instance.ml_bp08_multiclass_full_processing_ridge,
                        "catboost": class_instance.ml_bp09_multiclass_full_processing_catboost,
                        "xgboost":  class_instance.ml_bp01_multiclass_full_processing_xgb_prob,
                        "ngboost": class_instance.ml_bp04_multiclass_full_processing_ngboost,
                        "lgbm": class_instance.ml_bp02_multiclass_full_processing_lgbm_prob,
                        "tabnet": class_instance.ml_bp07_multiclass_full_processing_tabnet,
                        "vowpal_wabbit": class_instance.ml_bp05_multiclass_full_processing_vowpal_wabbit,
                        "sklearn_ensemble": class_instance.ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble,
                        "sgd": class_instance.ml_bp10_multiclass_full_processing_sgd,
                        }

    def call_regression_algorithm_mapping(self, class_instance):
        class_instance.regression_algorithms_functions = {
                        "linear_regression": class_instance.ml_bp10_train_test_regression_full_processing_linear_reg,
                        "elasticnet": class_instance.ml_bp19_regression_full_processing_elasticnet_reg,
                        "ridge": class_instance.ml_bp18_regression_full_processing_ridge_reg,
                        "catboost":  class_instance.ml_bp20_regression_full_processing_catboost,
                        "xgboost": class_instance.ml_bp11_regression_full_processing_xgboost,
                        "ngboost": class_instance.ml_bp14_regressions_full_processing_ngboost,
                        "lgbm": class_instance.ml_bp12_regressions_full_processing_lgbm,
                        "tabnet": class_instance.ml_bp17_regression_full_processing_tabnet_reg,
                        "vowpal_wabbit": class_instance.ml_bp15_regression_full_processing_vowpal_wabbit_reg,
                        "sklearn_ensemble": class_instance.ml_bp13_regression_full_processing_sklearn_stacking_ensemble,
                        "sgd": class_instance.ml_bp20_regression_full_processing_sgd
                        }

    def create_time_travel_checkpoints(self, class_instance, checkpoint_file_path=None, df=None, reload_instance=False):
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
            logging.info('Start blueprint.')
            class_instance.runtime_warnings(warn_about="future_architecture_change")
            class_instance.check_prediction_mode(df)
            class_instance.train_test_split(how=class_instance.train_split_type)
            self.last_checkpoint_reached = "train_test_split"
        else:
            pass

        self.call_preprocessing_functions_mapping(class_instance=class_instance)
        if class_instance.class_problem in ["binary", 'multiclass']:
            self.call_classification_algorithm_mapping(class_instance=class_instance)
        else:
            self.call_regression_algorithm_mapping(class_instance=class_instance)

        for key, value in class_instance.blueprint_step_selection_non_nlp.items():
            if class_instance.blueprint_step_selection_non_nlp[key] and (not class_instance.checkpoint_reached[key] or class_instance.prediction_mode):
                if (key == "regex_clean_text_data" and len(class_instance.nlp_transformer_columns) > 0) or \
                        (key == "tfidf_vectorizer" and len(class_instance.nlp_transformer_columns) > 0) or \
                        (key == "append_text_sentiment_score" and len(class_instance.nlp_transformer_columns) > 0) or \
                        (key not in ["regex_clean_text_data", "tfidf_vectorizer", "append_text_sentiment_score",
                                     "train_test_split"]):
                    if class_instance.preprocessing_funcs[key]["args"]:
                        try:
                            if len(np.array(class_instance.preprocessing_funcs[key]["args"])) == 1:
                                class_instance.preprocessing_funcs[key]["func"](class_instance.preprocessing_funcs[key]["args"])
                            else:
                                class_instance.preprocessing_funcs[key]["func"](*class_instance.preprocessing_funcs[key]["args"])
                        except TypeError:
                            class_instance.preprocessing_funcs[key]["func"](class_instance.preprocessing_funcs[key]["args"])
                    else:
                        class_instance.preprocessing_funcs[key]["func"]()
                else:
                    print(f"Skipped preprocessing step {key} as it has not been selected by user.")

                # save checkpoints, if not in predict
                if class_instance.prediction_mode:
                    pass
                else:
                    class_instance.checkpoint_reached[key] = True
                    self.last_checkpoint_reached = key
                    postprocessing.save_to_production(class_instance, file_name=f'blueprint_checkpoint_{key}', clean=False,
                                                  file_path=checkpoint_file_path)
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
            class_instance = postprocessing.load_for_production(file_name=f'blueprint_checkpoint_{self.last_checkpoint_reached}',
                                                                file_path=checkpoint_file_path)
        else:
            class_instance = postprocessing.load_for_production(file_name=f'blueprint_checkpoint_{checkpoint_to_load}',
                                                                file_path=checkpoint_file_path)
        return class_instance

    def timetravel_model_training(self, class_instance, algorithm='lgbm'):
        def overwrite_normal_preprocess_func(df=None):
            return "overwritten"
        # overwriting the preprocessing function as we have done this already
        class_instance.std_preprocessing_pipeline = overwrite_normal_preprocess_func

        if class_instance.class_problem in ["binary", 'multiclass']:
            class_instance.classification_algorithms_functions[algorithm]()
        else:
            class_instance.regression_algorithms_functions[algorithm]()

        logging.info('Finished blueprint.')


def timewalk_auto_exploration(class_instance, holdout_df, holdout_target, algs_to_test=None,
                              speed_up_model_tuning=True,
                              experiment_name="timewalk.pkl"):
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
    :param algs_to_test: (Optional). Expects a list object with algorithms to test. Will test on default:
    ["xgboost", "lgbm", "tabnet", "ridge", "ngboost", "sgd", "vowpal_wabbit"]
    :param experiment_name: Expects string. Will determine the name of the exported results dataframe (as pickle file).
    :return: Pandas DataFrame with results.
    """
    # define algorithms to consider
    if isinstance(algs_to_test, list):
        algorithms = algs_to_test
    else:
        algorithms = ["ridge", "xgboost", "lgbm", "tabnet", "ngboost", "sgd", "vowpal_wabbit", "logistic_regression",
                      "linear_regression", "elasticnet"]

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
    else:
        try:
            algorithms.remove("logistic_regression")
        except Exception:
            pass

    if speed_up_model_tuning:
        # we reduce the tuning rounds for all algorithms
        class_instance.hyperparameter_tuning_rounds = {"xgboost": 10,
                                                       "lgbm": 10,
                                                       "sgd": 10,
                                                       "tabnet": 10,
                                                       "ngboost": 3,
                                                       "sklearn_ensemble": 10,
                                                       "ridge": 10,
                                                       "elasticnet": 10,
                                                       "catboost": 10,
                                                       "bruteforce_random": 500,
                                                       "autoencoder_based_oversampling": 20,
                                                       "final_pca_dimensionality_reduction": 20}

        # we also limit the time for hyperparameter tuning
        class_instance.hyperparameter_tuning_max_runtime_secs = {"xgboost": 4*60*60,
                                                                 "sgd": 3*60*60,
                                                                 "lgbm": 3*60*60,
                                                                 "tabnet": 3*60*60,
                                                                 "ngboost": 3*60*60,
                                                                 "sklearn_ensemble": 3*60*60,
                                                                 "ridge": 3*60*60,
                                                                 "elasticnet": 3*60*60,
                                                                 "catboost": 4*60*60,
                                                                 "bruteforce_random": 3*60*60,
                                                                 "autoencoder_based_oversampling": 1*60*60,
                                                                 "final_pca_dimensionality_reduction": 1*60*60}
    else:
        pass

    # we adjust default preprocessing
    class_instance.blueprint_step_selection_non_nlp["autotuned_clustering"] = True
    class_instance.blueprint_step_selection_non_nlp["scale_data"] = True
    class_instance.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"] = False
    class_instance.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"] = True


    # we want to store our results
    scoring_results = []
    scoring_2_results = []
    scoring_3_results = []
    scoring_4_results = []
    algorithms_used = []
    preprocessing_steps_used = []
    elapsed_times = []
    unique_indices = []

    # define checkpoints to load
    checkpoints = ["scale_data", "autotuned_clustering", "early_numeric_only_feature_selection"]

    # define the type of scoring
    if class_instance.class_problem in ["binary", "multiclass"]:
        metric = "Matthews"
        metric_2 = "Accuracy"
        metric_3 = "Recall"
        ascending = False
    else:
        metric = "Mean absolute error"
        metric_2 = "R2 score"
        metric_3 = "RMSE"
        ascending = True

    # creating checkpoints and training the model
    automl_travel = TimeTravel()

    unique_indices_counter = 0
    for checkpoint in checkpoints:
        for alg in algorithms:
            start = time.time()
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Start iteration for algorithm {alg} at {start}.")
            try:
                unique_indices_counter += 1
                if len(scoring_results) == 0:
                    automl_travel.create_time_travel_checkpoints(class_instance, reload_instance=False)
                else:
                    class_instance = automl_travel.load_checkpoint(checkpoint_to_load=checkpoint)
                    if checkpoint == 'scale_data':
                        class_instance.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"] = True
                        class_instance.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"] = False
                    elif checkpoint == "autotuned_clustering":
                        class_instance.blueprint_step_selection_non_nlp["scale_data"] = False
                        class_instance.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"] = False
                        class_instance.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"] = False
                    elif checkpoint == "early_numeric_only_feature_selection":
                        class_instance.blueprint_step_selection_non_nlp["tfidf_vectorizer_to_pca"] = False
                        class_instance.blueprint_step_selection_non_nlp["data_binning"] = False
                        class_instance.blueprint_step_selection_non_nlp["scale_data"] = False
                        class_instance.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"] = False
                        class_instance.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"] = False
                    automl_travel.create_time_travel_checkpoints(class_instance, reload_instance=True)
                automl_travel.timetravel_model_training(class_instance, alg)
                automl_travel.create_time_travel_checkpoints(class_instance, df=holdout_df)
                automl_travel.timetravel_model_training(class_instance, alg)

                try:
                    if class_instance.class_problem in ["binary", "multiclass"]:
                        scoring_2 = accuracy_score(holdout_target, class_instance.predicted_classes[alg])
                        scoring_3 = recall_score(holdout_target, class_instance.predicted_classes[alg], average='weighted')
                        scoring = matthews_corrcoef(holdout_target, class_instance.predicted_classes[alg])
                        full_classification_report = classification_report(holdout_target, class_instance.predicted_classes[alg])
                        print(full_classification_report)
                    else:
                        scoring = mean_absolute_error(holdout_target, class_instance.predicted_values[alg])
                        scoring_2 = r2_score(holdout_target, class_instance.predicted_values[alg])
                        scoring_3 = mean_squared_error(holdout_target, class_instance.predicted_values[alg], squared=True)
                except Exception:
                    try:
                        if class_instance.class_problem in ["binary", "multiclass"]:
                            scoring = matthews_corrcoef(pd.Series(holdout_target).astype(bool),
                                                        class_instance.predicted_classes[alg])
                            scoring_2 = accuracy_score(pd.Series(holdout_target).astype(bool),
                                                       class_instance.predicted_classes[alg])
                            scoring_3 = recall_score(pd.Series(holdout_target).astype(bool),
                                                     class_instance.predicted_classes[alg], average='weighted')
                            full_classification_report = classification_report(pd.Series(holdout_target).astype(bool),
                                                                               class_instance.predicted_classes[alg])
                            print(full_classification_report)
                        else:
                            scoring = mean_absolute_error(pd.Series(holdout_target).astype(float), class_instance.predicted_values[alg])
                            scoring_2 = r2_score(pd.Series(holdout_target).astype(float), class_instance.predicted_values[alg])
                            scoring_3 = mean_squared_error(pd.Series(holdout_target).astype(float), class_instance.predicted_values[alg], squared=True)
                    except Exception:
                        print("Evaluation on holdout failed.")
                        if class_instance.class_problem in ["binary", "multiclass"]:
                            scoring = 0
                            scoring_2 = 0
                            scoring_3 = 0
                        else:
                            scoring = 999999999
                            scoring_2 = -1
                            scoring_3 = 99999999
            except Exception:
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
            preprocessing_steps_used.append(class_instance.blueprint_step_selection_non_nlp)
            del class_instance
            _ = gc.collect
            results_dict = {
                "Trial number": unique_indices,
                "Algorithm": algorithms_used,
                metric: scoring_results,
                "Preprocessing applied": preprocessing_steps_used,
                "Runtime in seconds": elapsed_times}

            results_df = pd.DataFrame(results_dict)
            results_df.to_pickle(experiment_name)
            print(f"End iteration for algorithm {alg} at {end}.")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    results_dict = {
        "Trial number": unique_indices,
        "Algorithm": algorithms_used,
        metric: scoring_results,
        metric_2: scoring_2_results,
        metric_3: scoring_3_results,
        "Preprocessing applied": preprocessing_steps_used,
        "Runtime in seconds": elapsed_times}

    results_df = pd.DataFrame(results_dict)
    results_df.to_pickle(experiment_name)

    try:
        print(results_df.sort_values(by=[metric], ascending=[ascending]))
        fig = px.line(results_df, x="Runtime in seconds", y=metric, color="Algorithm", text="Trial number")
        fig.update_traces(textposition="bottom right")
        fig.show()
    except Exception:
        pass
    return results_df

