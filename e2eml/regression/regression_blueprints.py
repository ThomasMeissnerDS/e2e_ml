import logging

from e2eml.full_processing.preprocessing_blueprints import PreprocessingBluePrint
from e2eml.regression.nlp_regression import NlpModel
from e2eml.regression.regression_models import RegressionModels


class RegressionBluePrint(RegressionModels, PreprocessingBluePrint, NlpModel):
    """
    Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
    if the predict_mode attribute is True.
    This class stores all regression blueprints.
    :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
    "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
    :param df: Accepts a dataframe to make predictions on new data.

    This class also stores all model training and prediction methods for regression tasks (inherited).
    This class also stores all pipeline relevant information (inherited from cpu preprocessing).
    The attribute "df_dict" always holds train and test as well as to predict data. The attribute
    "preprocess_decisions" stores encoders and other information generated during the
    model training. The attributes "predicted_classes" and "predicted_probs" store dictionaries (model names are dictionary keys)
    with predicted classes and probabilities (classification tasks) while "predicted_values" stores regression based
    predictions. The attribute "evaluation_scores" keeps track of model evaluation metrics (in dictionary format).
    :param datasource: Expects a Pandas dataframe (containing the target feature as a column)
    :param target_variable: Name of the target feature's column within the datasource dataframe.
    :param date_columns: Date columns can be passed as lists additionally for respective preprocessing. If not provided
    e2eml will try to detect datetime columns automatically. Date format is expected as YYYY-MM-DD anyway.
    :param categorical_columns: Categorical columns can be passed as lists additionally for respective preprocessing.
    If not provided e2eml will try to detect categorical columns automatically.
    :param nlp_columns: NLP columns can be passed specifically. This only makes sense, if the chosen blueprint runs under 'nlp' processing.
    If NLP columns are not declared, categorical columns will be interpreted as such.
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
    :return: Updates class attributes by its predictions.
    """

    def train_pred_selected_model(self, algorithm=None):  # noqa: C901
        logging.info(f"Start ML training {algorithm}")
        if algorithm == "xgboost":
            # train Xgboost
            if self.prediction_mode:
                pass
            else:
                self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
            self.xgboost_predict(feat_importance=False)
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "ngboost":
            # train Ngboost
            if self.prediction_mode:
                pass
            else:
                self.ngboost_train(tune_mode=self.tune_mode)
            self.ngboost_predict(feat_importance=True, importance_alg="permutation")
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "lgbm":
            # train LGBM
            if self.prediction_mode:
                pass
            else:
                try:
                    self.lgbm_train(tune_mode=self.tune_mode)
                except Exception:
                    self.lgbm_train(tune_mode=self.tune_mode)
            self.lgbm_predict(feat_importance=False)
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "vowpal_wabbit":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.vowpal_wabbit_train()
            self.vowpal_wabbit_predict(
                feat_importance=True, importance_alg="permutation"
            )
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "tabnet":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.tabnet_regression_train()
            self.tabnet_regression_predict()
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "ridge":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.ridge_regression_train()
            self.ridge_regression_predict()
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "elasticnet":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.elasticnet_regression_train()
            self.elasticnet_regression_predict()
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "catboost":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.catboost_regression_train()
            self.catboost_regression_predict()
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "linear_regression":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.linear_regression_train()
            self.linear_regression_predict()
            self.regression_eval(algorithm=algorithm)
        elif algorithm == "sklearn_ensemble":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.sklearn_ensemble_train()
            self.sklearn_ensemble_predict()
            self.regression_eval(algorithm=algorithm)

    def ml_bp10_train_test_regression_full_processing_linear_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.linear_regression_train()
        algorithm = "linear_regression"
        self.linear_regression_predict(
            feat_importance=True, importance_alg="permutation"
        )
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp11_regression_full_processing_xgboost(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.xg_boost_train(autotune=True)
        self.xgboost_predict(feat_importance=True)
        self.regression_eval("xgboost")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp12_regressions_full_processing_lgbm(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.lgbm_train(tune_mode=self.tune_mode)
        self.lgbm_predict(feat_importance=True)
        self.regression_eval("lgbm")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp13_regression_full_processing_sklearn_stacking_ensemble(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(
            feat_importance=True, importance_alg="permutation"
        )
        self.regression_eval("sklearn_ensemble")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp14_regressions_full_processing_ngboost(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.ngboost_train(tune_mode=self.tune_mode)
        self.ngboost_predict(feat_importance=True, importance_alg="permutation")
        self.regression_eval("ngboost")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp15_regression_full_processing_vowpal_wabbit_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.vowpal_wabbit_train()
        algorithm = "vowpal_wabbit"
        self.vowpal_wabbit_predict(feat_importance=True, importance_alg="permutation")
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp16_regressions_full_processing_bert_transformer(self, df=None):
        """
        Runs an NLP transformer blue print specifically for text regression. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_nlp_10" or "bp_nlp_11")
        :return: Updates class attributes by its predictions.
        """
        self.nlp_transformer_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.transformer_train()
        self.transformer_predict()
        algorithm = "nlp_transformer"
        self.regression_eval(algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp17_regression_full_processing_tabnet_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.tabnet_regression_train()
        algorithm = "tabnet"
        self.tabnet_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp18_regression_full_processing_ridge_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.ridge_regression_train()
        algorithm = "ridge"
        self.ridge_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp19_regression_full_processing_elasticnet_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.elasticnet_regression_train()
        algorithm = "elasticnet"
        self.elasticnet_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp20_regression_full_processing_catboost(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.catboost_regression_train()
        algorithm = "catboost"
        self.catboost_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp20_regression_full_processing_sgd(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.sgd_regression_train()
        algorithm = "sgd"
        self.sgd_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp21_regression_full_processing_ransac(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.ransac_regression_train()
        algorithm = "ransac"
        self.ransac_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp22_regression_full_processing_svm(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.svm_regression_train()
        algorithm = "svm_regression"
        self.svm_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_special_regression_full_processing_multimodel_avg_blender(  # noqa: C901
        self, df=None
    ):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            if self.special_blueprint_algorithms["ngboost"]:
                self.ngboost_train(tune_mode=self.tune_mode)
            if self.special_blueprint_algorithms["lgbm"]:
                self.lgbm_train(tune_mode=self.tune_mode)
            if self.special_blueprint_algorithms["xgboost"]:
                self.xg_boost_train()
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                self.vowpal_wabbit_train()
            if self.special_blueprint_algorithms["tabnet"]:
                self.tabnet_regression_train()
            if self.special_blueprint_algorithms["ridge"]:
                self.ridge_regression_train()
            if self.special_blueprint_algorithms["elasticnet"]:
                self.elasticnet_regression_train()
            if self.special_blueprint_algorithms["catboost"]:
                self.catboost_regression_train()
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                self.sklearn_ensemble_train()

        if self.special_blueprint_algorithms["ngboost"]:
            self.ngboost_predict(feat_importance=False)
        if self.special_blueprint_algorithms["lgbm"]:
            self.lgbm_predict(feat_importance=True)
        if self.special_blueprint_algorithms["xgboost"]:
            self.xgboost_predict()
        if self.special_blueprint_algorithms["vowpal_wabbit"]:
            self.vowpal_wabbit_predict(feat_importance=True)
        if self.special_blueprint_algorithms["tabnet"]:
            self.tabnet_regression_predict()
        if self.special_blueprint_algorithms["ridge"]:
            self.ridge_regression_predict()
        if self.special_blueprint_algorithms["elasticnet"]:
            self.elasticnet_regression_predict()
        if self.special_blueprint_algorithms["catboost"]:
            self.catboost_regression_predict()
        if self.special_blueprint_algorithms["sklearn_ensemble"]:
            self.sklearn_ensemble_predict()

        mode_cols = [
            alg for alg, value in self.special_blueprint_algorithms.items() if value
        ]

        if self.prediction_mode:
            if self.special_blueprint_algorithms["ngboost"]:
                self.dataframe["ngboost"] = self.predicted_values["ngboost"]
            if self.special_blueprint_algorithms["lgbm"]:
                self.dataframe["lgbm"] = self.predicted_values["lgbm"]
            if self.special_blueprint_algorithms["xgboost"]:
                self.dataframe["xgboost"] = self.predicted_values["xgboost"]
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                self.dataframe["vowpal_wabbit"] = self.predicted_values["vowpal_wabbit"]
            if self.special_blueprint_algorithms["tabnet"]:
                self.dataframe["tabnet"] = self.predicted_values["tabnet"]
            if self.special_blueprint_algorithms["ridge"]:
                self.dataframe["ridge"] = self.predicted_values["ridge"]
            if self.special_blueprint_algorithms["elasticnet"]:
                self.dataframe["elasticnet"] = self.predicted_values["elasticnet"]
            if self.special_blueprint_algorithms["catboost"]:
                self.dataframe["catboost"] = self.predicted_values["catboost"]
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                self.dataframe["sklearn_ensemble"] = self.predicted_values[
                    "sklearn_ensemble"
                ]

            self.dataframe["blended_preds"] = self.dataframe[mode_cols].sum(
                axis=1
            ) / len(mode_cols)
            self.predicted_values["blended_preds"] = self.dataframe["blended_preds"]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if self.special_blueprint_algorithms["lgbm"]:
                X_test["lgbm"] = self.predicted_values["lgbm"]
            if self.special_blueprint_algorithms["ngboost"]:
                X_test["ngboost"] = self.predicted_values["ngboost"]
            if self.special_blueprint_algorithms["xgboost"]:
                X_test["xgboost"] = self.predicted_values["xgboost"]
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                X_test["vowpal_wabbit"] = self.predicted_values["vowpal_wabbit"]
            if self.special_blueprint_algorithms["tabnet"]:
                X_test["tabnet"] = self.predicted_values["tabnet"]
            if self.special_blueprint_algorithms["ridge"]:
                X_test["ridge"] = self.predicted_values["ridge"]
            if self.special_blueprint_algorithms["elasticnet"]:
                X_test["elasticnet"] = self.predicted_values["elasticnet"]
            if self.special_blueprint_algorithms["catboost"]:
                X_test["catboost"] = self.predicted_values["catboost"]
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                X_test["sklearn_ensemble"] = self.predicted_values["sklearn_ensemble"]

            X_test["blended_preds"] = X_test[mode_cols].sum(axis=1) / len(mode_cols)
            self.predicted_values["blended_preds"] = X_test["blended_preds"]
        self.regression_eval("blended_preds")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_special_regression_auto_model_exploration(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.std_preprocessing_pipeline(df=df)
        if not self.prediction_mode:
            if self.special_blueprint_algorithms["ridge"]:
                self.train_pred_selected_model(algorithm="ridge")
            if self.special_blueprint_algorithms["elasticnet"]:
                self.train_pred_selected_model(algorithm="elasticnet")
            if self.special_blueprint_algorithms["catboost"]:
                self.train_pred_selected_model(algorithm="catboost")
            if self.special_blueprint_algorithms["lgbm"]:
                self.train_pred_selected_model(algorithm="lgbm")
            if self.special_blueprint_algorithms["xgboost"]:
                self.train_pred_selected_model(algorithm="xgboost")
            if self.special_blueprint_algorithms["ngboost"]:
                self.train_pred_selected_model(algorithm="ngboost")
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                self.train_pred_selected_model(algorithm="vowpal_wabbit")
            if self.special_blueprint_algorithms["tabnet"]:
                self.train_pred_selected_model(algorithm="tabnet")
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                self.train_pred_selected_model(algorithm="sklearn_ensemble")

            # select best model
            min_mae = 10000000
            self.best_model = "xgboost"
            for k, v in self.evaluation_scores.items():
                if (v["mae"]) < min_mae:
                    min_mae = v["mae"]
                    self.best_model = k
                    print(
                        f"Best model is {self.best_model} with mean absolute error of {v}"
                    )
            self.train_pred_selected_model(algorithm=self.best_model)
            self.prediction_mode = True
        else:
            self.train_pred_selected_model(algorithm=self.best_model)
        logging.info("Finished blueprint.")
