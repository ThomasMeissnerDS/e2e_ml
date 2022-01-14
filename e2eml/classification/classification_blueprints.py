import logging

from e2eml.classification.classification_models import ClassificationModels
from e2eml.classification.nlp_classification import NlpModel
from e2eml.full_processing.preprocessing_blueprints import PreprocessingBluePrint


class ClassificationBluePrint(ClassificationModels, PreprocessingBluePrint, NlpModel):
    """
    Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
    if the predict_mode attribute is True.
    This class stores all classification blueprints.

    :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
    :param df: Accepts a dataframe to make predictions on new data.
        This class also stores all model training and prediction methods for classification tasks (inherited).
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
            self.xgboost_predict(feat_importance=True)
            self.classification_eval(algorithm=algorithm)
            self.classification_eval(algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "ngboost":
            # train Ngboost
            if self.prediction_mode:
                pass
            else:
                self.ngboost_train(tune_mode=self.tune_mode)
            self.ngboost_predict(feat_importance=True, importance_alg="permutation")
            self.classification_eval(algorithm=algorithm)
            self.classification_eval(algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "lgbm":
            # train LGBM
            if self.prediction_mode:
                pass
            else:
                try:
                    self.lgbm_train(tune_mode=self.tune_mode)
                except Exception:
                    self.lgbm_train(tune_mode=self.tune_mode)
            self.lgbm_predict(feat_importance=True)
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "vowpal_wabbit":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.vowpal_wabbit_train()
            self.vowpal_wabbit_predict(
                feat_importance=True, importance_alg="permutation"
            )
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "tabnet":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.tabnet_train()
            self.tabnet_predict()
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "ridge":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.ridge_classifier_train()
            self.ridge_classifier_predict()
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "catboost":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.catboost_train()
            self.catboost_predict()
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "logistic_regression":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.logistic_regression_train()
            self.logistic_regression_predict()
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass
        elif algorithm == "sklearn_ensemble":
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.sklearn_ensemble_train()
            self.sklearn_ensemble_predict()
            self.classification_eval(algorithm=algorithm)
            if self.class_problem == "multiclass":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs[algorithm],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        algorithm
                    ],
                )
            else:
                pass

    def ml_bp00_train_test_binary_full_processing_log_reg_prob(self, df=None):
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
            self.logistic_regression_train()
        algorithm = "logistic_regression"
        self.logistic_regression_predict(
            feat_importance=True, importance_alg="permutation"
        )
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp01_multiclass_full_processing_xgb_prob(self, df=None):
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
            self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
        self.xgboost_predict(feat_importance=True)
        algorithm = "xgboost"
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp02_multiclass_full_processing_lgbm_prob(self, df=None):
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
        algorithm = "lgbm"
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        logging.info("Start blueprint.")
        self.runtime_warnings(warn_about="long runtime")
        self.std_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(
            feat_importance=True, importance_alg="permutation"
        )
        algorithm = "sklearn_ensemble"
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp04_multiclass_full_processing_ngboost(self, df=None):
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
        algorithm = "ngboost"
        self.classification_eval(algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp05_multiclass_full_processing_vowpal_wabbit(self, df=None):
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
        self.vowpal_wabbit_predict(feat_importance=True, importance_alg="permutation")
        algorithm = "vowpal_wabbit"
        self.classification_eval(algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp06_multiclass_full_processing_bert_transformer(self, df=None):
        """
        Runs an NLP transformer blue print specifically for text classification. Can be used as a pipeline to predict on new data,
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
        self.classification_eval(algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp07_multiclass_full_processing_tabnet(self, df=None):
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
            self.tabnet_train()
        algorithm = "tabnet"
        self.tabnet_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp08_multiclass_full_processing_ridge(self, df=None):
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
            self.ridge_classifier_train()
        algorithm = "ridge"
        self.ridge_classifier_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp09_multiclass_full_processing_catboost(self, df=None):
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
            self.catboost_train()
        algorithm = "catboost"
        self.catboost_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp10_multiclass_full_processing_sgd(self, df=None):
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
            self.sgd_classifier_train()
        algorithm = "sgd"
        self.sgd_classifier_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp11_multiclass_full_processing_quadratic_discriminant_analysis(
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
            self.quadratic_discriminant_analysis_train()
        algorithm = "quadratic_discriminant_analysis"
        self.quadratic_discriminant_analysis_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp12_multiclass_full_processing_svm(self, df=None):
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
            self.svm_train()
        algorithm = "svm"
        self.svm_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp13_multiclass_full_processing_multinomial_nb(self, df=None):
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
            self.multinomial_nb_train()
        algorithm = "multinomial_nb"
        self.multinomial_nb_predict()
        self.classification_eval(algorithm=algorithm)
        if self.class_problem == "binary":
            self.visualize_probability_distribution(
                probabilities=self.predicted_probs[algorithm],
                threshold=self.preprocess_decisions["probability_threshold"][algorithm],
            )
        else:
            pass
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp14_multiclass_full_processing_lgbm_focal(self, df=None):
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
            self.lgbm_focal_train()
        algorithm = "lgbm_focal"
        self.lgbm_focal_predict()
        self.classification_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp15_multiclass_full_processing_deesc(self, df=None):
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
            self.deesc_train()
        algorithm = "deesc"
        self.deesc_predict()
        self.classification_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_special_multiclass_full_processing_multimodel_max_voting(  # noqa: C901
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
            if self.special_blueprint_algorithms["lgbm"]:
                self.lgbm_train(tune_mode=self.tune_mode)
            if self.special_blueprint_algorithms["xgboost"]:
                self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                self.vowpal_wabbit_train()
            if self.special_blueprint_algorithms["tabnet"]:
                self.tabnet_train()
            if self.special_blueprint_algorithms["ridge"]:
                self.ridge_classifier_train()
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                self.sklearn_ensemble_train()
            if self.special_blueprint_algorithms["ngboost"]:
                self.ngboost_train()
            if self.special_blueprint_algorithms["catboost"]:
                self.catboost_train()

        if self.special_blueprint_algorithms["lgbm"]:
            self.lgbm_predict(feat_importance=False)
            self.classification_eval("lgbm")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["lgbm"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "lgbm"
                    ],
                )
        if self.special_blueprint_algorithms["xgboost"]:
            self.xgboost_predict(feat_importance=True)
            self.classification_eval("xgboost")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["xgboost"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "xgboost"
                    ],
                )
        if self.special_blueprint_algorithms["vowpal_wabbit"]:
            self.vowpal_wabbit_predict(feat_importance=True)
            self.classification_eval("vowpal_wabbit")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["vowpal_wabbit"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "vowpal_wabbit"
                    ],
                )
        if self.special_blueprint_algorithms["tabnet"]:
            self.tabnet_predict()
            self.classification_eval("tabnet")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["tabnet"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "tabnet"
                    ],
                )
        if self.special_blueprint_algorithms["ridge"]:
            self.ridge_classifier_predict()
            self.classification_eval("ridge")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["ridge"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "ridge"
                    ],
                )
        if self.special_blueprint_algorithms["sklearn_ensemble"]:
            self.sklearn_ensemble_predict()
            self.classification_eval("sklearn_ensemble")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["sklearn_ensemble"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "sklearn_ensemble"
                    ],
                )
        if self.special_blueprint_algorithms["ngboost"]:
            self.ngboost_predict(feat_importance=False)
            self.classification_eval("ngboost")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["ngboost"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "ngboost"
                    ],
                )
        if self.special_blueprint_algorithms["catboost"]:
            self.catboost_predict()
            self.classification_eval("catboost")
            if self.class_problem == "binary":
                self.visualize_probability_distribution(
                    probabilities=self.predicted_probs["catboost"],
                    threshold=self.preprocess_decisions["probability_threshold"][
                        "catboost"
                    ],
                )

        mode_cols = [
            alg for alg, value in self.special_blueprint_algorithms.items() if value
        ]
        try:
            mode_cols.remove("elasticnet")
        except Exception:
            pass

        if self.prediction_mode:
            if self.special_blueprint_algorithms["lgbm"]:
                self.dataframe["lgbm"] = self.predicted_classes["lgbm"]
            if self.special_blueprint_algorithms["xgboost"]:
                self.dataframe["xgboost"] = self.predicted_classes["xgboost"]
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                self.dataframe["vowpal_wabbit"] = self.predicted_classes[
                    "vowpal_wabbit"
                ]
            if self.special_blueprint_algorithms["tabnet"]:
                self.dataframe["tabnet"] = self.predicted_classes["tabnet"]
            if self.special_blueprint_algorithms["ridge"]:
                self.dataframe["ridge"] = self.predicted_classes["ridge"]
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                self.dataframe["sklearn_ensemble"] = self.predicted_classes[
                    "sklearn_ensemble"
                ]
            if self.special_blueprint_algorithms["ngboost"]:
                self.dataframe["ngboost"] = self.predicted_classes["ngboost"]
            if self.special_blueprint_algorithms["catboost"]:
                self.dataframe["catboost"] = self.predicted_classes["catboost"]
            self.dataframe["max_voting_class"] = self.dataframe[mode_cols].mode(axis=1)[
                0
            ]
            self.predicted_classes["max_voting"] = self.dataframe["max_voting_class"]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if self.special_blueprint_algorithms["lgbm"]:
                X_test["lgbm"] = self.predicted_classes["lgbm"]
            if self.special_blueprint_algorithms["xgboost"]:
                X_test["xgboost"] = self.predicted_classes["xgboost"]
            if self.special_blueprint_algorithms["vowpal_wabbit"]:
                X_test["vowpal_wabbit"] = self.predicted_classes["vowpal_wabbit"]
            if self.special_blueprint_algorithms["tabnet"]:
                X_test["tabnet"] = self.predicted_classes["tabnet"]
            if self.special_blueprint_algorithms["ridge"]:
                X_test["ridge"] = self.predicted_classes["ridge"]
            if self.special_blueprint_algorithms["sklearn_ensemble"]:
                X_test["sklearn_ensemble"] = self.predicted_classes["sklearn_ensemble"]
            if self.special_blueprint_algorithms["ngboost"]:
                X_test["ngboost"] = self.predicted_classes["ngboost"]
            if self.special_blueprint_algorithms["catboost"]:
                X_test["catboost"] = self.predicted_classes["catboost"]
            X_test["max_voting_class"] = X_test[mode_cols].mode(axis=1)[0]
            self.predicted_classes["max_voting"] = X_test["max_voting_class"]
        self.classification_eval("max_voting")
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_special_multiclass_auto_model_exploration(self, df=None):
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
            if self.special_blueprint_algorithms["catboost"]:
                self.train_pred_selected_model(algorithm="catboost")

            # select best model
            max_matthews = 0
            self.best_model = "xgboost"
            for k, v in self.evaluation_scores.items():
                if (v["matthews"]) > max_matthews:
                    max_matthews = v["matthews"]
                    self.best_model = k
                    print(f"Best model is {self.best_model} with matthews of {v}")
            self.train_pred_selected_model(algorithm=self.best_model)
            self.prediction_mode = True
        else:
            self.train_pred_selected_model(algorithm=self.best_model)
        logging.info("Finished blueprint.")
