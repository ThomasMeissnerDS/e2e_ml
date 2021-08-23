from e2eml.classification.classification_models import ClassificationModels
from e2eml.full_processing.preprocessing_blueprints import PreprocessingBluePrint
from e2eml.classification.nlp_classification import NlpModel
import logging


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

    def train_pred_selected_model(self, algorithm=None):
        logging.info(f'Start ML training {algorithm}')
        if algorithm == 'xgboost':
            # train Xgboost
            if self.prediction_mode:
                pass
            else:
                self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
            self.xgboost_predict(feat_importance=True)
            self.classification_eval(algorithm=algorithm)
        elif algorithm == 'ngboost':
            # train Ngboost
            if self.prediction_mode:
                pass
            else:
                self.ngboost_train(tune_mode=self.tune_mode)
            self.ngboost_predict(feat_importance=True, importance_alg='permutation')
            self.classification_eval(algorithm=algorithm)
        elif algorithm == 'lgbm':
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
        elif algorithm == 'vowpal_wabbit':
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.vowpal_wabbit_train()
            self.vowpal_wabbit_predict(feat_importance=True, importance_alg='permutation')
            self.classification_eval(algorithm=algorithm)

    def ml_bp00_train_test_binary_full_processing_log_reg_prob(self, df=None, preprocessing_type='full', preprocess_bp="bp_03"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.logistic_regression_train()
        algorithm = 'logistic_regression'
        self.logistic_regression_predict(feat_importance=True, importance_alg='permutation')
        self.classification_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp01_multiclass_full_processing_xgb_prob(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
        self.xgboost_predict(feat_importance=True)
        self.classification_eval('xgboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp02_multiclass_full_processing_lgbm_prob(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.lgbm_train(tune_mode=self.tune_mode)
        self.lgbm_predict(feat_importance=True)
        self.classification_eval('lgbm')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="long runtime")
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all')
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca()
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(how='static')
        self.data_binning(nb_bins=10)
        #self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='logloss')
        self.sort_columns_alphabetically()
        self.smote_data()
        if self.prediction_mode:
            pass
        else:
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'sklearn_ensemble'
        self.classification_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp04_multiclass_full_processing_ngboost(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.ngboost_train(tune_mode=self.tune_mode)
        self.ngboost_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'ngboost'
        self.classification_eval(algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp05_multiclass_full_processing_vowpal_wabbit(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.vowpal_wabbit_train()
        self.vowpal_wabbit_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'vowpal_wabbit'
        self.classification_eval(algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp06_multiclass_full_processing_bert_transformer(self, df=None, preprocess_bp="bp_nlp_10"):
        """
        Runs an NLP transformer blue print specifically for text classification. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_nlp_10" or "bp_nlp_11")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_nlp_10':
            self.pp_bp10_nlp_preprocessing(df=df)
        elif preprocess_bp == 'bp_nlp_11':
            self.pp_bp11_nlp_preprocessing(df=df)
        elif preprocess_bp == 'bp_nlp_12':
            self.pp_bp12_nlp_preprocessing(df=df)
        elif preprocess_bp == 'bp_nlp_13':
            self.pp_bp13_nlp_preprocessing(df=df)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.transformer_train()
        self.transformer_predict()
        algorithm = 'nlp_transformer'
        self.classification_eval(algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp07_multiclass_full_processing_tabnet(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.tabnet_train()
        algorithm = 'tabnet'
        self.tabnet_predict()
        self.classification_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_multiclass_full_processing_multimodel_max_voting(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if self.prediction_mode:
            pass
        else:
            self.lgbm_train(tune_mode=self.tune_mode)
            self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
            self.vowpal_wabbit_train()
            self.tabnet_train()
        self.lgbm_predict(feat_importance=False)
        self.classification_eval('lgbm')
        self.xgboost_predict(feat_importance=True)
        self.classification_eval('xgboost')
        self.vowpal_wabbit_predict(feat_importance=True)
        self.classification_eval('vowpal_wabbit')
        self.tabnet_predict()
        self.classification_eval('tabnet')
        algorithm = 'max_voting'
        mode_cols = ["lgbm_class",
                     "xgboost_class",
                     "vowpal_wabbit_class",
                     "tabnet"]
        if self.prediction_mode:
            self.dataframe["lgbm_class"] = self.predicted_classes[f"lgbm"]
            self.dataframe["xgboost_class"] = self.predicted_classes[f"xgboost"]
            self.dataframe["vowpal_wabbit_class"] = self.predicted_classes[f"vowpal_wabbit"]
            self.dataframe["tabnet"] = self.predicted_classes[f"tabnet"]
            self.dataframe["max_voting_class"] = self.dataframe[mode_cols].mode(axis=1)[0]
            self.predicted_classes[f"max_voting"] = self.dataframe["max_voting_class"]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_test["lgbm_class"] = self.predicted_classes[f"lgbm"]
            X_test["xgboost_class"] = self.predicted_classes[f"xgboost"]
            X_test["vowpal_wabbit_class"] = self.predicted_classes[f"vowpal_wabbit"]
            X_test["tabnet"] = self.predicted_classes[f"tabnet"]
            X_test["max_voting_class"] = X_test[mode_cols].mode(axis=1)[0]
            self.predicted_classes[f"max_voting"] = X_test["max_voting_class"]
        self.classification_eval('max_voting')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_multiclass_auto_model_exploration(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        if preprocess_bp == 'bp_01':
            self.pp_bp01_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_02':
            self.pp_bp02_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_03':
            self.pp_bp03_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        elif preprocess_bp == 'bp_04':
            self.pp_bp04_std_preprocessing(df=df, preprocessing_type=preprocessing_type)
        else:
            pass
        if not self.prediction_mode:
            self.train_pred_selected_model(algorithm='lgbm')
            self.train_pred_selected_model(algorithm='xgboost')
            self.train_pred_selected_model(algorithm='ngboost')
            self.train_pred_selected_model(algorithm="vowpal_wabbit")
            self.train_pred_selected_model(algorithm="tabnet")

            # select best model
            max_matthews = 0
            self.best_model = 'xgboost'
            for k, v in self.evaluation_scores.items():
                if (v['matthews']) > max_matthews:
                    max_matthews = (v['matthews'])
                    self.best_model = k
                    print(f"Best model is {self.best_model} with matthews of {v}")
            self.train_pred_selected_model(algorithm=self.best_model)
            self.prediction_mode = True
        else:
            self.train_pred_selected_model(algorithm=self.best_model)
        logging.info('Finished blueprint.')
