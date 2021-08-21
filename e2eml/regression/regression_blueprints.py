from e2eml.regression.regression_models import RegressionModels
from e2eml.full_processing.preprocessing_blueprints import PreprocessingBluePrint
from e2eml.regression.nlp_regression import NlpModel
import logging


class RegressionBluePrint(RegressionModels, PreprocessingBluePrint,  NlpModel):
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

    def train_pred_selected_model(self, algorithm=None):
        logging.info(f'Start ML training {algorithm}')
        if algorithm == 'xgboost':
            # train Xgboost
            if self.prediction_mode:
                pass
            else:
                self.xg_boost_train(autotune=True, tune_mode=self.tune_mode)
            self.xgboost_predict(feat_importance=False)
            self.regression_eval(algorithm=algorithm)
        elif algorithm == 'ngboost':
            # train Ngboost
            if self.prediction_mode:
                pass
            else:
                self.ngboost_train(tune_mode=self.tune_mode)
            self.ngboost_predict(feat_importance=True, importance_alg='permutation')
            self.regression_eval(algorithm=algorithm)
        elif algorithm == 'lgbm':
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
        elif algorithm == 'vowpal_wabbit':
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.vowpal_wabbit_train()
            self.vowpal_wabbit_predict(feat_importance=True, importance_alg='permutation')
            self.regression_eval(algorithm=algorithm)
        elif algorithm == 'tabnet':
            # train sklearn ensemble
            if self.prediction_mode:
                pass
            else:
                self.tabnet_regression_train()
            self.tabnet_regression_predict()
            self.regression_eval(algorithm=algorithm)

    def ml_bp10_train_test_regression_full_processing_linear_reg(self, df=None, preprocessing_type='full', preprocess_bp="bp_03"):
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
            self.linear_regression_train()
        algorithm = 'linear_regression'
        self.linear_regression_predict(feat_importance=True, importance_alg='permutation')
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp11_regression_full_processing_xgboost(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
            self.xg_boost_train(autotune=True)
        self.xgboost_predict(feat_importance=True)
        self.regression_eval('xgboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp12_regressions_full_processing_lgbm(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
        self.regression_eval('lgbm')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp13_regression_full_processing_sklearn_stacking_ensemble(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'sklearn_ensemble'
        self.regression_eval('sklearn_ensemble')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp14_regressions_full_processing_ngboost(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
        self.regression_eval('ngboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp15_regression_full_processing_vowpal_wabbit_reg(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
        algorithm = 'vowpal_wabbit'
        self.vowpal_wabbit_predict(feat_importance=True, importance_alg='permutation')
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp16_regressions_full_processing_bert_transformer(self, df=None, preprocess_bp='bp_nlp_10'):
        """
        Runs an NLP transformer blue print specifically for text regression. Can be used as a pipeline to predict on new data,
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
        self.regression_eval(algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp17_regression_full_processing_tabnet_reg(self, df=None, preprocessing_type='full', preprocess_bp="bp_04"):
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
            self.tabnet_regression_train()
        algorithm = 'tabnet'
        self.tabnet_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_regression_multiclass_full_processing_multimodel_avg_blender(self, df=None, preprocessing_type='full', preprocess_bp="bp_04"):
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
            self.lgbm_train(tune_mode=self.tune_mode)
            self.vowpal_wabbit_train()
            self.tabnet_regression_train()
        self.ngboost_predict(feat_importance=True)
        self.lgbm_predict(feat_importance=True)
        self.vowpal_wabbit_predict(feat_importance=True)
        self.tabnet_regression_predict()
        if self.prediction_mode:
            self.dataframe["lgbm_preds"] = self.predicted_values[f"lgbm"]
            self.dataframe["ngboost_preds"] = self.predicted_values[f"ngboost"]
            self.dataframe["tabnet"] = self.predicted_values[f"tabnet"]
            self.dataframe["blended_preds"] = (self.dataframe["lgbm_preds"] + self.dataframe["ngboost_preds"] + self.dataframe["tabnet"])/3
            self.predicted_values[f"blended_preds"] = self.dataframe["blended_preds"]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_test["lgbm_preds"] = self.predicted_values[f"lgbm"]
            X_test["ngboost_preds"] = self.predicted_values[f"ngboost"]
            X_test["tabnet_preds"] = self.predicted_values[f"tabnet"]
            X_test["blended_preds"] = (X_test["lgbm_preds"] + X_test["ngboost_preds"] + X_test["tabnet_preds"])/3
            self.predicted_values[f"blended_preds"] = X_test["blended_preds"]
        self.regression_eval('blended_preds')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_regression_auto_model_exploration(self, df=None, preprocessing_type='full', preprocess_bp="bp_01"):
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
            self.train_pred_selected_model(algorithm='vowpal_wabbit')
            self.train_pred_selected_model(algorithm='tabnet')

            # select best model
            min_mae = 10000000
            self.best_model = 'xgboost'
            for k, v in self.evaluation_scores.items():
                if (v['mae']) < min_mae:
                    min_mae = (v['mae'])
                    self.best_model = k
                    print(f"Best model is {self.best_model} with mean absolute error of {v}")
            self.train_pred_selected_model(algorithm=self.best_model)
            self.prediction_mode = True
        else:
            self.train_pred_selected_model(algorithm=self.best_model)
        logging.info('Finished blueprint.')