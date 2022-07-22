import logging

from e2eml.full_processing.preprocessing_blueprints import PreprocessingBluePrint
from e2eml.time_series.lstm_model import LstmModel
from e2eml.time_series.lstm_model_with_quantile_loss import LSTMQuantileModel
from e2eml.time_series.rnn_model import RNNModel
from e2eml.time_series.time_series_models import (
    RegressionForTimeSeriesModels,
    UnivariateTimeSeriesModels,
)


class TimeSeriesBluePrint(
    UnivariateTimeSeriesModels,
    PreprocessingBluePrint,
    LstmModel,
    RNNModel,
    RegressionForTimeSeriesModels,
    LSTMQuantileModel,
):
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

    def ml_bp100_univariate_timeseries_full_processing_auto_arima(
        self, df=None, n_forecast=1
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
        self.arima_preprocessing_pipeline()
        if self.prediction_mode:
            pass
        else:
            self.auto_arima_train()
        algorithm = "auto_arima"
        self.auto_arima_predict(n_forecast=n_forecast)
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp101_multivariate_timeseries_full_processing_lstm(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.lstm_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.lstm_train()
        algorithm = "lstm"
        self.lstm_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp102_multivariate_timeseries_full_processing_tabnet(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.regression_for_time_series_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.tabnet_regression_train()
        algorithm = "tabnet"
        self.tabnet_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp103_multivariate_timeseries_full_processing_rnn(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.lstm_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.rnn_train()
        algorithm = "rnn"
        self.rnn_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp104_univariate_timeseries_full_processing_holt_winters(
        self, df=None, n_forecast=1
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
        self.holt_winters_preprocessing_pipeline()
        if self.prediction_mode:
            pass
        else:
            self.holt_winters_train()
        algorithm = "holt_winters"
        self.holt_winters_predict(n_forecast)
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp105_multivariate_timeseries_full_processing_lstm_quantile(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :param preprocess_bp: Chose the preprocessing pipeline blueprint ("bp_01", "bp_02" or "bp_03")
        :return: Updates class attributes by its predictions.
        """
        self.lstm_quantile_preprocessing_pipeline(df=df)
        if self.prediction_mode:
            pass
        else:
            self.lstm_quantile_regression_train()
        algorithm = "lstm_quantile_regression"
        self.lstm_quantile_regression_predict()
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")

    def ml_bp106_univariate_timeseries_full_processing_thymeboost(
        self, df=None, n_forecast=1
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
        self.thymeboost_preprocessing_pipeline()
        if self.prediction_mode:
            pass
        else:
            self.thymeboost_train()
        algorithm = "thymeboost"
        self.thymeboost_predict(n_forecast=n_forecast)
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info("Finished blueprint.")
