import gc
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

from e2eml.full_processing import postprocessing

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnivariateTimeSeriesModels(postprocessing.FullPipeline):
    """
    This class stores all model training and prediction methods for regression tasks.
    This class stores all pipeline relevant information (inherited from cpu preprocessing).
    The attribute "df_dict" always holds train and test as well as
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
    :param nlp_columns: NLP columns can be passed specifically. This only makes sense, if the chosen blueprint runs under 'nlp' processing.
    If NLP columns are not declared, categorical columns will be interpreted as such.
    :param unique_identifier: A unique identifier (i.e. an ID column) can be passed as well to preserve this information
     for later processing.
    :param ml_task: Can be 'binary', 'multiclass' or 'regression'. On default will be determined automatically.
    :param preferred_training_mode: Must be 'cpu', if e2eml has been installed into an environment without LGBM and Xgboost on GPU.
    Can be set to 'gpu', if LGBM and Xgboost have been installed with GPU support. The default 'auto' will detect GPU support
    and optimize accordingly. Only TabNet can only run on GPU and will not be impacted from this parameter. (Default: 'auto')
    :param logging_file_path: Preferred location to save the log file. Will otherwise stored in the current folder.
    :param low_memory_mode: Adds a preprocessing feature to reduce dataframe memory footprint. Will lead to a loss in
    model performance. Will be extended by further memory savings features in future releases.
    However we highly recommend GPU usage to heavily decrease model training times.
    """

    def auto_arima_train(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            algorithm = "auto_arima"
            tscv = TimeSeriesSplit(n_splits=5)

            def objective(trial):
                if self.preprocess_decisions["arima_transformation_used"] == "None":
                    i_term = self.preprocess_decisions["arima_nb_differentiations"]
                else:
                    i_term = 0

                param = {
                    "p": trial.suggest_int("p", 1, 24),
                    "i": i_term,
                    "q": trial.suggest_int("q", 1, 24),
                }

                mean_abs_errors = []
                for train_index, test_index in tscv.split(X_train):
                    x_train, x_test = (
                        X_train.iloc[train_index],
                        X_train.iloc[test_index],
                    )
                    y_train, y_test = (  # noqa: F841
                        Y_train.iloc[train_index],
                        Y_train.iloc[test_index],
                    )
                    model = ARIMA(
                        x_train.values, order=(param["p"], param["i"], param["q"])
                    )
                    try:
                        model = model.fit()
                        preds = model.forecast(len(x_test.index))
                        if (
                            self.preprocess_decisions["arima_transformation_used"]
                            == "log"
                        ):
                            y_test = np.expm1(y_test.astype(float))
                            preds = np.expm1(preds.astype(float))
                        mae = mean_absolute_error(y_test, preds)
                        mean_abs_errors.append(mae)
                    except Exception as e:
                        mae = 9999999999
                        mean_abs_errors.append(mae)
                        print(e)
                return np.mean(np.asarray(mean_abs_errors))

            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=self.global_random_state
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
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = fig
                fig.show()
                fig = optuna.visualization.plot_param_importances(study)
                self.optuna_studies[f"{algorithm}_param_importance"] = fig
                fig.show()
            except ZeroDivisionError:
                pass

            best_parameters = study.best_trial.params
            if self.preprocess_decisions["arima_transformation_used"] == "None":
                best_parameters["i"] = 0
            elif self.preprocess_decisions["arima_transformation_used"] == "log":
                best_parameters["i"] = 0
            elif self.preprocess_decisions["arima_nb_differentiations"] > 0:
                best_parameters["i"] = self.preprocess_decisions[
                    "arima_nb_differentiations"
                ]

            model = ARIMA(
                pd.concat([X_train, X_test]).values,
                order=(
                    best_parameters["p"],
                    best_parameters["i"],
                    best_parameters["q"],
                ),
            )
            model = model.fit()
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def auto_arima_predict(self, n_forecast=1):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Auto Arima")
        algorithm = "auto_arima"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            predicted_values = model.forecast(n_forecast)
            predicted_values[np.isfinite(predicted_values) is False] = 0
            if self.preprocess_decisions["arima_transformation_used"] == "log":
                predicted_values = np.expm1(predicted_values.astype(float))
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            predicted_values = model.forecast(len(X_test.index))
            predicted_values[np.isfinite(predicted_values) is False] = 0
            if self.preprocess_decisions["arima_transformation_used"] == "log":
                predicted_values = np.expm1(predicted_values.astype(float))
        self.predicted_values[f"{algorithm}"] = {}
        self.predicted_values[f"{algorithm}"] = predicted_values
        del model
        _ = gc.collect()
