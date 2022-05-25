import gc
import logging
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from pandas.core.common import SettingWithCopyWarning
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


class RegressionForTimeSeriesModels(postprocessing.FullPipeline):
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

    def tabnet_regression_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Tabnet regression model")
        algorithm = "tabnet"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train_sample, y_train_sample = self.get_hyperparameter_tuning_sample_df()

            X_train[X_train.columns.to_list()] = X_train[
                X_train.columns.to_list()
            ].astype(float)
            X_test[X_test.columns.to_list()] = X_test[X_test.columns.to_list()].astype(
                float
            )
            x_train_sample[x_train_sample.columns.to_list()] = x_train_sample[
                x_train_sample.columns.to_list()
            ].astype(float)

            y_train_sample = y_train_sample.astype(int)

            if len(x_train_sample.index) < len(X_train.index):
                rec_batch_size = (len(x_train_sample.index) * 0.8) / 20
                if int(rec_batch_size) % 2 == 0:
                    rec_batch_size = int(rec_batch_size)
                else:
                    rec_batch_size = int(rec_batch_size) + 1
                if rec_batch_size > 16384:
                    rec_batch_size = 16384
                    virtual_batch_size = 4096
                else:
                    virtual_batch_size = int(rec_batch_size / 4)

                # update batch sizes in case hyperparameter tuning happens on samples
                self.tabnet_settings["batch_size"] = rec_batch_size
                self.tabnet_settings["virtual_batch_size"] = virtual_batch_size

            # load settings
            batch_size = self.tabnet_settings["batch_size"]
            virtual_batch_size = self.tabnet_settings["virtual_batch_size"]
            num_workers = self.tabnet_settings["num_workers"]

            def objective(trial):
                depths = trial.suggest_int("depths", 16, 64)
                factor = trial.suggest_uniform("factor", 0.1, 0.9)
                pretrain_difficulty = trial.suggest_uniform(
                    "pretrain_difficulty", 0.7, 0.9
                )
                mode = trial.suggest_categorical("mode", ["max", "min"])
                gamma = trial.suggest_loguniform("gamma", 1e-5, 2.0)
                lambda_sparse = trial.suggest_loguniform("lambda_sparse", 1e-6, 1e-3)
                mask_type = trial.suggest_categorical(
                    "mask_type", ["sparsemax", "entmax"]
                )
                n_shared = trial.suggest_int("n_shared", 1, 5)
                n_independent = trial.suggest_int("n_independent", 1, 5)
                pre_trainer_max_epochs = trial.suggest_int(
                    "pre_trainer_max_epochs", 10, 1000
                )
                trainer_max_epochs = trial.suggest_int("trainer_max_epochs", 10, 1000)
                # loss_func = trial.suggest_categorical('loss_func', ['rmsle', 'mae', 'rmse', 'mse'])
                # ['auc', 'accuracy', 'balanced_accuracy', 'logloss', 'mae', 'mse', 'rmsle', 'unsup_loss', 'rmse']"

                param = dict(
                    gamma=gamma,
                    lambda_sparse=lambda_sparse,
                    n_d=depths,
                    n_a=depths,
                    n_shared=n_shared,
                    n_independent=n_independent,
                    n_steps=trial.suggest_int("n_steps", 1, 5),
                    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                    mask_type=mask_type,
                    scheduler_params=dict(
                        mode=mode, patience=50, min_lr=1e-5, factor=factor
                    ),
                    scheduler_fn=ReduceLROnPlateau,
                    seed=self.global_random_state,
                    verbose=1,
                )
                mean_abs_errors = []
                skf = TimeSeriesSplit(n_splits=10)

                for train_index, test_index in skf.split(x_train_sample):
                    x_train, x_test = (
                        x_train_sample.iloc[train_index],
                        x_train_sample.iloc[test_index],
                    )
                    y_train, y_test = (
                        y_train_sample.iloc[train_index],
                        y_train_sample.iloc[test_index],
                    )
                    # numpy conversion
                    y_train = y_train.values.reshape(-1, 1)
                    y_test = y_test.values.reshape(-1, 1)
                    x_train = x_train.to_numpy()
                    x_test = x_test.to_numpy()

                    Y_train_num = y_train_sample.values.reshape(-1, 1)  # noqa: F841
                    Y_test_num = Y_test.values.reshape(-1, 1)
                    X_train_num = x_train_sample.to_numpy()  # noqa: F841
                    X_test_num = X_test.to_numpy()

                    pretrainer = TabNetPretrainer(**param)
                    pretrainer.fit(
                        x_train,
                        eval_set=[(x_test)],
                        max_epochs=pre_trainer_max_epochs,
                        patience=50,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        drop_last=True,
                        pretraining_ratio=pretrain_difficulty,
                    )

                    model = TabNetRegressor(**param)
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_test, y_test)],
                        eval_metric=["mae"],
                        patience=50,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        max_epochs=trainer_max_epochs,
                        drop_last=True,
                        from_unsupervised=pretrainer,
                    )
                    preds = model.predict(X_test_num)

                    mae = mean_absolute_error(Y_test_num, preds)
                    mean_abs_errors.append(mae)
                cv_mae = np.mean(mean_abs_errors)
                return cv_mae

            study = optuna.create_study(
                direction="minimize", study_name=f"{algorithm} tuning"
            )

            logging.info("Start Tabnet validation.")

            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds["tabnet"],
                timeout=self.hyperparameter_tuning_max_runtime_secs["tabnet"],
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

            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            Y_train = Y_train.values.reshape(-1, 1)
            Y_test = Y_test.values.reshape(-1, 1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            tabnet_best_param = study.best_trial.params
            param = dict(
                gamma=tabnet_best_param["gamma"],
                lambda_sparse=tabnet_best_param["lambda_sparse"],
                n_d=tabnet_best_param["depths"],
                n_a=tabnet_best_param["depths"],
                n_shared=tabnet_best_param["n_shared"],
                n_independent=tabnet_best_param["n_independent"],
                n_steps=tabnet_best_param["n_steps"],
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type=tabnet_best_param["mask_type"],
                scheduler_params=dict(
                    mode=tabnet_best_param["mode"],
                    patience=5,
                    min_lr=1e-5,
                    factor=tabnet_best_param["factor"],
                ),
                scheduler_fn=ReduceLROnPlateau,
                seed=self.global_random_state,
                verbose=1,
            )
            pretrainer = TabNetPretrainer(**param)
            pretrainer.fit(
                X_train,
                eval_set=[(X_test)],
                max_epochs=tabnet_best_param["pre_trainer_max_epochs"],
                patience=50,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                drop_last=True,
                pretraining_ratio=tabnet_best_param["pretrain_difficulty"],
            )

            model = TabNetRegressor(**param)

            model.fit(
                X_train,
                Y_train,
                eval_set=[(X_test, Y_test)],
                eval_metric=["mae"],
                patience=50,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                max_epochs=tabnet_best_param["trainer_max_epochs"],
                drop_last=True,
                from_unsupervised=pretrainer,
            )
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def tabnet_regression_predict(self):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Tabnet regression")
        algorithm = "tabnet"
        if self.prediction_mode:
            self.reset_test_train_index()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(self.dataframe.to_numpy(dtype="float32"))
            # self.dataframe[self.target_variable] = predicted_probs
            # self.scale_with_target(mode="reverse", drop_target=True)
            predicted_probs = self.target_skewness_handling(
                preds_to_reconvert=predicted_probs, mode="revert"
            )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(X_test.to_numpy(dtype="float32"))
            # X_test[self.target_variable] = predicted_probs
            # self.scale_with_target(mode="reverse", drop_target=True)
            predicted_probs = self.target_skewness_handling(
                preds_to_reconvert=predicted_probs, mode="revert"
            )

        self.predicted_values[f"{algorithm}"] = {}
        self.predicted_values[f"{algorithm}"] = predicted_probs
        del model
        _ = gc.collect()
