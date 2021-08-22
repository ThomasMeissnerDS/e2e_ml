from e2eml.full_processing import postprocessing
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
import torch
from vowpalwabbit.sklearn_vw import VWClassifier, VWRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Exponential, Normal, LogNormal
from sklearn.linear_model import SGDRegressor, BayesianRidge, ARDRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
import warnings
import gc

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegressionModels(postprocessing.FullPipeline):
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
    and optimize accordingly. (Default: 'auto')
    :param logging_file_path: Preferred location to save the log file. Will otherwise stored in the current folder.
    :param low_memory_mode: Adds a preprocessing feature to reduce dataframe memory footprint. Will lead to a loss in
    model performance. Will be extended by further memory savings features in future releases.
    However we highly recommend GPU usage to heavily decrease model training times.
    """

    def linear_regression_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task='Train logistic regression model')
        algorithm = 'linear_regression'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = LinearRegression().fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def linear_regression_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Logistic regression')
        algorithm = 'linear_regression'
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(self.dataframe)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(X_test)

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_values[f"{algorithm}"] = {}
        self.predicted_values[f"{algorithm}"] = predicted_probs
        del model
        _ = gc.collect()

    def tabnet_regression_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task='Train Tabnet regression model')
        algorithm = 'tabnet'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            # load settings
            batch_size = self.tabnet_settings["batch_size"]
            virtual_batch_size = self.tabnet_settings["virtual_batch_size"]
            num_workers = self.tabnet_settings["num_workers"]
            max_epochs = self.tabnet_settings["max_epochs"]
            optimization_rounds = self.tabnet_settings["optimization_rounds"]

            def objective(trial):
                depths = trial.suggest_int('depths', 16, 64)
                factor = trial.suggest_uniform('factor', 0.1, 0.9)
                pretrain_difficulty = trial.suggest_uniform('pretrain_difficulty', 0.7, 0.9)
                mode = trial.suggest_categorical('mode', ["max", "min"])
                if optimization_rounds >= 20:
                    gamma = trial.suggest_loguniform('gamma', 1e-5, 2.0)
                    lambda_sparse = trial.suggest_loguniform('lambda_sparse', 1e-6, 1e-3)
                    mask_type = trial.suggest_categorical('mask_type', ["sparsemax", "entmax"])
                    n_shared = trial.suggest_int('n_shared', 1, 5)
                    n_independent = trial.suggest_int('n_independent', 1, 5)
                else:
                    gamma = 1.3
                    lambda_sparse = 1e-3
                    mask_type = "entmax"
                    n_shared = 2
                    n_independent = 2
                #loss_func = trial.suggest_categorical('loss_func', ['rmsle', 'mae', 'rmse', 'mse'])
                # ['auc', 'accuracy', 'balanced_accuracy', 'logloss', 'mae', 'mse', 'rmsle', 'unsup_loss', 'rmse']"

                param = dict(
                    gamma=gamma,
                    lambda_sparse=lambda_sparse,
                    n_d=depths,
                    n_a=depths,
                    n_shared=n_shared,
                    n_independent=n_independent,
                    n_steps=trial.suggest_int('n_steps', 1, 5),
                    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                    mask_type=mask_type,
                    scheduler_params=dict(
                        mode=mode, patience=10, min_lr=1e-5, factor=factor),
                    scheduler_fn=ReduceLROnPlateau,
                    seed=42,
                    verbose=1
                )
                mean_abs_errors = []
                skf = KFold(n_splits=5, random_state=42, shuffle=True)

                for train_index, test_index in skf.split(X_train):
                    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
                    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
                    # numpy conversion
                    y_train = y_train.values.reshape(-1, 1)
                    y_test = y_test.values.reshape(-1, 1)
                    x_train = x_train.to_numpy()
                    x_test = x_test.to_numpy()

                    Y_train_num = Y_train.values.reshape(-1, 1)
                    Y_test_num = Y_test.values.reshape(-1, 1)
                    X_train_num = X_train.to_numpy()
                    X_test_num = X_test.to_numpy()

                    pretrainer = TabNetPretrainer(**param)
                    pretrainer.fit(x_train,
                                   eval_set=[(x_test)],
                                   max_epochs=max_epochs,
                                   patience=20,
                                   batch_size=batch_size,
                                   virtual_batch_size=virtual_batch_size,
                                   num_workers=num_workers,
                                   drop_last=True,
                                   pretraining_ratio=pretrain_difficulty)

                    model = TabNetRegressor(**param)
                    model.fit(
                        x_train, y_train,
                        eval_set=[(x_test, y_test)],
                        eval_metric=['mae'],
                        patience=20,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        max_epochs=max_epochs,
                        drop_last=True,
                        from_unsupervised=pretrainer
                    )
                    preds = model.predict(X_test_num)

                    mae = mean_absolute_error(Y_test_num, preds)
                    mean_abs_errors.append(mae)
                cv_mae = np.mean(mean_abs_errors)
                return cv_mae

            study = optuna.create_study(direction='minimize')

            logging.info(f'Start Tabnet validation.')

            study.optimize(objective, n_trials=optimization_rounds, timeout=60*60*10)
            self.optuna_studies[f"{algorithm}"] = {}
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)  # .show()
            optuna.visualization.plot_param_importances(study)  # .show()
            self.optuna_studies[
                f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
            self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(
                study)

            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            Y_train = Y_train.values.reshape(-1, 1)
            Y_test = Y_test.values.reshape(-1, 1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            tabnet_best_param = study.best_trial.params
            param = dict(
                gamma=tabnet_best_param['gamma'],
                lambda_sparse=tabnet_best_param['lambda_sparse'],
                n_d=tabnet_best_param['depths'],
                n_a=tabnet_best_param['depths'],
                n_shared=tabnet_best_param["n_shared"],
                n_independent=tabnet_best_param["n_independent"],
                n_steps=tabnet_best_param['n_steps'],
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type=tabnet_best_param["mask_type"],
                scheduler_params=dict(
                    mode=tabnet_best_param["mode"], patience=5, min_lr=1e-5, factor=tabnet_best_param["factor"]),
                scheduler_fn=ReduceLROnPlateau,
                seed=42,
                verbose=1
            )
            pretrainer = TabNetPretrainer(**param)
            pretrainer.fit(X_train,
                           eval_set=[(X_test)],
                           max_epochs=max_epochs,
                           patience=20,
                           batch_size=batch_size,
                           virtual_batch_size=virtual_batch_size,
                           num_workers=num_workers,
                           drop_last=True,
                           pretraining_ratio=tabnet_best_param["pretrain_difficulty"])

            model = TabNetRegressor(**param)

            model.fit(
                X_train, Y_train,
                eval_set=[(X_test, Y_test)],
                eval_metric=['mae'],
                patience=50,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                max_epochs=max_epochs,
                drop_last=True,
                from_unsupervised=pretrainer
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
        self.get_current_timestamp(task='Predict with Tabnet regression')
        algorithm = 'tabnet'
        if self.prediction_mode:
            self.reset_test_train_index()
            model = self.trained_models[f"{algorithm}"]
            print("Nb pred cols")
            print(len(self.dataframe.columns))
            print(self.dataframe.info())
            predicted_probs = model.predict(self.dataframe.to_numpy())
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            Y_train = Y_train.values.reshape(-1, 1)
            Y_test = Y_test.values.reshape(-1, 1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(X_test)

        self.predicted_values[f"{algorithm}"] = {}
        self.predicted_values[f"{algorithm}"] = predicted_probs
        del model
        _ = gc.collect()

    def vowpal_wabbit_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        # https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/reference/vowpalwabbit.sklearn.html
        self.get_current_timestamp(task='Train Vowpal Wabbit model')
        algorithm = 'vowpal_wabbit'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = VWRegressor(convert_labels=False)
            model.fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def vowpal_wabbit_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Vowpal Wabbit')
        algorithm = 'vowpal_wabbit'
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(self.dataframe)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(X_test)

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_values[f"{algorithm}"] = {}
        self.predicted_values[f"{algorithm}"] = predicted_probs
        del model
        _ = gc.collect()

    def xg_boost_train(self, param=None, autotune=True, tune_mode='accurate'):
        """
        Trains an XGboost model by the given parameters.
        :param param: Takes a dictionary with custom parameter settings. Might be deprecated in future versions.
        :param steps: Integer higher than 0. Defines maximum training steps, iuf not in autotune mode.
        :param autotune: Set "True" for automatic hyperparameter optimization. (Default: true)
        :param tune_mode: 'Simple' for simple 80-20 split validation. 'Accurate': Each hyperparameter set will be validated
        with 10-fold cross validation. Longer runtimes, but higher performance. (Default: 'Accurate')
        """
        self.get_current_timestamp(task='Train Xgboost')
        self.check_gpu_support(algorithm='xgboost')
        if self.preferred_training_mode == 'auto':
            train_on = self.preprocess_decisions[f"gpu_support"]["xgboost"]
        elif self.preferred_training_mode == 'gpu':
            train_on = 'gpu_hist'
        else:
            train_on = 'exact'
        if self.prediction_mode:
            pass
        else:
            if autotune:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                D_train = xgb.DMatrix(X_train, label=Y_train)
                D_test = xgb.DMatrix(X_test, label=Y_test)

                def objective(trial):
                    param = {
                        'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'gamma-nloglik',
                        'verbose': 0,
                        'tree_method': train_on,  # use GPU for training
                        'max_depth': trial.suggest_int('max_depth', 2, 10),
                        # maximum depth of the decision trees being trained
                        'alpha': trial.suggest_loguniform('alpha', 1, 1e6),
                        'lambda': trial.suggest_loguniform('lambda', 1, 1e6),
                        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),
                        'steps': trial.suggest_int('steps', 2, 70000),
                        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5)
                    }
                    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-gamma-nloglik")
                    if tune_mode == 'simple':
                        eval_set = [(D_train, 'train'), (D_test, 'test')]
                        model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                          evals=eval_set, callbacks=[pruning_callback])
                        preds = model.predict(D_test)
                        mae = mean_absolute_error(Y_test, preds)
                        return mae
                    else:
                        result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'],
                                        early_stopping_rounds=10,
                                        as_pandas=True, seed=42, callbacks=[pruning_callback], nfold=5)
                        return result['test-gamma-nloglik-mean'].mean()

                algorithm = 'xgboost'
                if tune_mode == 'simple':
                    study = optuna.create_study(direction='minimize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=30)
                self.optuna_studies[f"{algorithm}"] = {}
                # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                optuna.visualization.plot_optimization_history(study)
                optuna.visualization.plot_param_importances(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(
                    study)
                self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(
                    study)
                lgbm_best_param = study.best_trial.params
                param = {
                    'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                    'eval_metric': 'gamma-nloglik',
                    'verbose': 0,
                    'tree_method': train_on,  # use GPU for training
                    'max_depth': lgbm_best_param["max_depth"],  # maximum depth of the decision trees being trained
                    'alpha': lgbm_best_param["alpha"],
                    'lambda': lgbm_best_param["lambda"],
                    'num_leaves': lgbm_best_param["num_leaves"],
                    'subsample': lgbm_best_param["subsample"],
                    'min_child_samples': lgbm_best_param["min_child_samples"],
                    'eta': lgbm_best_param["eta"],
                    'steps': lgbm_best_param["steps"],
                    'num_parallel_tree': lgbm_best_param["num_parallel_tree"]
                }
                eval_set = [(D_train, 'train'), (D_test, 'test')]
                model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                  evals=eval_set)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                D_train = xgb.DMatrix(X_train, label=Y_train)
                D_test = xgb.DMatrix(X_test, label=Y_test)
                algorithm = 'xgboost'
                if not param:
                    param = {
                        'eta': 0.001,  # learning rate,
                        # 'gamma': 5, #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                        'verbosity': 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
                        'alpha': 10,
                        # L1 regularization term on weights. Increasing this value will make model more conservative. (default = 0)
                        'lambda': 15,
                        # L2 regularization term on weights. Increasing this value will make model more conservative. (default = 1)
                        'subsample': 0.8,
                        'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'gamma-nloglik',
                        # 'colsample_bytree': 0.3,
                        'max_depth': 2,  # maximum depth of the decision trees being trained
                        'tree_method': 'gpu_hist',  # use GPU for training
                        'steps': 50000
                    }  # the number of classes in the dataset
                else:
                    param = param

                eval_set = [(D_train, 'train'), (D_test, 'test')]
                model = xgb.train(param, D_train, num_boost_round=50000, early_stopping_rounds=10,
                                  evals=eval_set)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

    def xgboost_predict(self, feat_importance=True, importance_alg='auto'):
        """
        Predicts on test & also new data given the prediction_mode is activated in the class.
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task='Predict with Xgboost')
        algorithm = 'xgboost'
        if self.prediction_mode:
            D_test = xgb.DMatrix(self.dataframe)
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(D_test)
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted_probs
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            D_test = xgb.DMatrix(X_test, label=Y_test)
            try:
                D_test_sample = xgb.DMatrix(X_test.sample(10000, random_state=42), label=Y_test)
            except:
                D_test_sample = xgb.DMatrix(X_test, label=Y_test)
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(D_test)
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted_probs

            if feat_importance and importance_alg == 'auto':
                if self.preprocess_decisions[f"gpu_support"]["xgboost"] == 'gpu_hist':
                    self.shap_explanations(model=model, test_df=D_test_sample, cols=X_test.columns)
                else:
                    xgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
            elif feat_importance and importance_alg == 'SHAP':
                self.shap_explanations(model=model, test_df=D_test_sample, cols=X_test.columns)
            elif feat_importance and importance_alg == 'inbuilt':
                xgb.plot_importance(model)
                plt.figure(figsize=(16, 12))
                plt.show()
            else:
                pass
            del model
            _ = gc.collect()

    def lgbm_train(self, tune_mode='accurate', gpu_use_dp=True):
        self.get_current_timestamp()
        self.check_gpu_support(algorithm='lgbm')
        if self.preferred_training_mode == 'auto':
            train_on = self.preprocess_decisions[f"gpu_support"]["lgbm"]
        elif self.preferred_training_mode == 'gpu':
            train_on = 'gpu'
            gpu_use_dp = True
        else:
            train_on = 'cpu'
            gpu_use_dp = False

        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            def objective(trial):
                dtrain = lgb.Dataset(X_train, label=Y_train)
                param = {
                    # TODO: Move to additional folder with pyfile "constants" (use OS absolute path)
                    'objective': 'regression',
                    'metric': 'gamma',
                    'num_boost_round': trial.suggest_int('num_boost_round', 100, 70000),
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 1e6),
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 1e6),
                    'linear_lambda': trial.suggest_loguniform('linear_lambda', 1, 1e6),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 512),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 1.0),
                    'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0, 1),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                    'verbose': -1,
                    'device': train_on,
                    'gpu_use_dp': gpu_use_dp
                }
                if tune_mode == 'simple':
                    gbm = lgb.train(param, dtrain, verbose_eval=False)
                    preds = gbm.predict(X_test)
                    mae = mean_absolute_error(Y_test, preds)
                    return mae
                else:
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "gamma")
                    result = lgb.cv(param, train_set=dtrain, nfold=10, num_boost_round=param['num_boost_round'],
                                    stratified=False, callbacks=[pruning_callback],
                                    early_stopping_rounds=10, seed=42,
                                    verbose_eval=False)
                    avg_result = result['gamma-mean'][-1]
                    return avg_result

            algorithm = 'lgbm'
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            self.optuna_studies[f"{algorithm}"] = {}
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)
            optuna.visualization.plot_param_importances(study)
            self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(
                study)
            self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
            lgbm_best_param = study.best_trial.params
            param = {
                'objective': 'regression',
                'metric': 'gamma',
                'num_boost_round': lgbm_best_param["num_boost_round"],
                'lambda_l1': lgbm_best_param["lambda_l1"],
                'lambda_l2': lgbm_best_param["lambda_l2"],
                'linear_lambda': lgbm_best_param["linear_lambda"],
                'num_leaves': lgbm_best_param["num_leaves"],
                'feature_fraction': lgbm_best_param["feature_fraction"],
                'bagging_fraction': lgbm_best_param["bagging_fraction"],
                'min_child_samples': lgbm_best_param["min_child_samples"],
                'min_gain_to_split': lgbm_best_param["min_gain_to_split"],
                'learning_rate': lgbm_best_param["learning_rate"],
                'verbose': -1,
                'device': train_on,
                'gpu_use_dp': gpu_use_dp
            }
            dtrain = lgb.Dataset(X_train, label=Y_train)
            dtest = lgb.Dataset(X_test, label=Y_test)
            model = lgb.train(param, dtrain, valid_sets=[dtrain, dtest], valid_names=['train', 'valid'],
                              early_stopping_rounds=10)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def lgbm_predict(self, feat_importance=True, importance_alg='auto'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated based on SHAP values.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with LBGM')
        algorithm = 'lgbm'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted_probs = model.predict(X_test)
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted_probs
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted_probs = model.predict(X_test)

            if feat_importance and importance_alg == 'auto':
                if self.preprocess_decisions[f"gpu_support"]["lgbm"] == 'gpu':
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
                else:
                    lgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
            elif feat_importance and importance_alg == 'SHAP':
                self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'inbuilt':
                lgb.plot_importance(model)
                plt.figure(figsize=(16, 12))
                plt.show()
            else:
                pass
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted_probs
        del model
        _ = gc.collect()
        return self.predicted_probs

    def sklearn_ensemble_train(self):
        """
        Trains an sklearn stacking regressor ensemble. Will automatically test different stacked regressor combinations.
        Expect very long runtimes due to CPU usage.
        """
        self.get_current_timestamp(task='Train sklearn ensemble')
        algorithm = 'sklearn_ensemble'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            def objective(trial):
                ensemble_variation = trial.suggest_categorical("ensemble_variant", ["2_boosters",
                                                                                    "no_boost_forest",
                                                                                    "reversed_boosters",
                                                                                    "full_ensemble"])
                # Step 2. Setup values for the hyperparameters:
                if ensemble_variation == '2_boosters':
                    level0 = list()
                    level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                    level1 = GradientBoostingRegressor(n_estimators=5000)
                    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'no_boost_forest':
                    level0 = list()
                    level0.append(('sgd', SGDRegressor()))
                    level0.append(('svr', LinearSVR()))
                    level0.append(('ard', ARDRegression()))
                    level0.append(('ridge', RidgeCV()))
                    level1 = GradientBoostingRegressor()
                    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'reversed_boosters':
                    level0 = list()
                    level0.append(('xgb', GradientBoostingRegressor(n_estimators=5000)))
                    level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                    level1 = LinearRegression(n_jobs=-2)
                    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'full_ensemble':
                    level0 = list()
                    level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                    level0.append(('lr', LinearRegression(n_jobs=-2)))
                    level0.append(('gdc', GradientBoostingRegressor(n_estimators=5000)))
                    level0.append(('sgd', SGDRegressor()))
                    level0.append(('svr', LinearSVR()))
                    level0.append(('ard', ARDRegression()))
                    level0.append(('ridge', RidgeCV()))
                    level0.append(('qda', BayesianRidge()))
                    level0.append(('rdf', RandomForestRegressor(max_depth=5, n_jobs=-2)))
                    # define meta learner model
                    level1 = GradientBoostingRegressor(n_estimators=5000)
                    # define the stacking ensemble
                    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)

                # Step 3: Scoring method:
                model.fit(X_train, Y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(Y_test, preds)
                return mae

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            best_variant = study.best_trial.params["ensemble_variant"]
            if best_variant == '2_boosters':
                level0 = list()
                level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                level1 = GradientBoostingRegressor(n_estimators=5000)
                model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
            elif best_variant == 'no_boost_forest':
                level0 = list()
                level0.append(('sgd', SGDRegressor()))
                level0.append(('svr', LinearSVR()))
                level0.append(('ard', ARDRegression()))
                level0.append(('ridge', RidgeCV()))
                level1 = GradientBoostingRegressor()
                model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
            elif best_variant == 'reversed_boosters':
                level0 = list()
                level0.append(('xgb', GradientBoostingRegressor(n_estimators=5000)))
                level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                level1 = LinearRegression(n_jobs=-2)
                model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
            elif best_variant == 'full_ensemble':
                level0 = list()
                level0.append(('lgbm', LGBMRegressor(n_estimators=5000)))
                level0.append(('lr', LinearRegression(n_jobs=-2)))
                level0.append(('gdc', GradientBoostingRegressor(n_estimators=5000)))
                level0.append(('sgd', SGDRegressor()))
                level0.append(('svr', LinearSVR()))
                level0.append(('ard', ARDRegression()))
                level0.append(('ridge', RidgeCV()))
                level0.append(('qda', BayesianRidge()))
                level0.append(('rdf', RandomForestRegressor(max_depth=5, n_jobs=-2)))
                # define meta learner model
                level1 = GradientBoostingRegressor(n_estimators=5000)
                # define the stacking ensemble
                model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
            model.fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def sklearn_ensemble_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with sklearn ensemble')
        algorithm = 'sklearn_ensemble'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted = model.predict(X_test)
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted = model.predict(X_test)

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted
        del model
        _ = gc.collect()
        return self.predicted_probs

    def ngboost_train(self, tune_mode='accurate'):
        """
        Trains an Ngboost regressor.
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task='Train Ngboost')
        algorithm = 'ngboost'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            def objective(trial):
                base_learner_choice = trial.suggest_categorical("base_learner", ["DecTree_depth2",
                                                                                 "DecTree_depth5",
                                                                                 "DecTree_depthNone",
                                                                                 "GradientBoost_depth2",
                                                                                 "GradientBoost_depth5"])

                if base_learner_choice == "DecTree_depth2":
                    base_learner_choice = DecisionTreeRegressor(max_depth=2)
                elif base_learner_choice == "DecTree_depth5":
                    base_learner_choice = DecisionTreeRegressor(max_depth=5)
                elif base_learner_choice == "DecTree_depthNone":
                    base_learner_choice = DecisionTreeRegressor(max_depth=None)
                elif base_learner_choice == "GradientBoost_depth2":
                    base_learner_choice = GradientBoostingRegressor(max_depth=2,
                                                                    n_estimators=1000,
                                                                    n_iter_no_change=10,
                                                                    random_state=42)
                elif base_learner_choice == "GradientBoost_depth5":
                    base_learner_choice = GradientBoostingRegressor(max_depth=5,
                                                                    n_estimators=10000,
                                                                    n_iter_no_change=10,
                                                                    random_state=42)

                dist_choice = trial.suggest_categorical('Dist', ["Normal", "LogNormal", "Exponential"])
                if dist_choice == "Normal":
                    dist_choice = Normal
                elif dist_choice == "LogNormal":
                    dist_choice = LogNormal
                elif dist_choice == "Exponential":
                    dist_choice = Exponential

                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 2, 50000),
                    'minibatch_frac': trial.suggest_uniform('minibatch_frac', 0.4, 1.0),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
                }
                if tune_mode == 'simple':
                    model = NGBRegressor(n_estimators=param["n_estimators"],
                                         minibatch_frac=param["minibatch_frac"],
                                         Dist=dist_choice,
                                         Base=base_learner_choice,
                                         learning_rate=param["learning_rate"]).fit(X_train, Y_train, X_val=X_test,
                                                                                   Y_val=Y_test,
                                                                                   early_stopping_rounds=10)
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(Y_test, preds)
                    return mae
                else:
                    model = NGBRegressor(n_estimators=param["n_estimators"],
                                         minibatch_frac=param["minibatch_frac"],
                                         Dist=dist_choice,
                                         Base=base_learner_choice,
                                         learning_rate=param["learning_rate"],
                                         random_state=42)
                    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
                                             fit_params={'X_val': X_test, 'Y_val': Y_test, 'early_stopping_rounds': 10})
                    mae = np.mean(scores)
                    return mae

            algorithm = 'ngboost'
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            self.optuna_studies[f"{algorithm}"] = {}
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)
            optuna.visualization.plot_param_importances(study)
            self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(
                study)
            self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
            lgbm_best_param = study.best_trial.params

            if lgbm_best_param["base_learner"] == "DecTree_depth2":
                base_learner_choice = DecisionTreeRegressor(max_depth=2)
            elif lgbm_best_param["base_learner"] == "DecTree_depth5":
                base_learner_choice = DecisionTreeRegressor(max_depth=5)
            elif lgbm_best_param["base_learner"] == "DecTree_depthNone":
                base_learner_choice = DecisionTreeRegressor(max_depth=None)
            elif lgbm_best_param["base_learner"] == "GradientBoost_depth2":
                base_learner_choice = GradientBoostingRegressor(max_depth=2,
                                                                n_estimators=1000,
                                                                n_iter_no_change=10,
                                                                random_state=42)
            elif lgbm_best_param["base_learner"] == "GradientBoost_depth5":
                base_learner_choice = GradientBoostingRegressor(max_depth=5,
                                                                n_estimators=10000,
                                                                n_iter_no_change=10,
                                                                random_state=42)

            if lgbm_best_param["Dist"] == "Normal":
                dist_choice = Normal
            elif lgbm_best_param["Dist"] == "LogNormal":
                dist_choice = LogNormal
            elif lgbm_best_param["Dist"] == "Exponential":
                dist_choice = Exponential

            param = {
                'Dist': lgbm_best_param["Dist"],
                'n_estimators': lgbm_best_param["n_estimators"],
                'minibatch_frac': lgbm_best_param["minibatch_frac"],
                'learning_rate': lgbm_best_param["learning_rate"],
                'random_state': 42
            }
            model = NGBRegressor(n_estimators=param["n_estimators"],
                                 minibatch_frac=param["minibatch_frac"],
                                 Dist=dist_choice,
                                 Base=base_learner_choice,
                                 learning_rate=param["learning_rate"]).fit(X_train, Y_train, X_val=X_test, Y_val=Y_test,
                                                                           early_stopping_rounds=10)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def ngboost_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Ngboost')
        algorithm = 'ngboost'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted = model.predict(X_test)
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted = model.predict(X_test)

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns, explainer="kernel")
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns, explainer="kernel")
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted
        del model
        _ = gc.collect()
        return self.predicted_values
