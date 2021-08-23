from e2eml.full_processing import postprocessing
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from ngboost import NGBClassifier
from vowpalwabbit.sklearn_vw import VWClassifier
from ngboost.distns import k_categorical
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
import logging
import gc

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class Matthews(Metric):
    def __init__(self):
        self._name = "matthews"
        self._maximize = True

    def __call__(self, y_true, y_score):
        try:
            predicted_probs = np.asarray([line[1] for line in y_score])
            predicted_classes = predicted_probs > 0.5
        except Exception:
            predicted_classes = np.asarray([np.argmax(line) for line in y_score])

        try:
            matthews = matthews_corrcoef(y_true, predicted_classes)
        except Exception:
            matthews = 0
        return matthews


class ClassificationModels(postprocessing.FullPipeline, Matthews):
    """
    This class stores all model training and prediction methods for classification tasks.
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

    def threshold_refiner(self, probs, targets, algorithm):
        """
        Loops through predicted class probabilities in binary contexts and measures performance with
        Matthew correlation.
        :param probs: Takes predicted class probabilities.
        :param targets: Takes actual targets.
        :return: Stores the best threshold as class attribute.
        """
        self.get_current_timestamp()
        if "probability_threshold" in self.preprocess_decisions:
            pass
        else:
            self.preprocess_decisions[f"probability_threshold"] = {}

        loop_spots = np.linspace(0, 1, 100, endpoint=False)
        max_matthew = 0
        best_threshold = 0
        for iteration in loop_spots:
            blended_pred = probs > iteration
            try:
                matthews = matthews_corrcoef(targets, blended_pred)
            except Exception:
                try:
                    partial_probs = np.asarray([line[1] for line in probs])
                    blended_pred = partial_probs > iteration
                    matthews = matthews_corrcoef(targets, blended_pred)
                except Exception:
                    matthews = 0
            if matthews > max_matthew:
                max_matthew = matthews
                best_threshold = iteration
        self.preprocess_decisions[f"probability_threshold"][algorithm] = best_threshold
        return self.preprocess_decisions[f"probability_threshold"][algorithm]

    def logistic_regression_train(self):
        """
        Trains a simple Logistic regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task='Train logistic regression model')
        algorithm = 'logistic_regression'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = LogisticRegression(random_state=0).fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def logistic_regression_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Logistic Regression')
        algorithm = 'logistic_regression'
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test.astype(int), n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
            del model
            _ = gc.collect()

    def tabnet_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task='Train Tabnet classification model')
        algorithm = 'tabnet'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train[X_train.columns.to_list()] = X_train[X_train.columns.to_list()].astype(float)
            X_test[X_test.columns.to_list()] = X_test[X_test.columns.to_list()].astype(float)

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
                gamma = trial.suggest_loguniform('gamma', 1e-5, 2.0)
                lambda_sparse = trial.suggest_loguniform('lambda_sparse', 1e-6, 1e-3)
                mask_type = trial.suggest_categorical('mask_type', ["sparsemax", "entmax"])
                n_shared = trial.suggest_int('n_shared', 1, 5)
                n_independent = trial.suggest_int('n_independent', 1, 5)

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
                mean_matthew_corr = []
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

                for train_index, test_index in skf.split(X_train, Y_train):
                    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
                    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
                    # numpy conversion
                    y_train = y_train.values.reshape(-1)
                    y_test = y_test.values.reshape(-1)
                    x_train = x_train.to_numpy()
                    x_test = x_test.to_numpy()

                    Y_train_num = Y_train.values.reshape(-1)
                    Y_test_num = Y_test.values.reshape(-1)
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

                    model = TabNetClassifier(**param)
                    model.fit(
                        x_train, y_train,
                        eval_set=[(x_test, y_test)],
                        eval_metric=[Matthews],
                        patience=25,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        max_epochs=max_epochs,
                        drop_last=True,
                        from_unsupervised=pretrainer
                    )
                    partial_probs = model.predict_proba(X_test_num)
                    if self.class_problem == 'binary':
                        predicted_probs = np.asarray([line[1] for line in partial_probs])
                        self.threshold_refiner(predicted_probs, Y_test_num, algorithm)
                        predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
                    else:
                        predicted_probs = partial_probs
                        predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
                    matthew = matthews_corrcoef(Y_test_num, predicted_classes)
                    mean_matthew_corr.append(matthew)
                cv_matthew = np.mean(mean_matthew_corr)
                return cv_matthew

            study = optuna.create_study(direction='maximize')
            logging.info(f'Start Tabnet validation.')

            study.optimize(objective, n_trials=optimization_rounds, timeout=60*60*15)
            self.optuna_studies[f"{algorithm}"] = {}
            # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)  # .show()
            optuna.visualization.plot_param_importances(study)  # .show()
            self.optuna_studies[
                f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
            self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(
                study)

            Y_train = Y_train.values.reshape(-1)
            Y_test = Y_test.values.reshape(-1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            tabnet_best_param = study.best_trial.params
            param = dict(
               gamma=tabnet_best_param['gamma'],
               lambda_sparse=tabnet_best_param['lambda_sparse'],
               n_d=tabnet_best_param['depths'],
               n_a=tabnet_best_param['depths'],
               n_steps=tabnet_best_param['n_steps'],
               n_shared=tabnet_best_param["n_shared"],
               n_independent=tabnet_best_param["n_independent"],
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
                           patience=25,
                           batch_size=batch_size,
                           virtual_batch_size=virtual_batch_size,
                           num_workers=num_workers,
                           drop_last=True,
                           pretraining_ratio=tabnet_best_param["pretrain_difficulty"])

            model = TabNetClassifier(**param)
            model.fit(
                X_train, Y_train,
                eval_set=[(X_test, Y_test)],
                eval_metric=[Matthews],
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

    def tabnet_predict(self):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Tabnet classification')
        algorithm = 'tabnet'
        if self.prediction_mode:
            self.dataframe[self.dataframe.columns.to_list()] = self.dataframe[self.dataframe.columns.to_list()].astype(float)
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe.to_numpy())
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            Y_train = Y_train.values.reshape(-1)
            Y_test = Y_test.values.reshape(-1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])

        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def vowpal_wabbit_train(self):
        """
        Trains a simple Logistic regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task='Train Vowpal Wabbit model')
        algorithm = 'vowpal_wabbit'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = VWClassifier().fit(X_train, Y_train)
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
            partial_probs = model.predict_proba(self.dataframe)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test.astype(int), n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
            del model
            _ = gc.collect()

    def xg_boost_train(self, param=None, steps=None, autotune=True, tune_mode='accurate'):
        """
        Trains an XGboost model by the given parameters.
        :param param: Takes a dictionary with custom parameter settings.
        :param steps: Integer higher than 0. Defines maximum training steps, iuf not in autotune mode.
        :param autotune: Set "True" for automatic hyperparameter optimization. (Default: true)
        :param tune_mode: 'Simple' for simple 80-20 split validation. 'Accurate': Each hyperparameter set will be validated
        with 5-fold crossvalidation. Longer runtimes, but higher performance. (Default: 'Accurate')
        """
        self.get_current_timestamp(task='Train Xgboost')
        self.check_gpu_support(algorithm='xgboost')
        if self.preferred_training_mode == 'auto':
            train_on = self.preprocess_decisions[f"gpu_support"]["xgboost"]
        elif self.preferred_training_mode == 'gpu':
            train_on = 'gpu_hist'
            logging.info(f'Start Xgboost model training on {self.preferred_training_mode}.')
        else:
            train_on = 'exact'
            logging.info(f'Start Xgboost model training on {self.preferred_training_mode}.')
        if self.prediction_mode:
            pass
        else:
            if autotune:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=Y_train
                )
                D_train = xgb.DMatrix(X_train, label=Y_train, weight=classes_weights)
                D_test = xgb.DMatrix(X_test, label=Y_test)

                if self.class_problem == 'binary':
                    def objective(trial):
                        param = {
                            'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                            'eval_metric': 'mlogloss',
                            'verbose': 0,
                            'tree_method': train_on,  # use GPU for training
                            'num_class': Y_train.nunique(),
                            'max_depth': trial.suggest_int('max_depth', 2, 10),
                            # maximum depth of the decision trees being trained
                            'alpha': trial.suggest_loguniform('alpha', 1, 1e6),
                            'lambda': trial.suggest_loguniform('lambda', 1, 1e6),
                            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                            'min_child_samples': trial.suggest_int('min_child_samples', 5, 1000),
                            'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),
                            'steps': trial.suggest_int('steps', 2, 70000),
                            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5)
                        }
                        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mlogloss")
                        if tune_mode == 'simple':
                            eval_set = [(D_train, 'train'), (D_test, 'test')]
                            model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                              evals=eval_set, callbacks=[pruning_callback])
                            preds = model.predict(D_test)
                            pred_labels = np.asarray([np.argmax(line) for line in preds])
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'],
                                            early_stopping_rounds=10, nfold=10,
                                            as_pandas=True, seed=42, callbacks=[pruning_callback])
                            # avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result['test-mlogloss-mean'].mean()

                    algorithm = 'xgboost'
                    if tune_mode == 'simple':
                        study = optuna.create_study(direction='maximize')
                        logging.info(f'Start Xgboost simple validation.')
                    else:
                        study = optuna.create_study(direction='minimize')
                        logging.info(f'Start Xgboost advanced validation.')
                    study.optimize(objective, n_trials=40)
                    self.optuna_studies[f"{algorithm}"] = {}
                    # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                    # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                    optuna.visualization.plot_optimization_history(study)  # .show()
                    optuna.visualization.plot_param_importances(study)  # .show()
                    self.optuna_studies[
                        f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                    self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(
                        study)

                    lgbm_best_param = study.best_trial.params
                    param = {
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'mlogloss',
                        'verbose': 0,
                        'tree_method': train_on,  # use GPU for training
                        'num_class': Y_train.nunique(),
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
                    logging.info(f'Start Xgboost final model training with optimized hyperparamers.')
                    model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                      evals=eval_set)
                    self.trained_models[f"{algorithm}"] = {}
                    self.trained_models[f"{algorithm}"] = model
                    return self.trained_models

                else:
                    def objective(trial):
                        param = {
                            'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                            'eval_metric': 'mlogloss',
                            'verbose': 0,
                            'tree_method': train_on,  # use GPU for training
                            'num_class': Y_train.nunique(),
                            'max_depth': trial.suggest_int('max_depth', 2, 10),
                            # maximum depth of the decision trees being trained
                            'alpha': trial.suggest_loguniform('alpha', 1, 1e6),
                            'lambda': trial.suggest_loguniform('lambda', 1, 1e6),
                            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                            'min_child_samples': trial.suggest_int('min_child_samples', 5, 1000),
                            'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),  # 0.001
                            'steps': trial.suggest_int('steps', 2, 70000),
                            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5)
                        }
                        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mlogloss")
                        if tune_mode == 'simple':
                            eval_set = [(D_train, 'train'), (D_test, 'test')]
                            model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                              evals=eval_set, callbacks=[pruning_callback])
                            preds = model.predict(D_test)
                            pred_labels = np.asarray([np.argmax(line) for line in preds])
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'],
                                            early_stopping_rounds=10, nfold=10,
                                            as_pandas=True, seed=42, callbacks=[pruning_callback])
                            # avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result['test-mlogloss-mean'].mean()

                    algorithm = 'xgboost'
                    if tune_mode == 'simple':
                        logging.info(f'Start Xgboost simple validation.')
                        study = optuna.create_study(direction='maximize')
                    else:
                        logging.info(f'Start Xgboost advanced validation.')
                        study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=30)
                    self.optuna_studies[f"{algorithm}"] = {}
                    # optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                    # optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                    optuna.visualization.plot_optimization_history(study)
                    optuna.visualization.plot_param_importances(study)
                    self.optuna_studies[
                        f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                    self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(
                        study)

                    lgbm_best_param = study.best_trial.params
                    param = {
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'mlogloss',
                        'verbose': 0,
                        'tree_method': train_on,  # use GPU for training
                        'num_class': Y_train.nunique(),
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
                    logging.info(f'Start Xgboost final model training with optimized hyperparamers.')
                    model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                      evals=eval_set)
                    self.trained_models[f"{algorithm}"] = {}
                    self.trained_models[f"{algorithm}"] = model
                    del model
                    _ = gc.collect()
                    return self.trained_models
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=Y_train)
                D_train = xgb.DMatrix(X_train, label=Y_train, weight=classes_weights)
                D_test = xgb.DMatrix(X_test, label=Y_test)
                algorithm = 'xgboost'
                if not param:
                    param = {
                        'eta': 0.001,  # learning rate,
                        'scale_pos_weight': 1,
                        # A typical value to consider: sum(negative instances) / sum(positive instances) (default = 1)
                        # 'gamma': 5, #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                        'verbosity': 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
                        'alpha': 1,
                        # L1 regularization term on weights. Increasing this value will make model more conservative. (default = 0)
                        'lambda': 1,
                        # L2 regularization term on weights. Increasing this value will make model more conservative. (default = 1)
                        'subsample': 0.8,
                        'eval_metric': "mlogloss",  # 'mlogloss','auc','rmsle'
                        # 'colsample_bytree': 0.3,
                        'max_depth': 2,  # maximum depth of the decision trees being trained
                        'tree_method': train_on,  # use GPU for training
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'steps': 50000,
                        'num_class': self.num_classes}  # the number of classes in the dataset
                else:
                    param = param

                if not steps:
                    steps = 10000
                else:
                    steps = steps
                eval_set = [(D_train, 'train'), (D_test, 'test')]
                logging.info(f'Start Xgboost simple model training with predefined hyperparamers.')
                model = xgb.train(param, D_train, num_boost_round=50000, early_stopping_rounds=10,
                                  evals=eval_set)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

    def xgboost_predict(self, feat_importance=True, importance_alg='auto'):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated based on SHAP values.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Predict with Xgboost')
        algorithm = 'xgboost'
        if self.prediction_mode:
            D_test = xgb.DMatrix(self.dataframe)
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict(D_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            D_test = xgb.DMatrix(X_test, label=Y_test)
            try:
                D_test_sample = xgb.DMatrix(X_test.sample(10000, random_state=42), label=Y_test)
            except:
                D_test_sample = xgb.DMatrix(X_test, label=Y_test)
            model = self.trained_models[f"{algorithm}"]
            if self.class_problem == 'binary' or self.class_problem == 'multiclass':
                partial_probs = model.predict(D_test)
                if self.class_problem == 'binary':
                    predicted_probs = np.asarray([line[1] for line in partial_probs])
                    self.threshold_refiner(predicted_probs, Y_test, algorithm)
                    predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
                else:
                    predicted_probs = partial_probs
                    predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])

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
                self.predicted_probs[f"{algorithm}"] = {}
                self.predicted_classes[f"{algorithm}"] = {}
                self.predicted_probs[f"{algorithm}"] = predicted_probs
                self.predicted_classes[f"{algorithm}"] = predicted_classes
                del model
                _ = gc.collect()
                return self.predicted_probs
            elif self.xgboost_objective == 'regression':
                self.xg_boost_regression = model.predict(D_test)
                return self.xg_boost_regression

    def lgbm_train(self, tune_mode='accurate', gpu_use_dp=True):
        """
        Trains an LGBM model by the given parameters.
        :param tune_mode: 'Simple' for simple 80-20 split validation. 'Accurate': Each hyperparameter set will be validated
        with 10-fold cross validation. Longer runtimes, but higher performance. (Default: 'Accurate')
        :param gpu_use_dp: If True and when GPU accelerated, LGBM will use bigger floats for higher accuracy, but at the
        cost of longer runtimes (Default: True)
        """
        self.get_current_timestamp(task='Train LGBM')
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
            classes_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=Y_train)

            if self.class_problem == 'binary':
                weights_for_lgb = self.calc_scale_pos_weight()

                def objective(trial):
                    dtrain = lgb.Dataset(X_train, label=Y_train)
                    param = {
                        # TODO: Move to additional folder with pyfile "constants" (use OS absolute path)
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        # 'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1e3),
                        'num_boost_round': trial.suggest_int('num_boost_round', 100, 50000),
                        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 1e6),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 1e6),
                        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 0.1),
                        'verbose': -1,
                        'device': train_on,
                        'gpu_use_dp': gpu_use_dp
                    }

                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
                    if tune_mode == 'simple':
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.rint(preds)
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        result = lgb.cv(param, train_set=dtrain, nfold=10, num_boost_round=param['num_boost_round'],
                                        early_stopping_rounds=10, callbacks=[pruning_callback], seed=42,
                                        verbose_eval=False)
                        avg_result = np.mean(np.array(result["binary_logloss-mean"]))
                        return avg_result

                algorithm = 'lgbm'
                if tune_mode == 'simple':
                    study = optuna.create_study(direction='maximize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=40)
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
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    # 'scale_pos_weight': lgbm_best_param["scale_pos_weight"],
                    'num_boost_round': lgbm_best_param["num_boost_round"],
                    'lambda_l1': lgbm_best_param["lambda_l1"],
                    'lambda_l2': lgbm_best_param["lambda_l2"],
                    'num_leaves': lgbm_best_param["num_leaves"],
                    'feature_fraction': lgbm_best_param["feature_fraction"],
                    'bagging_freq': lgbm_best_param["bagging_freq"],
                    'min_child_samples': lgbm_best_param["min_child_samples"],
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

            else:
                nb_classes = self.num_classes

                def objective(trial):
                    dtrain = lgb.Dataset(X_train, label=Y_train, weight=classes_weights)
                    param = {
                        'objective': 'multiclass',
                        'metric': 'multi_logloss',
                        'num_boost_round': trial.suggest_int('num_boost_round', 100, 50000),
                        'num_class': nb_classes,
                        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 1e6),
                        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 1e6),
                        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'min_gain_to_split': trial.suggest_uniform('min_gain_to_split', 0, 1),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 0.1),
                        'verbose': -1,
                        'device': train_on,
                        'gpu_use_dp': gpu_use_dp
                    }

                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
                    if tune_mode == 'simple':
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.asarray([np.argmax(line) for line in preds])
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        try:
                            result = lgb.cv(param, train_set=dtrain, nfold=10, num_boost_round=param['num_boost_round'],
                                            early_stopping_rounds=10, callbacks=[pruning_callback], seed=42,
                                            verbose_eval=False)
                            # fobj=lgb_matth_score)
                            avg_result = np.mean(np.array(result["multi_logloss-mean"]))  # Planned: matthew-mean
                        except Exception:
                            avg_result = 100
                        return avg_result

                algorithm = 'lgbm'
                if tune_mode == 'simple':
                    study = optuna.create_study(direction='maximize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=40)
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
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'class_weight': classes_weights,
                    'num_boost_round': lgbm_best_param["num_boost_round"],
                    'num_class': nb_classes,
                    'lambda_l1': lgbm_best_param["lambda_l1"],
                    'lambda_l2': lgbm_best_param["lambda_l2"],
                    'num_leaves': lgbm_best_param["num_leaves"],
                    'feature_fraction': lgbm_best_param["feature_fraction"],
                    'bagging_freq': lgbm_best_param["bagging_freq"],
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
        self.get_current_timestamp(task='Predict with LGBM')
        algorithm = 'lgbm'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted_probs = model.predict(X_test)
            if self.class_problem == 'binary':
                partial_probs = predicted_probs
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted_probs = model.predict(X_test)
            if self.class_problem == 'binary':
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])

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

            if feat_importance:
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            else:
                pass
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()
        return self.predicted_probs

    def sklearn_ensemble_train(self):
        """
        Trains an sklearn stacking classifier ensemble. Will automatically test different stacked classifier combinations.
        Expect very long runtimes due to CPU usage.
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task='Train sklearn ensemble')
        algorithm = 'sklearn_ensemble'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            def objective(trial):
                ensemble_variation = trial.suggest_categorical("ensemble_variant", ["2_boosters",
                                                                                    "3_boosters",
                                                                                    "trees_forest",
                                                                                    "reversed_boosters",
                                                                                    "full_ensemble"])
                # Step 2. Setup values for the hyperparameters:
                if ensemble_variation == '2_boosters':
                    level0 = list()
                    level0.append(('lgbm', LGBMClassifier(n_estimators=5000)))
                    level1 = GradientBoostingClassifier(n_estimators=5000)
                    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == '3_boosters':
                    level0 = list()
                    level0.append(('lgbm', LGBMClassifier(n_estimators=5000)))
                    level0.append(('abc', AdaBoostClassifier(n_estimators=100)))
                    level1 = GradientBoostingClassifier(n_estimators=5000)
                    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'trees_forest':
                    level0 = list()
                    level0.append(('cart', DecisionTreeClassifier(max_depth=3)))
                    level0.append(('rdf', RandomForestClassifier(max_depth=3)))
                    level1 = GradientBoostingClassifier()
                    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'reversed_boosters':
                    level0 = list()
                    level0.append(('xgb', GradientBoostingClassifier(n_estimators=5000)))
                    level0.append(('lgbm', LGBMClassifier(n_estimators=5000)))
                    level1 = LogisticRegression(class_weight='balanced', max_iter=500)
                    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
                elif ensemble_variation == 'full_ensemble':
                    level0 = list()
                    level0.append(('lgbm', LGBMClassifier(n_estimators=5000)))
                    level0.append(('lr', LogisticRegressionCV(class_weight='balanced', max_iter=500,
                                                              penalty='elasticnet',
                                                              l1_ratios=[0.1, 0.5, 0.9],
                                                              solver='saga')))
                    level0.append(('gdc', GradientBoostingClassifier(n_estimators=5000)))
                    level0.append(('cart', DecisionTreeClassifier(max_depth=5)))
                    level0.append(('abc', AdaBoostClassifier(n_estimators=100)))
                    level0.append(('qda', QuadraticDiscriminantAnalysis()))
                    level0.append(('rdf', RandomForestClassifier(max_depth=5)))
                    # define meta learner model
                    level1 = GradientBoostingClassifier(n_estimators=5000)
                    # define the stacking ensemble
                    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)

                # Step 3: Scoring method:
                model.fit(X_train, Y_train)
                predicted_probs = model.predict_proba(X_test)
                if self.class_problem == 'binary':
                    self.threshold_refiner(predicted_probs, Y_test, algorithm)
                    partial_probs = np.asarray([line[1] for line in predicted_probs])
                    predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
                else:
                    predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
                matthews = matthews_corrcoef(Y_test, predicted_classes)
                return matthews

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)
            best_variant = study.best_trial.params["ensemble_variant"]
            if best_variant == '2_boosters':
                level0 = list()
                level0.append(('lgbm', LGBMClassifier()))
                level1 = GradientBoostingClassifier()
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            elif best_variant == '3_boosters':
                level0 = list()
                level0.append(('lgbm', LGBMClassifier()))
                level0.append(('abc', AdaBoostClassifier()))
                level1 = GradientBoostingClassifier()
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            elif best_variant == 'trees_forest':
                level0 = list()
                level0.append(('cart', DecisionTreeClassifier(max_depth=5)))
                level0.append(('rdf', RandomForestClassifier(max_depth=5)))
                level1 = GradientBoostingClassifier()
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            elif best_variant == 'reversed_boosters':
                level0 = list()
                level0.append(('xgb', GradientBoostingClassifier()))
                level0.append(('lgbm', LGBMClassifier()))
                level1 = LogisticRegression()
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            elif best_variant == 'full_ensemble':
                level0 = list()
                level0.append(('lgbm', LGBMClassifier()))
                level0.append(('lr', LogisticRegressionCV(class_weight='balanced', max_iter=500,
                                                          penalty='elasticnet',
                                                          l1_ratios=[0.1, 0.5, 0.9],
                                                          solver='saga')))
                level0.append(('gdc', GradientBoostingClassifier()))
                level0.append(('cart', DecisionTreeClassifier(max_depth=5)))
                level0.append(('abc', AdaBoostClassifier()))
                level0.append(('qda', QuadraticDiscriminantAnalysis()))
                level0.append(('rdf', RandomForestClassifier(max_depth=5)))
                # define meta learner model
                level1 = GradientBoostingClassifier()
                # define the stacking ensemble
                model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
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
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns, explainer='kernel')
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns, explainer='kernel')
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test.astype(int), n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
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
            classes_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=Y_train)
            nb_classes = k_categorical(Y_train.nunique())
            try:
                Y_train = Y_train.astype(int)
                Y_test = Y_test.astype(int)
            except Exception:
                Y_train = np.int(Y_train)
                Y_test = np.int(Y_test)

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

                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 2, 50000),
                    'minibatch_frac': trial.suggest_uniform('minibatch_frac', 0.4, 1.0),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
                }
                if tune_mode == 'simple':
                    model = NGBClassifier(n_estimators=param["n_estimators"],
                                          minibatch_frac=param["minibatch_frac"],
                                          Dist=nb_classes,
                                          Base=base_learner_choice,
                                          learning_rate=param["learning_rate"]).fit(X_train,
                                                                                    Y_train,
                                                                                    X_val=X_test,
                                                                                    Y_val=Y_test,
                                                                                    sample_weight=classes_weights,
                                                                                    early_stopping_rounds=10)
                    pred_labels = model.predict(X_test)
                    try:
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                    except Exception:
                        matthew = 0
                    return matthew
                else:
                    model = NGBClassifier(n_estimators=param["n_estimators"],
                                          minibatch_frac=param["minibatch_frac"],
                                          Dist=nb_classes,
                                          Base=base_learner_choice,
                                          learning_rate=param["learning_rate"],
                                          random_state=42)
                    try:
                        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='f1_weighted',
                                                 fit_params={'X_val': X_test,
                                                             'Y_val': Y_test,
                                                             'sample_weight': classes_weights,
                                                             'early_stopping_rounds': 10})
                        mae = np.mean(scores)
                    except Exception:
                        mae = 0
                    return mae

            algorithm = 'ngboost'
            if tune_mode == 'simple':
                study = optuna.create_study(direction='maximize')
            else:
                study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=15)
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
            param = {
                'Dist': nb_classes,
                'n_estimators': lgbm_best_param["n_estimators"],
                'minibatch_frac': lgbm_best_param["minibatch_frac"],
                'learning_rate': lgbm_best_param["learning_rate"]
            }
            model = NGBClassifier(n_estimators=param["n_estimators"],
                                  minibatch_frac=param["minibatch_frac"],
                                  Dist=nb_classes,
                                  Base=base_learner_choice,
                                  learning_rate=param["learning_rate"],
                                  random_state=42).fit(X_train,
                                                       Y_train,
                                                       X_val=X_test,
                                                       Y_val=Y_test,
                                                       sample_weight=classes_weights,
                                                       early_stopping_rounds=10)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def ngboost_predict(self, feat_importance=True, importance_alg='SHAP'):
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
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
                predicted_classes = predicted_classes.astype(int)
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray([np.argmax(line) for line in partial_probs])
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            partial_probs = model.predict_proba(X_test)
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            if self.class_problem == 'binary':
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"][algorithm]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])

            if feat_importance and importance_alg == 'SHAP':
                self.runtime_warnings(warn_about='shap_cpu')
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42),
                                           cols=X_test.columns, explainer="kernel")
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns, explainer="kernel")
            elif feat_importance and importance_alg == 'permutation':
                result = permutation_importance(
                    model, X_test, Y_test.astype(int), n_repeats=10, random_state=42, n_jobs=-1)
                permutation_importances = pd.Series(result.importances_mean, index=X_test.columns)
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = {}
        self.predicted_classes[f"{algorithm}"] = {}
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()
        return self.predicted_probs
