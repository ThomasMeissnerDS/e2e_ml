from e2eml.full_processing import postprocessing
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Exponential, Normal, LogNormal
from sklearn.linear_model import SGDRegressor, BayesianRidge, ARDRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class RegressionModels(postprocessing.FullPipeline):
    def linear_regression_train(self):
        self.get_current_timestamp(task='Train logistic regression model')
        algorithm = 'linear_regression'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = LinearRegression().fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            return self.trained_models

    def linear_regression_predict(self, feat_importance=True, importance_alg='permutation'):
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
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns)
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

    def xg_boost_train(self, param=None, steps=None, autotune=True, tune_mode='accurate'):
        """
        Trains an XGboost model by the given parameters.
        :param param: Takes a dictionary with custom parameter settings.
        :param steps: Integer higher than 0. Defines maximum training steps.
        :param objective: Will be deprecated.
        :param use_case: Chose 'binary' or 'regression'
        :return:
        """
        self.get_current_timestamp(task='Train Xgboost')
        if self.preferred_training_mode == 'gpu':
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
                        'eval_metric': 'mae',
                        'verbose': 0,
                        'tree_method': train_on, #use GPU for training
                        'max_depth': trial.suggest_int('max_depth', 2, 10),  #maximum depth of the decision trees being trained
                        'alpha': trial.suggest_loguniform('alpha', 1, 1e6),
                        'lambda': trial.suggest_loguniform('lambda', 1, 1e6),
                        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),
                        'steps': trial.suggest_int('steps', 2, 70000),
                        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5)
                    }
                    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mae")
                    if tune_mode == 'simple':
                        eval_set = [(D_train, 'train'), (D_test, 'test')]
                        model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                          evals=eval_set, callbacks=[pruning_callback])
                        preds = model.predict(D_test)
                        mae = mean_absolute_error(Y_test, preds)
                        return mae
                    else:
                        result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                        as_pandas=True, seed=42, callbacks=[pruning_callback], nfold=5)
                        return result['test-mae-mean'].mean()

                algorithm = 'xgboost'
                if tune_mode == 'simple':
                    study = optuna.create_study(direction='minimize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=30)
                self.optuna_studies[f"{algorithm}"] = {}
                #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                optuna.visualization.plot_optimization_history(study)
                optuna.visualization.plot_param_importances(study)
                self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
                lgbm_best_param = study.best_trial.params
                param = {
                    'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                    'eval_metric': 'mae',
                    'verbose': 0,
                    'tree_method': train_on, #use GPU for training
                    'max_depth': lgbm_best_param["max_depth"],  #maximum depth of the decision trees being trained
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
                return self.trained_models

            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                D_train = xgb.DMatrix(X_train, label=Y_train)
                D_test = xgb.DMatrix(X_test, label=Y_test)
                algorithm = 'xgboost'
                if not param:
                    param = {
                        'eta': 0.001, #learning rate,
                        #'gamma': 5, #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                        'verbosity': 0, #0 (silent), 1 (warning), 2 (info), 3 (debug)
                        'alpha' : 10, #L1 regularization term on weights. Increasing this value will make model more conservative. (default = 0)
                        'lambda': 15, #L2 regularization term on weights. Increasing this value will make model more conservative. (default = 1)
                        'subsample': 0.8,
                        'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'mae',
                        #'colsample_bytree': 0.3,
                        'max_depth': 2, #maximum depth of the decision trees being trained
                        'tree_method': 'gpu_hist', #use GPU for training
                        'steps': 50000
                        } #the number of classes in the dataset
                else:
                    param = param

                eval_set = [(D_train, 'train'), (D_test, 'test')]
                model = xgb.train(param, D_train, num_boost_round=50000, early_stopping_rounds=10,
                                  evals=eval_set)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                return self.trained_models

    def xgboost_predict(self, feat_importance=True):
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

            if feat_importance:
                self.shap_explanations(model=model, test_df=D_test_sample, cols=X_test.columns)
            else:
                pass

    def lgbm_train(self, tune_mode='accurate', gpu_use_dp=True):
        self.get_current_timestamp(task='Train with LGBM')
        if self.preferred_training_mode == 'gpu':
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
                    'metric': 'mean_absolute_error',
                    'num_boost_round': trial.suggest_int('num_boost_round', 100, 50000),
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1, 1e6),
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1, 1e6),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                    'verbose': -1,
                    'device': train_on,
                    'gpu_use_dp': gpu_use_dp
                }
                pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "mean_absolute_error")
                if tune_mode == 'simple':
                    gbm = lgb.train(param, dtrain, verbose_eval=False)
                    preds = gbm.predict(X_test)
                    mae = mean_absolute_error(Y_test, preds)
                    return mae
                else:
                    result = lgb.cv(param, train_set=dtrain, nfold=5, num_boost_round=param['num_boost_round'],
                                    early_stopping_rounds=10, callbacks=[pruning_callback], seed=42, verbose_eval=False)
                    avg_result = np.mean(np.array(result["mean_absolute_error-mean"]))
                    return avg_result
            algorithm = 'lgbm'
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            self.optuna_studies[f"{algorithm}"] = {}
            #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)
            optuna.visualization.plot_param_importances(study)
            self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
            self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
            lgbm_best_param = study.best_trial.params
            param = {
                'objective': 'regression',
                'metric': 'mean_absolute_error',
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
            return self.trained_models

    def lgbm_predict(self, feat_importance=True):
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

            if feat_importance:
                try:
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns)
                except Exception:
                    self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            else:
                pass
            self.predicted_values[f"{algorithm}"] = {}
            self.predicted_values[f"{algorithm}"] = predicted_probs
        return self.predicted_probs

    def sklearn_ensemble_train(self):
        """
        Trains an sklearn stacking classifier ensemble.
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
                    level0.append(('rdf',  RandomForestRegressor(max_depth=5, n_jobs=-2)))
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
                level0.append(('rdf',  RandomForestRegressor(max_depth=5, n_jobs=-2)))
                # define meta learner model
                level1 = GradientBoostingRegressor(n_estimators=5000)
                # define the stacking ensemble
                model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5, n_jobs=-2)
            model.fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            return self.trained_models

    def sklearn_ensemble_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Predicts on test & also new data given the prediction_mode is activated in the class.
        :return: Updates class attributes by its predictions.
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
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns)
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
            #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            optuna.visualization.plot_optimization_history(study)
            optuna.visualization.plot_param_importances(study)
            self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
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
            return self.trained_models

    def ngboost_predict(self, feat_importance=True, importance_alg='permutation'):
        """
        Predicts on test & also new data given the prediction_mode is activated in the class.
        :return: Updates class attributes by its predictions.
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
                    self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns, explainer="kernel")
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
        return self.predicted_values

