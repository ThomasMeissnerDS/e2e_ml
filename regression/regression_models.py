from full_processing import postprocessing
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class RegressionModels(postprocessing.FullPipeline):

    def xg_boost_train(self, param=None, steps=None, autotune=False, tune_mode='accurate'):
        """
        Trains an XGboost model by the given parameters.
        :param param: Takes a dictionary with custom parameter settings.
        :param steps: Integer higher than 0. Defines maximum training steps.
        :param objective: Will be deprecated.
        :param use_case: Chose 'binary' or 'regression'
        :return:
        """
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
                        'tree_method': 'gpu_hist', #use GPU for training
                        'max_depth': trial.suggest_int('max_depth', 2, 30),  #maximum depth of the decision trees being trained
                        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
                        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
                        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'eta': trial.suggest_loguniform('eta', 0.1, 0.3),
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
                                        as_pandas=True, seed=42, callbacks=[pruning_callback])
                        return result['test-mae-mean'].mean()

                algorithm = 'xgboost'
                if tune_mode == 'simple':
                    study = optuna.create_study(direction='maximize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=10)
                self.optuna_studies[f"{algorithm}"] = {}
                #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
                lgbm_best_param = study.best_trial.params
                param = {
                    'objective': 'reg:squarederror',  # OR  'binary:logistic' #the loss function being used
                    'eval_metric': 'mae',
                    'verbose': 0,
                    'tree_method': 'gpu_hist', #use GPU for training
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
                        'eval_metric' : "mlogloss", #'mlogloss','auc','rmsle'
                        #'colsample_bytree': 0.3,
                        'max_depth': 2, #maximum depth of the decision trees being trained
                        'tree_method': 'gpu_hist', #use GPU for training
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
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

    def lgbm_train(self, tune_mode='accurate', run_on='gpu', gpu_use_dp=True):
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
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 0.1),
                    'verbose': -1,
                    'device': run_on,
                    'gpu_use_dp': gpu_use_dp
                }
                pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "mean_absolute_error")
                if tune_mode == 'simple':
                    gbm = lgb.train(param, dtrain, verbose_eval=False)
                    preds = gbm.predict(X_test)
                    mae = mean_absolute_error(Y_test, preds)
                    return mae
                else:
                    result = lgb.cv(param, train_set=dtrain, nfold=5,num_boost_round=param['num_boost_round'],
                                    early_stopping_rounds=10, callbacks=[pruning_callback], seed=42, verbose_eval=False)
                    avg_result = np.mean(np.array(result["mean_absolute_error-mean"]))
                    return avg_result
            algorithm = 'lgbm'
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            #self.optuna_studies[f"{algorithm}"] = {}
            #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
            #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
            #self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
            #self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)
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
                'device': run_on,
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
