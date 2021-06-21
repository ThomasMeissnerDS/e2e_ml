from full_preprocessing import cpu_preprocessing
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ClassificationModels(cpu_preprocessing.MlPipeline):
    def threshold_refiner(self, probs, targets):
        """
        Loops through predicted class probabilities in binary contexts and measures performance with
        Matthew correlation.
        :param probs: Takes predicted class probabilities.
        :param targets: Takes actual targets.
        :return: Stores the best threshold as class attribute.
        """
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
        self.preprocess_decisions[f"probability_threshold"] = best_threshold
        return self.preprocess_decisions[f"probability_threshold"]

    def shap_explanations(self, model, test_df, cols, explainer='tree', algorithm=None):
        """
        See explanations under:
        https://medium.com/rapids-ai/gpu-accelerated-shap-values-with-xgboost-1-3-and-rapids-587fad6822
        :param model: Trained ML model
        :param test_df: Test data to predict on.
        :param explainer: Set "tree" for TreeExplainer. Otherwise uses KernelExplainer.
        :param algorithm: Define name of the chosen ml algorithm as a string.
        :return: Returns plot of feature importance and interactions.
        """
        # print the JS visualization code to the notebook
        shap.initjs()
        if self.prediction_mode:
            to_pred = test_df
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            to_pred = X_test
        if explainer == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(to_pred)
            shap.summary_plot(shap_values, to_pred, plot_type="bar", show=False)
            plt.savefig(f'{algorithm}Shap_feature_importance.png')
            plt.show()
        else:
            model_shap_explainer = shap.KernelExplainer(model.predict, to_pred)
            model_shap_values = model_shap_explainer.shap_values(to_pred)
            shap.summary_plot(model_shap_values, to_pred, show=False)
            plt.savefig(f'{algorithm}Shap_feature_importance.png')
            plt.show()

    def classification_eval(self, algorithm, pred_probs=None, pred_class=None):
        """
        Takes in the algorithm name. This is needed to grab saved predictions and to store cvlassification scores
        of different evaluation functions within the class. Returns the evaluation dictionary.
        :param algorithm: Name of the used algorithm
        :param pred_probs: Probabilities of predictions
        :param pred_class: Predicted classes
        :return: Returns the evaluation dictionary.
        """
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            """
            We need a fallback logic as we might receive different types of data.
            If pred_class will not work with numpy arrays and needs the .any() function.
            """
            try:
                if pred_class:
                    y_hat = pred_class
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]
            except Exception:
                if pred_class.any():
                    y_hat = pred_class
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]

            try:
                if pred_probs:
                    y_hat_probs = pred_probs
                else:
                    y_hat_probs = self.predicted_probs[f"{algorithm}"]
            except Exception:
                if pred_probs.any():
                    y_hat_probs = pred_probs
                else:
                    y_hat_probs = self.predicted_probs[f"{algorithm}"]

            """
            Calculating Matthews, ROC_AUC score and different F1 scores.
            """
            try:
                matthews = matthews_corrcoef(Y_test, y_hat)
            except Exception:
                matthews = 0
            print(f"The Matthew correlation is {matthews}")

            roc_auc = roc_auc_score(Y_test, y_hat_probs)
            print(f"The ROC_AUC score is {roc_auc}")
            f1_score_macro = f1_score(Y_test, y_hat, average='macro')
            print(f"The macro F1 score is {f1_score_macro}")
            f1_score_micro = f1_score(Y_test, y_hat, average='micro')
            print(f"The micro F1 score is {f1_score_micro}")
            f1_score_weighted = f1_score(Y_test, y_hat, average='weighted')
            print(f"The weighted F1 score is {f1_score_weighted}")

            full_classification_report = classification_report(Y_test, y_hat)
            print(full_classification_report)
            self.evaluation_scores[f"{algorithm}"] = {
                'matthews': matthews,
                'roc_auc': roc_auc,
                'f1_score_macro': f1_score_macro,
                'f1_score_micro': f1_score_micro,
                'f1_score_weighted': f1_score_weighted,
                'classfication_report': full_classification_report
            }
            return self.evaluation_scores

    def logistic_regression_train(self):
        algorithm = 'logistic_regression'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = LogisticRegression(random_state=0).fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            return self.trained_models

    def logistic_regression_predict(self):
        algorithm = 'logistic_regression'
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict_proba(self.dataframe)
            if self.class_problem == 'binary':
                partial_probs = np.asarray([line[1] for line in predicted_probs])
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict_proba(X_test)
            self.threshold_refiner(predicted_probs, Y_test)
            if self.class_problem == 'binary':
                partial_probs = np.asarray([line[1] for line in predicted_probs])
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            try:
                self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns, explainer='kernel')
            except Exception:
                self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns, explainer='kernel')
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes

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
                            'tree_method': 'gpu_hist', #use GPU for training
                            'num_class': Y_train.nunique(),
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
                        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mlogloss")
                        if tune_mode == 'simple':
                            eval_set = [(D_train, 'train'), (D_test, 'test')]
                            model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                              evals=eval_set, callbacks=[pruning_callback])
                            preds = model.predict(D_test)
                            pred_labels = np.rint(preds)
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                            as_pandas=True, seed=42, callbacks=[pruning_callback])
                            #avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result['test-mlogloss-mean'].mean()

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
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'mlogloss',
                        'verbose': 0,
                        'tree_method': 'gpu_hist', #use GPU for training
                        'num_class': Y_train.nunique(),
                        'max_depth': 2,  #maximum depth of the decision trees being trained
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
                    def objective(trial):
                        param = {
                            'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                            'eval_metric': 'mlogloss',
                            'verbose': 0,
                            'tree_method': 'gpu_hist', #use GPU for training
                            'num_class': Y_train.nunique(),
                            'max_depth': trial.suggest_int('max_depth', 2, 30),  #maximum depth of the decision trees being trained
                            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
                            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
                            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                            'eta': trial.suggest_loguniform('eta', 0.001, 0.3), #0.001
                            'steps': trial.suggest_int('steps', 2, 70000),
                            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5)
                        }
                        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mlogloss")
                        if tune_mode == 'simple':
                            eval_set = [(D_train, 'train'), (D_test, 'test')]
                            model = xgb.train(param, D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                              evals=eval_set, callbacks=[pruning_callback])
                            preds = model.predict(D_test)
                            pred_labels = np.rint(preds)
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(params=param, dtrain=D_train, num_boost_round=param['steps'], early_stopping_rounds=10,
                                            as_pandas=True, seed=42, callbacks=[pruning_callback])
                            #avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result['test-mlogloss-mean'].mean()

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
                        'objective': 'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'eval_metric': 'mlogloss',
                        'verbose': 0,
                        'tree_method': 'gpu_hist', #use GPU for training
                        'num_class': Y_train.nunique(),
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
                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced',
                    y=Y_train
                )
                D_train = xgb.DMatrix(X_train, label=Y_train, weight=classes_weights)
                D_test = xgb.DMatrix(X_test, label=Y_test)
                algorithm = 'xgboost'
                if not param:
                    param = {
                        'eta': 0.001, #learning rate,
                        'scale_pos_weight' : 1, #A typical value to consider: sum(negative instances) / sum(positive instances) (default = 1)
                        #'gamma': 5, #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                        'verbosity': 0, #0 (silent), 1 (warning), 2 (info), 3 (debug)
                        'alpha' : 10, #L1 regularization term on weights. Increasing this value will make model more conservative. (default = 0)
                        'lambda': 15, #L2 regularization term on weights. Increasing this value will make model more conservative. (default = 1)
                        'subsample': 0.8,
                        'eval_metric' : "mlogloss", #'mlogloss','auc','rmsle'
                        #'colsample_bytree': 0.3,
                        'max_depth': 2, #maximum depth of the decision trees being trained
                        'tree_method': 'gpu_hist', #use GPU for training
                        'objective':'multi:softprob',  # OR  'binary:logistic' #the loss function being used
                        'steps' : 50000,
                        'num_class': self.num_classes} #the number of classes in the dataset
                else:
                    param = param

                if not steps:
                    steps = 10000
                else:
                    steps = steps
                eval_set = [(D_train, 'train'), (D_test, 'test')]
                model = xgb.train(param, D_train, num_boost_round=50000, early_stopping_rounds=10,
                                  evals=eval_set)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                return self.trained_models

    def xgboost_predict(self, show_shap=True):
        """
        Predicts on test & also new data given the prediction_mode is activated in the class.
        :return: Updates class attributes by its predictions.
        """
        algorithm = 'xgboost'
        if self.prediction_mode:
            D_test = xgb.DMatrix(self.dataframe)
            model = self.trained_models[f"{algorithm}"]
            predicted_probs = model.predict(D_test)
            if self.class_problem == 'binary':
                partial_probs = np.asarray([line[1] for line in predicted_probs])
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
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
                predicted_probs = model.predict(D_test)
                if self.class_problem == 'binary':
                    self.threshold_refiner(predicted_probs, Y_test)
                    partial_probs = np.asarray([line[1] for line in predicted_probs])
                    predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
                else:
                    predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
                self.shap_explanations(model=model, test_df=D_test_sample, cols=X_test.columns)
                self.predicted_probs[f"{algorithm}"] = {}
                self.predicted_classes[f"{algorithm}"] = {}
                self.predicted_probs[f"{algorithm}"] = predicted_probs
                self.predicted_classes[f"{algorithm}"] = predicted_classes
                return self.predicted_probs
            elif self.xgboost_objective == 'regression':
                self.xg_boost_regression = model.predict(D_test)
                return self.xg_boost_regression

    def lgbm_train(self, tune_mode='accurate', run_on='gpu', gpu_use_dp=True):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if self.class_problem == 'binary':
                def objective(trial):
                    dtrain = lgb.Dataset(X_train, label=Y_train)
                    param = {
                        # TODO: Move to additional folder with pyfile "constants" (use OS absolute path)
                        'objective': 'binary',
                        'metric': 'binary_logloss',
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

                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
                    if tune_mode == 'simple':
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.rint(preds)
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        result = lgb.cv(param, train_set=dtrain, nfold=5,num_boost_round=param['num_boost_round'],
                                        early_stopping_rounds=10, callbacks=[pruning_callback], seed=42, verbose_eval=False)
                        avg_result = np.mean(np.array(result["multi_logloss-mean"]))
                        return avg_result

                algorithm = 'lgbm'
                if self.class_problem == 'binary':
                    study = optuna.create_study(direction='maximize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=20)
                #self.optuna_studies[f"{algorithm}"] = {}
                #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                #self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                #self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)


                lgbm_best_param = study.best_trial.params
                param = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
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

            else:
                def objective(trial):
                    dtrain = lgb.Dataset(X_train, label=Y_train)
                    param = {
                        'objective': 'multiclass',
                        'metric': 'multi_logloss',
                        'num_boost_round': trial.suggest_int('num_boost_round', 100, 50000),
                        'num_class': Y_train.nunique(),
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

                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
                    if tune_mode == 'simple':
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.asarray([np.argmax(line) for line in preds])
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        result = lgb.cv(param, train_set=dtrain, nfold=5, num_boost_round=param['num_boost_round'],
                                        early_stopping_rounds=10, callbacks=[pruning_callback], seed=42, verbose_eval=False)
                        #fobj=lgb_matth_score)
                        print(result)
                        avg_result = np.mean(np.array(result["multi_logloss-mean"])) # Planned: matthew-mean
                        return avg_result

                algorithm = 'lgbm'
                if self.class_problem == 'binary':
                    study = optuna.create_study(direction='maximize')
                else:
                    study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=20)
                #self.optuna_studies[f"{algorithm}"] = {}
                #optuna.visualization.plot_optimization_history(study).write_image('LGBM_optimization_history.png')
                #optuna.visualization.plot_param_importances(study).write_image('LGBM_param_importances.png')
                #self.optuna_studies[f"{algorithm}_plot_optimization"] = optuna.visualization.plot_optimization_history(study)
                #self.optuna_studies[f"{algorithm}_param_importance"] = optuna.visualization.plot_param_importances(study)

                lgbm_best_param = study.best_trial.params
                param = {
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'num_boost_round': lgbm_best_param["num_boost_round"],
                    'num_class': Y_train.nunique(),
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

    def lgbm_predict(self):
        algorithm = 'lgbm'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted_probs = model.predict(X_test)
            if self.class_problem == 'binary':
                partial_probs = predicted_probs
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
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
                self.threshold_refiner(predicted_probs, Y_test)
                predicted_classes = predicted_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            try:
                self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns)
            except Exception:
                self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns)
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        return self.predicted_probs

    def sklearn_ensemble_train(self):
        """
        Trains an sklearn stacking classifier ensemble.
        :return: Updates class attributes by its predictions.
        """
        algorithm = 'sklearn_ensemble'
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            level0 = list()
            level0.append(('lgbm', LGBMClassifier()))
            level0.append(('lr', LogisticRegression()))
            level0.append(('kbc', KNeighborsClassifier(2)))
            level0.append(('gdc', GradientBoostingClassifier()))
            level0.append(('cart', DecisionTreeClassifier(max_depth=7)))
            level0.append(('abc', AdaBoostClassifier()))
            level0.append(('qda', QuadraticDiscriminantAnalysis()))
            level0.append(('rdf',  RandomForestClassifier(max_depth=7)))
            # define meta learner model
            level1 = GradientBoostingClassifier()
            # define the stacking ensemble
            model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            print(X_train.info())
            model.fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            return self.trained_models

    def sklearn_ensemble_predict(self):
        """
        Predicts on test & also new data given the prediction_mode is activated in the class.
        :return: Updates class attributes by its predictions.
        """
        algorithm = 'sklearn_ensemble'
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                partial_probs = np.asarray([line[1] for line in predicted_probs])
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted_probs = model.predict_proba(X_test)
            if self.class_problem == 'binary':
                self.threshold_refiner(predicted_probs, Y_test)
                partial_probs = np.asarray([line[1] for line in predicted_probs])
                predicted_classes = partial_probs > self.preprocess_decisions[f"probability_threshold"]
            else:
                predicted_classes = np.asarray([np.argmax(line) for line in predicted_probs])
            try:
                self.shap_explanations(model=model, test_df=X_test.sample(10000, random_state=42), cols=X_test.columns, explainer='kernel')
            except Exception:
                self.shap_explanations(model=model, test_df=X_test, cols=X_test.columns, explainer='kernel')
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        return self.predicted_probs