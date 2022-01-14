import gc
import logging
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from pandas.core.common import SettingWithCopyWarning
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy import optimize, special
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import _ConstantPredictor
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import class_weight
from sklearn.utils.extmath import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vowpalwabbit.sklearn_vw import VWClassifier

from e2eml.full_processing import postprocessing

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


class RidgeClassifierWithProba(RidgeClassifier):
    def predict_proba(self, X):
        d = self.decision_function(X)
        d_2d = np.c_[-d, d]
        return softmax(d_2d)


class FocalLoss:
    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(), bounds=(0, 1), method="bounded"
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return "focal_loss", self(y, p).mean(), is_higher_better


class OneVsRestLightGBMWithCustomizedLoss:
    def __init__(self, loss, n_jobs=4):
        self.loss = loss
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        if "eval_set" in fit_params:
            # use eval_set for early stopping
            X_val, y_val = fit_params["eval_set"][0]
            Y_val = self.label_binarizer_.transform(y_val)
            Y_val = Y_val.tocsc()
            columns_val = (col.toarray().ravel() for col in Y_val.T)
            self.results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_binary)(X, column, X_val, column_val, **fit_params)
                for i, (column, column_val) in enumerate(zip(columns, columns_val))
            )
        else:
            # eval set not available
            self.results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_binary)(X, column, None, None, **fit_params)
                for i, column in enumerate(columns)
            )

        return self

    def _fit_binary(self, X, y, X_val, y_val, **fit_params):
        unique_y = np.unique(y)
        init_score_value = self.loss.init_score(y)
        if len(unique_y) == 1:
            estimator = _ConstantPredictor().fit(X, unique_y)
        else:
            fit_params["verbose"] = -1
            fit = lgb.Dataset(
                X, y, init_score=np.full_like(y, init_score_value, dtype=float)
            )
            if "eval_set" in fit_params:
                val = lgb.Dataset(
                    X_val,
                    y_val,
                    init_score=np.full_like(y_val, init_score_value, dtype=float),
                    reference=fit,
                    silent=True,
                )

                estimator = lgb.train(
                    params=fit_params,
                    train_set=fit,
                    valid_sets=(fit, val),
                    valid_names=("fit", "val"),
                    early_stopping_rounds=10,
                    fobj=self.loss.lgb_obj,
                    feval=self.loss.lgb_eval,
                    verbose_eval=False,
                )
            else:
                estimator = lgb.train(
                    params=fit_params,
                    train_set=fit,
                    fobj=self.loss.lgb_obj,
                    feval=self.loss.lgb_eval,
                    verbose_eval=False,
                )

        return estimator, init_score_value

    def predict(self, X):

        n_samples = X.shape[0]
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)

        for i, (e, init_score) in enumerate(self.results_):
            margins = e.predict(X, raw_score=True)
            prob = special.expit(margins + init_score)
            np.maximum(maxima, prob, out=maxima)
            argmaxima[maxima == prob] = i

        return argmaxima

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], len(self.results_)))
        for i, (e, init_score) in enumerate(self.results_):
            margins = e.predict(X, raw_score=True)
            y[:, i] = special.expit(margins + init_score)
        y /= np.sum(y, axis=1)[:, np.newaxis]
        return y


class ClassificationModels(
    postprocessing.FullPipeline,
    Matthews,
    FocalLoss,
    OneVsRestLightGBMWithCustomizedLoss,
):
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
    and optimize accordingly. (Default: 'auto'). Only TabNet can only run on GPU and will not be impacted from this parameter.
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
            self.preprocess_decisions["probability_threshold"] = {}

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
        self.preprocess_decisions["probability_threshold"][algorithm] = best_threshold
        return self.preprocess_decisions["probability_threshold"][algorithm]

    def logistic_regression_train(self):
        """
        Trains a simple Logistic regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train logistic regression model")
        algorithm = "logistic_regression"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            try:
                model = LogisticRegression(random_state=0).fit(X_train, Y_train)
            except AttributeError:
                model = LogisticRegression(random_state=0, solver="liblinear").fit(
                    X_train, Y_train
                )
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def logistic_regression_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Logistic Regression")
        algorithm = "logistic_regression"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model,
                    X_test,
                    Y_test.astype(int),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
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

    def lgbm_focal_train(self):
        """
        Trains a simple LGBM with focal loss classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train LGBM with focal loss model")
        algorithm = "lgbm_focal"
        # self.check_gpu_support(algorithm='lgbm')

        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            fit_params = {"eval_set": [(X_test, Y_test)]}
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()

            if self.class_problem == "binary":

                def objective(trial):
                    param = {
                        "learning_rate": trial.suggest_loguniform(
                            "learning_rate", 0.01, 0.1
                        ),
                        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                        "num_boost_round": trial.suggest_int(
                            "num_boost_round", 100, 50000
                        ),
                        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10),
                        "gamma": trial.suggest_int("gamma", 1, 100),
                        "verbose": -1,
                    }
                    fl = FocalLoss(alpha=param["alpha"], gamma=param["gamma"])
                    dtrain = lgb.Dataset(
                        x_train,
                        label=y_train,
                        init_score=np.full_like(
                            y_train, fl.init_score(y_train), dtype=float
                        ),
                    )

                    result = lgb.cv(
                        param,
                        train_set=dtrain,
                        nfold=10,
                        num_boost_round=param["num_boost_round"],
                        early_stopping_rounds=10,
                        seed=42,
                        fobj=fl.lgb_obj,
                        feval=fl.lgb_eval,
                        verbose_eval=False,
                    )
                    avg_result = np.mean(np.array(result["focal_loss-mean"]))
                    return avg_result

            else:

                def objective(trial):
                    param = {
                        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10),
                        "gamma": trial.suggest_int("gamma", 1, 100),
                        "num_boost_round": trial.suggest_int(
                            "num_boost_round", 100, 1000
                        ),
                        "verbose": -1,
                        "learning_rate": trial.suggest_loguniform(
                            "learning_rate", 0.01, 0.1
                        ),
                    }
                    loss = FocalLoss(alpha=param["alpha"], gamma=param["gamma"])
                    model = OneVsRestLightGBMWithCustomizedLoss(loss=loss)

                    scores = []
                    for fold in range(5):
                        print(f"Calculating CV number {fold}...")
                        try:
                            (
                                x_train,
                                y_train,
                            ) = self.get_hyperparameter_tuning_sample_df()
                            model.fit(x_train, y_train, **fit_params)
                            y_pred = model.predict(X_test)
                            mae = matthews_corrcoef(Y_test, y_pred)
                        except Exception:
                            mae = 0
                        scores.append(mae)
                    final_cv_score = np.mean(np.array(scores))
                    return final_cv_score

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            if self.class_problem == "binary":

                study = optuna.create_study(
                    direction="maximize", sampler=sampler, study_name=f"{algorithm}"
                )
                study.optimize(
                    objective,
                    n_trials=self.hyperparameter_tuning_rounds[algorithm],
                    timeout=self.hyperparameter_tuning_max_runtime_secs[algorithm],
                    gc_after_trial=True,
                    show_progress_bar=True,
                )
                self.optuna_studies[f"{algorithm}"] = {}
            else:
                study = optuna.create_study(
                    direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            if self.class_problem == "binary":
                best_parameters = study.best_trial.params
                fl = FocalLoss(
                    alpha=best_parameters["alpha"], gamma=best_parameters["gamma"]
                )
                try:
                    X_train = X_train.drop(self.target_variable, axis=1)
                except Exception:
                    pass
                Dtrain = lgb.Dataset(X_train, label=Y_train)
                Dtest = lgb.Dataset(X_test, label=Y_test)
                model = lgb.train(
                    best_parameters,
                    Dtrain,
                    valid_sets=[Dtrain, Dtest],
                    valid_names=["train", "valid"],
                    early_stopping_rounds=10,
                    fobj=fl.lgb_obj,
                    feval=fl.lgb_eval,
                )

                self.preprocess_decisions["focal_init_score"] = fl.init_score(Y_train)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models
            else:
                best_parameters = study.best_trial.params
                loss = FocalLoss(
                    alpha=best_parameters["alpha"], gamma=best_parameters["gamma"]
                )
                model = OneVsRestLightGBMWithCustomizedLoss(loss=loss)
                model.fit(X_train, Y_train, **fit_params)
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

    def lgbm_focal_predict(self, feat_importance=True, importance_alg="SHAP"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with LGBM with focal loss")
        algorithm = "lgbm_focal"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            if self.class_problem == "binary":
                fl_init_score = self.preprocess_decisions["focal_init_score"]
                predicted_probs = special.expit(
                    fl_init_score + model.predict(self.dataframe)
                )
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = model.predict(self.dataframe)
                predicted_classes = predicted_probs
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            if self.class_problem == "binary":
                fl_init_score = self.preprocess_decisions["focal_init_score"]
                predicted_probs = special.expit(fl_init_score + model.predict(X_test))
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = model.predict(X_test)
                predicted_classes = predicted_probs

            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
            del model
            _ = gc.collect()

    def quadratic_discriminant_analysis_train(self):
        """
        Trains a simple Quadratic Discriminant model.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Quadratic Discriminant model")
        algorithm = "quadratic_discriminant_analysis"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = QuadraticDiscriminantAnalysis().fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def quadratic_discriminant_analysis_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Quadratic Discriminant Analysis")
        algorithm = "quadratic_discriminant_analysis"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model,
                    X_test,
                    Y_test.astype(int),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
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

    def ridge_classifier_train(self):
        """
        Trains a simple ridge regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Ridge classifier model")
        algorithm = "ridge"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            metric = make_scorer(matthews_corrcoef)

            def objective(trial):
                solver = trial.suggest_categorical(
                    "solver",
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                )
                param = {
                    "alpha": trial.suggest_loguniform("alpha", 1e-3, 1e3),
                    "max_iter": trial.suggest_int("max_iter", 10, 10000),
                    "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
                    "normalize": trial.suggest_categorical("normalize", [True, False]),
                }
                model = RidgeClassifier(
                    alpha=param["alpha"],
                    max_iter=param["max_iter"],
                    tol=param["tol"],
                    normalize=param["normalize"],
                    solver=solver,
                    random_state=42,
                )  # .fit(x_train, y_train)
                try:
                    scores = cross_val_score(
                        model, x_train, y_train, cv=10, scoring=metric
                    )
                    mae = np.mean(scores)
                except Exception:
                    mae = 0
                return mae

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            best_parameters = study.best_trial.params
            model = RidgeClassifierWithProba(
                alpha=best_parameters["alpha"],
                max_iter=best_parameters["max_iter"],
                tol=best_parameters["tol"],
                normalize=best_parameters["normalize"],
                solver=best_parameters["solver"],
                random_state=42,
            ).fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def ridge_classifier_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Ridge classifier")
        algorithm = "ridge"

        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def svm_train(self):
        """
        Trains a simple Support Vector Machine classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Support Vector Machine classifier model")
        algorithm = "svm"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            metric = make_scorer(matthews_corrcoef)

            def objective(trial):
                param = {
                    "C": trial.suggest_loguniform("C", 0.5, 1e3),
                    "max_iter": trial.suggest_int("max_iter", 1, 10000),
                    "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                    "class_weight": trial.suggest_categorical(
                        "class_weight", ["balanced", None]
                    ),
                }
                model = SVC(
                    C=param["C"],
                    probability=True,
                    max_iter=param["max_iter"],
                    tol=param["tol"],
                    gamma=param["gamma"],
                    class_weight=param["class_weight"],
                    random_state=42,
                )  # .fit(x_train, y_train)
                try:
                    scores = cross_val_score(
                        model, x_train, y_train, cv=10, scoring=metric
                    )
                    mae = np.mean(scores)
                except Exception:
                    mae = 0
                return mae

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            best_parameters = study.best_trial.params
            model = SVC(
                C=best_parameters["C"],
                probability=True,
                max_iter=best_parameters["max_iter"],
                tol=best_parameters["tol"],
                gamma=best_parameters["gamma"],
                class_weight=best_parameters["class_weight"],
                random_state=42,
            ).fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def svm_predict(self, feat_importance=True, importance_alg="permutation"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(
            task="Predict with Support Vector Machine classifier"
        )
        algorithm = "svm"

        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def multinomial_nb_train(self):
        """
        Trains a Multinomial Naive Bayes classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(
            task="Train Multinomial Naive Bayes classifier model"
        )
        algorithm = "multinomial_nb"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            metric = make_scorer(matthews_corrcoef)

            def objective(trial):
                param = {"alpha": trial.suggest_loguniform("alpha", 1e-6, 1e2)}
                model = MultinomialNB(alpha=param["alpha"])  # .fit(x_train, y_train)
                try:
                    scores = cross_val_score(
                        model, x_train, y_train, cv=10, scoring=metric
                    )
                    mae = np.mean(scores)
                except Exception:
                    mae = 0
                return mae

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            best_parameters = study.best_trial.params
            model = MultinomialNB(alpha=best_parameters["alpha"]).fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def multinomial_nb_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(
            task="Predict with Multinomial Naive Bayes classifier"
        )
        algorithm = "multinomial_nb"

        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def sgd_classifier_train(self):
        """
        Trains a simple sgd classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train SGD classifier model")
        algorithm = "sgd"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            metric = make_scorer(matthews_corrcoef)

            def objective(trial):
                loss = trial.suggest_categorical("loss", ["modified_huber", "log"])
                param = {
                    "alpha": trial.suggest_loguniform("alpha", 1e-3, 1e3),
                    "l1_ratio": trial.suggest_loguniform("l1_ratio", 1e-3, 0.9999),
                    "epsilon": trial.suggest_loguniform("epsilon", 1e-3, 0.3),
                    "max_iter": trial.suggest_int("max_iter", 10, 30000),
                    "tol": trial.suggest_loguniform("tol", 1e-5, 1e-1),
                    "normalize": trial.suggest_categorical("normalize", [True, False]),
                    "power_t": trial.suggest_loguniform("power_t", 0.1, 0.7),
                }
                model = SGDClassifier(
                    alpha=param["alpha"],
                    max_iter=param["max_iter"],
                    tol=param["tol"],
                    l1_ratio=param["l1_ratio"],
                    power_t=param["power_t"],
                    penalty="elasticnet",
                    epsilon=param["epsilon"],
                    loss=loss,
                    early_stopping=True,
                    validation_fraction=0.2,
                    class_weight="balanced",
                    random_state=42,
                )  # .fit(X_train, Y_train)
                try:
                    scores = cross_val_score(
                        model, x_train, y_train, cv=5, scoring=metric
                    )
                    mae = np.mean(scores)
                except Exception:
                    mae = 0
                return mae

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            best_parameters = study.best_trial.params
            model = SGDClassifier(
                alpha=best_parameters["alpha"],
                max_iter=best_parameters["max_iter"],
                tol=best_parameters["tol"],
                l1_ratio=best_parameters["l1_ratio"],
                penalty="elasticnet",
                loss=best_parameters["loss"],
                power_t=best_parameters["power_t"],
                epsilon=best_parameters["epsilon"],
                early_stopping=True,
                validation_fraction=0.2,
                class_weight="balanced",
                random_state=42,
            ).fit(X_train, Y_train)

            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def sgd_classifier_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with SGD classifier")
        algorithm = "sgd"

        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def catboost_train(self):
        """
        Trains a Ridge regression model.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train catboost regression model")
        self.check_gpu_support(algorithm="catboost")
        algorithm = "catboost"
        metric = make_scorer(matthews_corrcoef)

        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            eval_dataset = Pool(X_test, Y_test)

            def objective(trial):
                class_weighting = trial.suggest_categorical(
                    "class_weighting", ["None", "Balanced", "SqrtBalanced"]
                )
                param = {
                    "iterations": trial.suggest_int("iterations", 10, 50000),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 1e-3, 0.3
                    ),
                    "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 1e6),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                }
                model = CatBoostClassifier(
                    iterations=param["iterations"],
                    learning_rate=param["learning_rate"],
                    l2_leaf_reg=param["l2_leaf_reg"],
                    max_depth=param["max_depth"],
                    early_stopping_rounds=10,
                    loss_function="MultiClass",
                    # eval_metric=["MultiClass"],
                    auto_class_weights=class_weighting,
                    verbose=500,
                    random_state=42,
                )  # .fit(x_train, y_train,
                #     eval_set=eval_dataset,
                #     early_stopping_rounds=10)
                try:
                    scores = cross_val_score(
                        model, x_train, y_train, cv=10, scoring=metric
                    )
                    mae = np.mean(scores)
                except Exception:
                    mae = 0
                return mae

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm}"
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

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            best_parameters = study.best_trial.params
            model = CatBoostClassifier(
                iterations=best_parameters["iterations"],
                learning_rate=best_parameters["learning_rate"],
                l2_leaf_reg=best_parameters["l2_leaf_reg"],
                max_depth=best_parameters["max_depth"],
                early_stopping_rounds=10,
                loss_function="MultiClass",
                # eval_metric=["MultiClass"],
                auto_class_weights=best_parameters["class_weighting"],
                verbose=500,
                random_state=42,
            ).fit(
                X_train,
                Y_train,
                eval_set=eval_dataset,
                early_stopping_rounds=10,
                plot=True,
            )
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def catboost_predict(self, feat_importance=True, importance_alg="permutation"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with catboost regression")
        algorithm = "catboost"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)

            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=-1
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
                fig, ax = plt.subplots()
                permutation_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            else:
                pass
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def tabnet_train(self):
        """
        Trains a simple Linear regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Tabnet classification model")
        algorithm = "tabnet"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train_sample, y_train_sample = self.get_hyperparameter_tuning_sample_df()

            """# test
            categorical_columns = []
            categorical_dims =  {}
            for col in X_train.columns[X_train.dtypes == object]:
                from sklearn.preprocessing import LabelEncoder
                l_enc = LabelEncoder()
                X_train[col] = X_train[col].fillna("VV_likely")
                X_train[col] = l_enc.fit_transform(X_train[col].values)
                X_test[col] = l_enc.transform(X_test[col].values)
                categorical_columns.append(col)
                categorical_dims[col] = len(l_enc.classes_)
            cat_idxs = [i for i, f in enumerate(X_train.columns.to_list()) if f in categorical_columns]
            cat_dims = [categorical_dims[f] for i, f in enumerate(X_train.columns.to_list()) if f in categorical_columns]"""

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
                weights = trial.suggest_int("weights", 0, 1)
                pre_trainer_max_epochs = trial.suggest_int(
                    "pre_trainer_max_epochs", 10, 1000
                )
                trainer_max_epochs = trial.suggest_int("trainer_max_epochs", 10, 1000)

                param = dict(
                    gamma=gamma,
                    # cat_idxs=cat_idxs,
                    # cat_dims=cat_dims,
                    lambda_sparse=lambda_sparse,
                    n_d=depths,
                    n_a=depths,
                    n_shared=n_shared,
                    n_independent=n_independent,
                    n_steps=trial.suggest_int("n_steps", 1, 5),
                    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                    mask_type=mask_type,
                    scheduler_params=dict(
                        mode=mode, patience=30, min_lr=1e-5, factor=factor
                    ),
                    scheduler_fn=ReduceLROnPlateau,
                    seed=42,
                    verbose=0,
                    # device_name='gpu'
                )
                mean_matthew_corr = []
                from sklearn.model_selection import StratifiedKFold

                skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

                for train_index, test_index in skf.split(
                    x_train_sample, y_train_sample
                ):
                    x_train, x_test = (
                        X_train.iloc[train_index],
                        X_train.iloc[test_index],
                    )
                    y_train, y_test = (
                        Y_train.iloc[train_index],
                        Y_train.iloc[test_index],
                    )
                    # numpy conversion
                    y_train = y_train.values.reshape(-1)
                    y_test = y_test.values.reshape(-1)
                    x_train = x_train.to_numpy()
                    x_test = x_test.to_numpy()

                    Y_test_num = Y_test.values.reshape(-1)
                    X_test_num = X_test.to_numpy()

                    pretrainer = TabNetPretrainer(**param)
                    pretrainer.fit(
                        x_train,
                        eval_set=[(x_test)],
                        max_epochs=pre_trainer_max_epochs,
                        patience=30,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        drop_last=True,
                        pretraining_ratio=pretrain_difficulty,
                    )

                    model = TabNetClassifier(**param)
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_test, y_test)],
                        eval_metric=[Matthews],
                        patience=30,
                        batch_size=batch_size,
                        virtual_batch_size=virtual_batch_size,
                        num_workers=num_workers,
                        drop_last=True,
                        from_unsupervised=pretrainer,
                        weights=weights,
                        max_epochs=trainer_max_epochs,
                    )
                    partial_probs = model.predict_proba(X_test_num)
                    if self.class_problem == "binary":
                        predicted_probs = np.asarray(
                            [line[1] for line in partial_probs]
                        )
                        self.threshold_refiner(predicted_probs, Y_test_num, algorithm)
                        predicted_classes = (
                            predicted_probs
                            > self.preprocess_decisions["probability_threshold"][
                                algorithm
                            ]
                        )
                    else:
                        predicted_probs = partial_probs
                        predicted_classes = np.asarray(
                            [np.argmax(line) for line in predicted_probs]
                        )
                    try:
                        matthew = matthews_corrcoef(Y_test_num, predicted_classes)
                    except Exception:
                        matthew = 0
                    mean_matthew_corr.append(matthew)
                    print(mean_matthew_corr)
                # meissner_cv = self.meissner_cv_score(mean_matthew_corr)
                meissner_cv = np.mean(mean_matthew_corr)
                return meissner_cv

            study = optuna.create_study(
                direction="maximize", study_name=f"{algorithm} tuning"
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
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            try:
                fig = optuna.visualization.plot_param_importances(study)
                self.optuna_studies[f"{algorithm}_param_importance"] = fig
                fig.show()
            except ZeroDivisionError:
                print(
                    "Plotting of hyperparameter performances failed. This usually implicates an error during training."
                )

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass

            Y_train = Y_train.astype(int)
            Y_test = Y_test.astype(int)

            Y_train = Y_train.values.reshape(-1)
            Y_test = Y_test.values.reshape(-1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            tabnet_best_param = study.best_trial.params
            param = dict(
                gamma=tabnet_best_param["gamma"],
                lambda_sparse=tabnet_best_param["lambda_sparse"],
                n_d=tabnet_best_param["depths"],
                n_a=tabnet_best_param["depths"],
                n_steps=tabnet_best_param["n_steps"],
                n_shared=tabnet_best_param["n_shared"],
                n_independent=tabnet_best_param["n_independent"],
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                mask_type=tabnet_best_param["mask_type"],
                scheduler_params=dict(
                    mode=tabnet_best_param["mode"],
                    patience=5,
                    min_lr=1e-5,
                    factor=tabnet_best_param["factor"],
                ),
                scheduler_fn=ReduceLROnPlateau,
                seed=42,
                verbose=1,
            )

            pretrainer = TabNetPretrainer(**param)
            pretrainer.fit(
                X_train,
                eval_set=[(X_test)],
                patience=30,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                drop_last=True,
                pretraining_ratio=tabnet_best_param["pretrain_difficulty"],
                max_epochs=tabnet_best_param["pre_trainer_max_epochs"],
            )

            model = TabNetClassifier(**param)
            model.fit(
                X_train,
                Y_train,
                eval_set=[(X_test, Y_test)],
                eval_metric=[Matthews],
                patience=50,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                drop_last=True,
                from_unsupervised=pretrainer,
                weights=tabnet_best_param["weights"],
                max_epochs=tabnet_best_param["trainer_max_epochs"],
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
        self.get_current_timestamp(task="Predict with Tabnet classification")
        algorithm = "tabnet"
        if self.prediction_mode:
            self.dataframe[self.dataframe.columns.to_list()] = self.dataframe[
                self.dataframe.columns.to_list()
            ].astype(float)
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe.to_numpy())
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            Y_train = Y_train.values.reshape(-1)
            Y_test = Y_test.values.reshape(-1)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()

    def vowpal_wabbit_train(self):
        """
        Trains a simple Logistic regression classifier.
        :return: Trained model.
        """
        self.get_current_timestamp(task="Train Vowpal Wabbit model")
        algorithm = "vowpal_wabbit"
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

    def vowpal_wabbit_predict(self, feat_importance=True, importance_alg="permutation"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Vowpal Wabbit")
        algorithm = "vowpal_wabbit"
        if self.prediction_mode:
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(self.dataframe)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                    )
                except Exception:
                    self.shap_explanations(
                        model=model, test_df=X_test, cols=X_test.columns
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model,
                    X_test,
                    Y_test.astype(int),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
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

    def xg_boost_train(  # noqa: C901
        self, param=None, steps=None, autotune=True, tune_mode="accurate"
    ):
        """
        Trains an XGboost model by the given parameters.
        :param param: Takes a dictionary with custom parameter settings.
        :param steps: Integer higher than 0. Defines maximum training steps, iuf not in autotune mode.
        :param autotune: Set "True" for automatic hyperparameter optimization. (Default: true)
        :param tune_mode: 'Simple' for simple 80-20 split validation. 'Accurate': Each hyperparameter set will be validated
        with 5-fold crossvalidation. Longer runtimes, but higher performance. (Default: 'Accurate')
        """
        self.get_current_timestamp(task="Train Xgboost")
        self.check_gpu_support(algorithm="xgboost")
        if self.preferred_training_mode == "auto":
            train_on = self.preprocess_decisions["gpu_support"]["xgboost"]
        elif self.preferred_training_mode == "gpu":
            train_on = "gpu_hist"
            logging.info(
                "Start Xgboost model training on {self.preferred_training_mode}."
            )
        else:
            train_on = "exact"
            logging.info(
                "Start Xgboost model training on {self.preferred_training_mode}."
            )
        if self.prediction_mode:
            pass
        else:
            if autotune:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                classes_weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=Y_train
                )
                D_test = xgb.DMatrix(X_test, label=Y_test)

                x_train, y_train = self.get_hyperparameter_tuning_sample_df()
                classes_weights_sample = class_weight.compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
                d_test = xgb.DMatrix(X_test, label=Y_test)
                # get sample size to run brute force feature selection against

                if self.class_problem == "binary":

                    def objective(trial):
                        param = {
                            "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
                            "eval_metric": "mlogloss",
                            "verbose": 0,
                            "tree_method": train_on,  # use GPU for training
                            "num_class": Y_train.nunique(),
                            "max_depth": trial.suggest_int("max_depth", 2, 7),  # 4
                            # maximum depth of the decision trees being trained
                            "alpha": trial.suggest_loguniform("alpha", 1, 1e6),
                            "lambda": trial.suggest_loguniform("lambda", 1, 1e6),
                            "num_leaves": trial.suggest_int("num_leaves", 2, 128),  # 8
                            "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
                            "colsample_bytree": trial.suggest_uniform(
                                "colsample_bytree", 0.5, 1.0
                            ),
                            "colsample_bylevel": trial.suggest_uniform(
                                "colsample_bylevel", 0.5, 1.0
                            ),
                            "colsample_bynode": trial.suggest_uniform(
                                "colsample_bynode", 0.5, 1.0
                            ),
                            "min_child_samples": trial.suggest_int(
                                "min_child_samples", 5, 1000
                            ),
                            "eta": trial.suggest_loguniform("eta", 1e-3, 0.3),
                            "steps": trial.suggest_int("steps", 2, 70000),  # 100
                            "num_parallel_tree": trial.suggest_int(
                                "num_parallel_tree", 1, 5
                            ),  # 2
                        }
                        sample_weight = trial.suggest_categorical(
                            "sample_weight", [True, False]
                        )
                        if sample_weight:
                            d_train = xgb.DMatrix(
                                x_train, label=y_train, weight=classes_weights_sample
                            )
                        else:
                            d_train = xgb.DMatrix(x_train, label=y_train)
                        pruning_callback = optuna.integration.XGBoostPruningCallback(
                            trial, "test-mlogloss"
                        )

                        if tune_mode == "simple":
                            eval_set = [(d_train, "train"), (d_test, "test")]
                            model = xgb.train(
                                param,
                                d_train,
                                num_boost_round=param["steps"],
                                early_stopping_rounds=10,
                                evals=eval_set,
                                callbacks=[pruning_callback],
                            )
                            preds = model.predict(D_test)
                            pred_labels = np.asarray(
                                [np.argmax(line) for line in preds]
                            )
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(
                                params=param,
                                dtrain=d_train,
                                num_boost_round=param["steps"],
                                early_stopping_rounds=10,
                                nfold=10,
                                as_pandas=True,
                                seed=42,
                                callbacks=[pruning_callback],
                            )
                            # avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result["test-mlogloss-mean"].mean()

                    algorithm = "xgboost"
                    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
                    if tune_mode == "simple":
                        study = optuna.create_study(
                            direction="maximize",
                            sampler=sampler,
                            study_name=f"{algorithm} tuning",
                        )
                        logging.info("Start Xgboost simple validation.")
                    else:
                        study = optuna.create_study(
                            direction="minimize",
                            sampler=sampler,
                            study_name=f"{algorithm} tuning",
                        )
                        logging.info("Start Xgboost advanced validation.")
                    study.optimize(
                        objective,
                        n_trials=self.hyperparameter_tuning_rounds["xgboost"],
                        timeout=self.hyperparameter_tuning_max_runtime_secs["xgboost"],
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

                    lgbm_best_param = study.best_trial.params
                    param = {
                        "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
                        "eval_metric": "mlogloss",
                        "verbose": 0,
                        "tree_method": train_on,  # use GPU for training
                        "num_class": Y_train.nunique(),
                        "max_depth": lgbm_best_param[
                            "max_depth"
                        ],  # maximum depth of the decision trees being trained
                        "alpha": lgbm_best_param["alpha"],
                        "lambda": lgbm_best_param["lambda"],
                        "num_leaves": lgbm_best_param["num_leaves"],
                        "subsample": lgbm_best_param["subsample"],
                        "colsample_bytree": lgbm_best_param["colsample_bytree"],
                        "colsample_bylevel": lgbm_best_param["colsample_bylevel"],
                        "colsample_bynode": lgbm_best_param["colsample_bynode"],
                        "min_child_samples": lgbm_best_param["min_child_samples"],
                        "eta": lgbm_best_param["eta"],
                        "steps": lgbm_best_param["steps"],
                        "num_parallel_tree": lgbm_best_param["num_parallel_tree"],
                    }
                    sample_weight = lgbm_best_param["sample_weight"]
                    try:
                        X_train = X_train.drop(self.target_variable, axis=1)
                    except Exception:
                        pass
                    if sample_weight:
                        D_train = xgb.DMatrix(
                            X_train, label=Y_train, weight=classes_weights
                        )
                    else:
                        D_train = xgb.DMatrix(X_train, label=Y_train)
                    D_test = xgb.DMatrix(X_test, label=Y_test)
                    eval_set = [(D_train, "train"), (D_test, "test")]
                    logging.info(
                        "Start Xgboost final model training with optimized hyperparamers."
                    )

                    model = xgb.train(
                        param,
                        D_train,
                        num_boost_round=param["steps"],
                        early_stopping_rounds=10,
                        evals=eval_set,
                    )
                    self.trained_models[f"{algorithm}"] = {}
                    self.trained_models[f"{algorithm}"] = model
                    return self.trained_models

                else:

                    def objective(trial):
                        param = {
                            "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
                            "eval_metric": "mlogloss",
                            # 'booster': 'dart',
                            # 'skip_drop': trial.suggest_uniform('skip_drop', 0.1, 1.0),
                            # 'rate_drop': trial.suggest_uniform('rate_drop', 0.1, 1.0),
                            "verbose": 0,
                            "tree_method": train_on,  # use GPU for training
                            "num_class": Y_train.nunique(),
                            "max_depth": trial.suggest_int("max_depth", 2, 7),
                            # maximum depth of the decision trees being trained
                            "alpha": trial.suggest_loguniform("alpha", 1, 1e6),
                            "lambda": trial.suggest_loguniform("lambda", 1, 1e6),
                            "num_leaves": trial.suggest_int("num_leaves", 2, 40),
                            "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
                            "colsample_bytree": trial.suggest_uniform(
                                "colsample_bytree", 0.5, 1.0
                            ),
                            "colsample_bylevel": trial.suggest_uniform(
                                "colsample_bylevel", 0.5, 1.0
                            ),
                            "colsample_bynode": trial.suggest_uniform(
                                "colsample_bynode", 0.5, 1.0
                            ),
                            "min_child_samples": trial.suggest_int(
                                "min_child_samples", 5, 1000
                            ),
                            "eta": trial.suggest_loguniform("eta", 1e-3, 0.3),  # 0.001
                            "steps": trial.suggest_int("steps", 2, 70000),
                            "num_parallel_tree": trial.suggest_int(
                                "num_parallel_tree", 1, 5
                            ),
                        }
                        sample_weight = trial.suggest_categorical(
                            "sample_weight", [True, False]
                        )
                        if sample_weight:
                            d_train = xgb.DMatrix(
                                x_train, label=y_train, weight=classes_weights_sample
                            )
                        else:
                            d_train = xgb.DMatrix(x_train, label=y_train)
                        pruning_callback = optuna.integration.XGBoostPruningCallback(
                            trial, "test-mlogloss"
                        )
                        if tune_mode == "simple":
                            eval_set = [(d_train, "train"), (d_test, "test")]
                            model = xgb.train(
                                param,
                                d_train,
                                num_boost_round=param["steps"],
                                early_stopping_rounds=10,
                                evals=eval_set,
                                callbacks=[pruning_callback],
                            )
                            preds = model.predict(D_test)
                            pred_labels = np.asarray(
                                [np.argmax(line) for line in preds]
                            )
                            matthew = matthews_corrcoef(Y_test, pred_labels)
                            return matthew
                        else:
                            result = xgb.cv(
                                params=param,
                                dtrain=d_train,
                                num_boost_round=param["steps"],
                                early_stopping_rounds=10,
                                nfold=10,
                                as_pandas=True,
                                seed=42,
                                callbacks=[pruning_callback],
                            )
                            # avg_result = (result['train-mlogloss-mean'].mean() + result['test-mlogloss-mean'].mean())/2
                            return result["test-mlogloss-mean"].mean()

                    algorithm = "xgboost"
                    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
                    if tune_mode == "simple":
                        logging.info("Start Xgboost simple validation.")
                        study = optuna.create_study(
                            direction="maximize",
                            sampler=sampler,
                            study_name=f"{algorithm} tuning",
                        )
                    else:
                        logging.info("Start Xgboost advanced validation.")
                        study = optuna.create_study(
                            direction="minimize",
                            sampler=sampler,
                            study_name=f"{algorithm} tuning",
                        )
                    study.optimize(
                        objective,
                        n_trials=self.hyperparameter_tuning_rounds["xgboost"],
                        timeout=self.hyperparameter_tuning_max_runtime_secs["xgboost"],
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

                    lgbm_best_param = study.best_trial.params
                    param = {
                        "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
                        "eval_metric": "mlogloss",
                        # 'booster': 'dart',
                        # 'skip_drop': lgbm_best_param["skip_drop"],
                        # 'rate_drop': lgbm_best_param["rate_drop"],
                        "verbose": 0,
                        "tree_method": train_on,  # use GPU for training
                        "num_class": Y_train.nunique(),
                        "max_depth": lgbm_best_param[
                            "max_depth"
                        ],  # maximum depth of the decision trees being trained
                        "alpha": lgbm_best_param["alpha"],
                        "lambda": lgbm_best_param["lambda"],
                        "num_leaves": lgbm_best_param["num_leaves"],
                        "subsample": lgbm_best_param["subsample"],
                        "colsample_bytree": lgbm_best_param["colsample_bytree"],
                        "colsample_bylevel": lgbm_best_param["colsample_bylevel"],
                        "colsample_bynode": lgbm_best_param["colsample_bynode"],
                        "min_child_samples": lgbm_best_param["min_child_samples"],
                        "eta": lgbm_best_param["eta"],
                        "steps": lgbm_best_param["steps"],
                        "num_parallel_tree": lgbm_best_param["num_parallel_tree"],
                    }
                    sample_weight = lgbm_best_param["sample_weight"]
                    try:
                        X_train = X_train.drop(self.target_variable, axis=1)
                    except Exception:
                        pass
                    if sample_weight:
                        D_train = xgb.DMatrix(
                            X_train, label=Y_train, weight=classes_weights
                        )
                    else:
                        D_train = xgb.DMatrix(X_train, label=Y_train)
                    D_test = xgb.DMatrix(X_test, label=Y_test)
                    eval_set = [(D_train, "train"), (D_test, "test")]
                    logging.info(
                        "Start Xgboost final model training with optimized hyperparamers."
                    )
                    model = xgb.train(
                        param,
                        D_train,
                        num_boost_round=param["steps"],
                        early_stopping_rounds=10,
                        evals=eval_set,
                    )
                    self.trained_models[f"{algorithm}"] = {}
                    self.trained_models[f"{algorithm}"] = model
                    del model
                    _ = gc.collect()
                    return self.trained_models
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                classes_weights = class_weight.compute_sample_weight(
                    class_weight="balanced", y=Y_train
                )
                if self.binary_unbalanced:
                    D_train = xgb.DMatrix(
                        X_train, label=Y_train, weight=classes_weights
                    )
                else:
                    D_train = xgb.DMatrix(X_train, label=Y_train)
                D_test = xgb.DMatrix(X_test, label=Y_test)
                algorithm = "xgboost"
                if not param:
                    param = {
                        "eta": 0.001,  # learning rate,
                        "scale_pos_weight": 1,
                        # A typical value to consider: sum(negative instances) / sum(positive instances) (default = 1)
                        # 'gamma': 5, #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                        "verbosity": 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
                        "alpha": 1,
                        # L1 regularization term on weights. Increasing this value will make model more conservative. (default = 0)
                        "lambda": 1,
                        # L2 regularization term on weights. Increasing this value will make model more conservative. (default = 1)
                        "subsample": 0.8,
                        "eval_metric": "mlogloss",  # 'mlogloss','auc','rmsle'
                        # 'colsample_bytree': 0.3,
                        "max_depth": 2,  # maximum depth of the decision trees being trained
                        "tree_method": train_on,  # use GPU for training
                        "objective": "multi:softprob",  # OR  'binary:logistic' #the loss function being used
                        "steps": 50000,
                        "num_class": self.num_classes,
                    }  # the number of classes in the dataset
                else:
                    param = param

                if not steps:
                    steps = 10000
                else:
                    steps = steps

                try:
                    X_train = X_train.drop(self.target_variable, axis=1)
                except Exception:
                    pass
                eval_set = [(D_train, "train"), (D_test, "test")]
                logging.info(
                    "Start Xgboost simple model training with predefined hyperparamers."
                )
                model = xgb.train(
                    param,
                    D_train,
                    num_boost_round=50000,
                    early_stopping_rounds=10,
                    evals=eval_set,
                )
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

    def xgboost_predict(self, feat_importance=True, importance_alg="auto"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated based on SHAP values.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Xgboost")
        algorithm = "xgboost"
        if self.prediction_mode:
            D_test = xgb.DMatrix(self.dataframe)
            model = self.trained_models[f"{algorithm}"]
            partial_probs = model.predict(D_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in partial_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            D_test = xgb.DMatrix(X_test, label=Y_test)
            try:
                D_test_sample = xgb.DMatrix(
                    X_test.sample(10000, random_state=42), label=Y_test
                )
            except Exception:
                D_test_sample = xgb.DMatrix(X_test, label=Y_test)
            model = self.trained_models[f"{algorithm}"]
            if self.class_problem == "binary" or self.class_problem == "multiclass":
                partial_probs = model.predict(D_test)
                if self.class_problem == "binary":
                    predicted_probs = np.asarray([line[1] for line in partial_probs])
                    self.threshold_refiner(predicted_probs, Y_test, algorithm)
                    predicted_classes = (
                        predicted_probs
                        > self.preprocess_decisions["probability_threshold"][algorithm]
                    )
                else:
                    predicted_probs = partial_probs
                    predicted_classes = np.asarray(
                        [np.argmax(line) for line in partial_probs]
                    )

                if feat_importance and importance_alg == "auto":
                    if (
                        self.preprocess_decisions["gpu_support"]["xgboost"]
                        == "gpu_hist"
                    ):
                        if self.class_problem == "binary":
                            self.shap_explanations(
                                model=model, test_df=D_test_sample, cols=X_test.columns
                            )
                        else:
                            xgb.plot_importance(model)
                        plt.figure(figsize=(16, 12))
                        plt.show()
                    else:
                        xgb.plot_importance(model)
                        plt.figure(figsize=(16, 12))
                        plt.show()
                elif feat_importance and importance_alg == "SHAP":
                    if self.class_problem == "binary":
                        self.shap_explanations(
                            model=model, test_df=D_test_sample, cols=X_test.columns
                        )
                    else:
                        xgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
                elif feat_importance and importance_alg == "inbuilt":
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
            elif self.xgboost_objective == "regression":
                self.xg_boost_regression = model.predict(D_test)
                return self.xg_boost_regression

    def lgbm_train(self, tune_mode="accurate", gpu_use_dp=True):  # noqa: C901
        """
        Trains an LGBM model by the given parameters.
        :param tune_mode: 'Simple' for simple 80-20 split validation. 'Accurate': Each hyperparameter set will be validated
        with 10-fold cross validation. Longer runtimes, but higher performance. (Default: 'Accurate')
        :param gpu_use_dp: If True and when GPU accelerated, LGBM will use bigger floats for higher accuracy, but at the
        cost of longer runtimes (Default: True)
        """
        self.get_current_timestamp(task="Train LGBM")
        self.check_gpu_support(algorithm="lgbm")
        if self.preferred_training_mode == "auto":
            train_on = self.preprocess_decisions["gpu_support"]["lgbm"]
        elif self.preferred_training_mode == "gpu":
            train_on = "gpu"
            gpu_use_dp = True
        else:
            train_on = "cpu"
            gpu_use_dp = False

        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()

            try:
                x_train = x_train.drop(self.target_variable, axis=1)
            except Exception:
                pass
            dtrain = lgb.Dataset(x_train, label=y_train)

            if self.class_problem == "binary":
                # weights_for_lgb = self.calc_scale_pos_weight()

                def objective(trial):
                    param = {
                        "objective": "binary",
                        "metric": "binary_logloss",
                        # 'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1e3),
                        "num_boost_round": trial.suggest_int(
                            "num_boost_round", 100, 50000
                        ),
                        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1, 1e6),
                        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1, 1e6),
                        # 'max_depth': trial.suggest_int('max_depth', 2, 8),
                        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                        "feature_fraction": trial.suggest_uniform(
                            "feature_fraction", 0.4, 1.0
                        ),
                        "feature_fraction_bynode": trial.suggest_uniform(
                            "feature_fraction_bynode", 0.4, 1.0
                        ),
                        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        "learning_rate": trial.suggest_loguniform(
                            "learning_rate", 1e-5, 0.1
                        ),
                        "verbose": -1,
                        "device": train_on,
                        "gpu_use_dp": gpu_use_dp,
                    }

                    pruning_callback = optuna.integration.LightGBMPruningCallback(
                        trial, "binary_logloss"
                    )
                    if tune_mode == "simple":
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.rint(preds)
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        result = lgb.cv(
                            param,
                            train_set=dtrain,
                            nfold=10,
                            num_boost_round=param["num_boost_round"],
                            early_stopping_rounds=10,
                            callbacks=[pruning_callback],
                            seed=42,
                            verbose_eval=False,
                        )
                        avg_result = np.mean(np.array(result["binary_logloss-mean"]))
                        return avg_result

                algorithm = "lgbm"
                sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
                if tune_mode == "simple":
                    study = optuna.create_study(
                        direction="maximize",
                        sampler=sampler,
                        study_name=f"{algorithm} tuning",
                    )
                else:
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=sampler,
                        study_name=f"{algorithm} tuning",
                    )

                study.optimize(
                    objective,
                    n_trials=self.hyperparameter_tuning_rounds["lgbm"],
                    timeout=self.hyperparameter_tuning_max_runtime_secs["lgbm"],
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

                lgbm_best_param = study.best_trial.params
                param = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    # 'scale_pos_weight': lgbm_best_param["scale_pos_weight"],
                    # 'max_depth': lgbm_best_param["max_depth"],
                    "num_boost_round": lgbm_best_param["num_boost_round"],
                    "lambda_l1": lgbm_best_param["lambda_l1"],
                    "lambda_l2": lgbm_best_param["lambda_l2"],
                    "num_leaves": lgbm_best_param["num_leaves"],
                    "feature_fraction": lgbm_best_param["feature_fraction"],
                    "feature_fraction_bynode": lgbm_best_param[
                        "feature_fraction_bynode"
                    ],
                    # 'bagging_freq': lgbm_best_param["bagging_freq"],
                    # 'min_child_samples': lgbm_best_param["min_child_samples"],
                    "learning_rate": lgbm_best_param["learning_rate"],
                    "verbose": -1,
                    "device": train_on,
                    "gpu_use_dp": gpu_use_dp,
                }
                try:
                    X_train = X_train.drop(self.target_variable, axis=1)
                except Exception:
                    pass
                Dtrain = lgb.Dataset(X_train, label=Y_train)
                Dtest = lgb.Dataset(X_test, label=Y_test)
                model = lgb.train(
                    param,
                    Dtrain,
                    valid_sets=[Dtrain, Dtest],
                    valid_names=["train", "valid"],
                    early_stopping_rounds=10,
                )
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

            else:
                nb_classes = self.num_classes

                def objective(trial):
                    param = {
                        "objective": "multiclass",
                        "metric": "multi_logloss",
                        "boosting": "dart",
                        "drop_rate": trial.suggest_uniform("drop_rate", 0.1, 1.0),
                        "skip_drop": trial.suggest_uniform("skip_drop", 0.1, 1.0),
                        "num_boost_round": trial.suggest_int(
                            "num_boost_round", 100, 50000
                        ),
                        "num_class": nb_classes,
                        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1, 1e6),
                        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1, 1e6),
                        # 'max_depth': trial.suggest_int('max_depth', 2, 8), #-1
                        "num_leaves": trial.suggest_int("num_leaves", 2, 50),
                        "feature_fraction": trial.suggest_uniform(
                            "feature_fraction", 0.2, 1.0
                        ),
                        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        "feature_fraction_bynode": trial.suggest_uniform(
                            "feature_fraction_bynode", 0.4, 1.0
                        ),
                        # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        "min_gain_to_split": trial.suggest_uniform(
                            "min_gain_to_split", 0, 15
                        ),
                        "learning_rate": trial.suggest_loguniform(
                            "learning_rate", 1e-5, 0.1
                        ),
                        "verbose": -1,
                        "device": train_on,
                        "gpu_use_dp": gpu_use_dp,
                    }

                    pruning_callback = optuna.integration.LightGBMPruningCallback(
                        trial, "multi_logloss"
                    )
                    if tune_mode == "simple":
                        gbm = lgb.train(param, dtrain, verbose_eval=False)
                        preds = gbm.predict(X_test)
                        pred_labels = np.asarray([np.argmax(line) for line in preds])
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                        return matthew
                    else:
                        try:
                            result = lgb.cv(
                                param,
                                train_set=dtrain,
                                nfold=10,
                                num_boost_round=param["num_boost_round"],
                                early_stopping_rounds=10,
                                callbacks=[pruning_callback],
                                seed=42,
                                verbose_eval=False,
                            )
                            avg_result = np.mean(np.array(result["multi_logloss-mean"]))
                            # avg_result = self.meissner_cv_score(result["multi_logloss-mean"], penality_is_deducted=False)
                            return avg_result
                        except Exception:
                            avg_result = 100
                        return avg_result

                algorithm = "lgbm"
                sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
                if tune_mode == "simple":
                    study = optuna.create_study(
                        direction="maximize",
                        sampler=sampler,
                        study_name=f"{algorithm} tuning",
                    )
                else:
                    study = optuna.create_study(
                        direction="minimize",
                        sampler=sampler,
                        study_name=f"{algorithm} tuning",
                    )
                study.optimize(
                    objective,
                    n_trials=self.hyperparameter_tuning_rounds["lgbm"],
                    timeout=self.hyperparameter_tuning_max_runtime_secs["lgbm"],
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

                lgbm_best_param = study.best_trial.params
                param = {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    # 'class_weight': classes_weights,
                    "boosting": "dart",
                    "drop_rate": lgbm_best_param["drop_rate"],
                    "skip_drop": lgbm_best_param["skip_drop"],
                    "num_boost_round": lgbm_best_param["num_boost_round"],
                    "num_class": nb_classes,
                    "lambda_l1": lgbm_best_param["lambda_l1"],
                    "lambda_l2": lgbm_best_param["lambda_l2"],
                    # 'max_depth': lgbm_best_param["max_depth"],
                    "num_leaves": lgbm_best_param["num_leaves"],
                    "feature_fraction": lgbm_best_param["feature_fraction"],
                    "feature_fraction_bynode": lgbm_best_param[
                        "feature_fraction_bynode"
                    ],
                    # 'bagging_freq': lgbm_best_param["bagging_freq"],
                    # 'min_child_samples': lgbm_best_param["min_child_samples"],
                    "min_gain_to_split": lgbm_best_param["min_gain_to_split"],
                    "learning_rate": lgbm_best_param["learning_rate"],
                    "verbose": -1,
                    "device": train_on,
                    "gpu_use_dp": gpu_use_dp,
                }
                try:
                    X_train = X_train.drop(self.target_variable, axis=1)
                except Exception:
                    pass
                Dtrain = lgb.Dataset(X_train, label=Y_train)
                Dtest = lgb.Dataset(X_test, label=Y_test)
                model = lgb.train(
                    param,
                    Dtrain,
                    valid_sets=[Dtrain, Dtest],
                    valid_names=["train", "valid"],
                    early_stopping_rounds=10,
                )
                self.trained_models[f"{algorithm}"] = {}
                self.trained_models[f"{algorithm}"] = model
                del model
                _ = gc.collect()
                return self.trained_models

    def lgbm_predict(self, feat_importance=True, importance_alg="auto"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated based on SHAP values.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with LGBM")
        algorithm = "lgbm"
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            predicted_probs = model.predict(X_test)
            if self.class_problem == "binary":
                partial_probs = predicted_probs
                predicted_classes = (
                    partial_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted_probs = model.predict(X_test)
            if self.class_problem == "binary":
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "auto":
                if self.preprocess_decisions["gpu_support"]["lgbm"] == "gpu":
                    if self.class_problem == "binary":
                        self.shap_explanations(
                            model=model, test_df=X_test, cols=X_test.columns
                        )
                    else:
                        lgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
                else:
                    lgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
            elif feat_importance and importance_alg == "SHAP":
                if self.preprocess_decisions["gpu_support"]["lgbm"] == "gpu":
                    if self.class_problem == "binary":
                        self.shap_explanations(
                            model=model, test_df=X_test, cols=X_test.columns
                        )
                    else:
                        lgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
                else:
                    lgb.plot_importance(model)
                    plt.figure(figsize=(16, 12))
                    plt.show()
            elif feat_importance and importance_alg == "inbuilt":
                lgb.plot_importance(model)
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

    def sklearn_ensemble_train(self):
        """
        Trains an sklearn stacking classifier ensemble. Will automatically test different stacked classifier combinations.
        Expect very long runtimes due to CPU usage.
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task="Train sklearn ensemble")
        algorithm = "sklearn_ensemble"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()

            def objective(trial):
                ensemble_variation = trial.suggest_categorical(
                    "ensemble_variant",
                    [
                        "2_boosters",
                        "3_boosters",
                        "trees_forest",
                        "reversed_boosters",
                        "full_ensemble",
                        "full_ensemble_weighted",
                    ],
                )
                # Step 2. Setup values for the hyperparameters:
                if ensemble_variation == "2_boosters":
                    level0 = list()
                    level0.append(
                        (
                            "lr",
                            LogisticRegressionCV(
                                class_weight="balanced",
                                max_iter=500,
                                penalty="elasticnet",
                                l1_ratios=[0.1, 0.5, 0.9],
                                solver="saga",
                            ),
                        )
                    )
                    level0.append(("qdr", QuadraticDiscriminantAnalysis()))
                    level1 = LGBMClassifier()
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )
                elif ensemble_variation == "3_boosters":
                    level0 = list()
                    level0.append(("lgbm", LGBMClassifier()))
                    level0.append(("abc", AdaBoostClassifier()))
                    level1 = GradientBoostingClassifier()
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )
                elif ensemble_variation == "trees_forest":
                    level0 = list()
                    level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                    level0.append(("rdf", RandomForestClassifier(max_depth=5)))
                    level1 = GradientBoostingClassifier()
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )
                elif ensemble_variation == "reversed_boosters":
                    level0 = list()
                    level0.append(("xgb", GradientBoostingClassifier()))
                    level0.append(("lgbm", LGBMClassifier()))
                    level0.append(("qdr", QuadraticDiscriminantAnalysis()))
                    level0.append(("svc", SVC()))
                    level1 = LogisticRegression()
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )
                elif ensemble_variation == "full_ensemble":
                    level0 = list()
                    level0.append(("lgbm", LGBMClassifier()))
                    level0.append(
                        (
                            "lr",
                            LogisticRegressionCV(
                                class_weight="balanced",
                                max_iter=500,
                                penalty="elasticnet",
                                l1_ratios=[0.1, 0.5, 0.9],
                                solver="saga",
                            ),
                        )
                    )
                    level0.append(("gdc", GradientBoostingClassifier()))
                    level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                    level0.append(("abc", AdaBoostClassifier()))
                    level0.append(("qda", QuadraticDiscriminantAnalysis()))
                    level0.append(("rid", RidgeClassifier()))
                    level0.append(("svc", SVC()))
                    level0.append(("rdf", RandomForestClassifier(max_depth=5)))
                    # define meta learner model
                    level1 = GradientBoostingClassifier()
                    # define the stacking ensemble
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )
                elif ensemble_variation == "full_ensemble_weighted":
                    level0 = list()
                    level0.append(("lgbm", LGBMClassifier()))
                    level0.append(
                        (
                            "lr",
                            LogisticRegressionCV(
                                class_weight="balanced",
                                max_iter=500,
                                penalty="elasticnet",
                                l1_ratios=[0.1, 0.5, 0.9],
                                solver="saga",
                            ),
                        )
                    )
                    level0.append(("gdc", GradientBoostingClassifier()))
                    level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                    level0.append(("abc", AdaBoostClassifier()))
                    level0.append(("qda", QuadraticDiscriminantAnalysis()))
                    level0.append(("rid", RidgeClassifier(class_weight="balanced")))
                    level0.append(("svc", SVC(class_weight="balanced")))
                    level0.append(
                        (
                            "rdf",
                            RandomForestClassifier(
                                max_depth=5, class_weight="balanced"
                            ),
                        )
                    )
                    # define meta learner model
                    level1 = GradientBoostingClassifier()
                    # define the stacking ensemble
                    model = StackingClassifier(
                        estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                    )

                # Step 3: Scoring method:
                model.fit(x_train, y_train)
                predicted_probs = model.predict_proba(X_test)
                if self.class_problem == "binary":
                    self.threshold_refiner(predicted_probs, Y_test, algorithm)
                    partial_probs = np.asarray([line[1] for line in predicted_probs])
                    predicted_classes = (
                        partial_probs
                        > self.preprocess_decisions["probability_threshold"][algorithm]
                    )
                else:
                    predicted_classes = np.asarray(
                        [np.argmax(line) for line in predicted_probs]
                    )
                matthews = matthews_corrcoef(Y_test, predicted_classes)
                return matthews

            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, study_name=f"{algorithm} tuning"
            )
            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds["sklearn_ensemble"],
                timeout=self.hyperparameter_tuning_max_runtime_secs["sklearn_ensemble"],
                gc_after_trial=True,
                show_progress_bar=True,
            )
            best_variant = study.best_trial.params["ensemble_variant"]
            if best_variant == "2_boosters":
                level0 = list()
                level0.append(
                    (
                        "lr",
                        LogisticRegressionCV(
                            class_weight="balanced",
                            max_iter=500,
                            penalty="elasticnet",
                            l1_ratios=[0.1, 0.5, 0.9],
                            solver="saga",
                        ),
                    )
                )
                level0.append(("qdr", QuadraticDiscriminantAnalysis()))
                level1 = LGBMClassifier()
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )
            elif best_variant == "3_boosters":
                level0 = list()
                level0.append(("lgbm", LGBMClassifier()))
                level0.append(("abc", AdaBoostClassifier()))
                level1 = GradientBoostingClassifier()
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )
            elif best_variant == "trees_forest":
                level0 = list()
                level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                level0.append(("rdf", RandomForestClassifier(max_depth=5)))
                level1 = GradientBoostingClassifier()
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )
            elif best_variant == "reversed_boosters":
                level0 = list()
                level0.append(("xgb", GradientBoostingClassifier()))
                level0.append(("lgbm", LGBMClassifier()))
                level0.append(("qdr", QuadraticDiscriminantAnalysis()))
                level0.append(("svc", SVC()))
                level1 = LogisticRegression()
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )
            elif best_variant == "full_ensemble":
                level0 = list()
                level0.append(("lgbm", LGBMClassifier()))
                level0.append(
                    (
                        "lr",
                        LogisticRegressionCV(
                            class_weight="balanced",
                            max_iter=500,
                            penalty="elasticnet",
                            l1_ratios=[0.1, 0.5, 0.9],
                            solver="saga",
                        ),
                    )
                )
                level0.append(("gdc", GradientBoostingClassifier()))
                level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                level0.append(("abc", AdaBoostClassifier()))
                level0.append(("qda", QuadraticDiscriminantAnalysis()))
                level0.append(("rid", RidgeClassifier()))
                level0.append(("svc", SVC()))
                level0.append(("rdf", RandomForestClassifier(max_depth=5)))
                # define meta learner model
                level1 = GradientBoostingClassifier()
                # define the stacking ensemble
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )
            elif best_variant == "full_ensemble_weighted":
                level0 = list()
                level0.append(("lgbm", LGBMClassifier()))
                level0.append(
                    (
                        "lr",
                        LogisticRegressionCV(
                            class_weight="balanced",
                            max_iter=500,
                            penalty="elasticnet",
                            l1_ratios=[0.1, 0.5, 0.9],
                            solver="saga",
                        ),
                    )
                )
                level0.append(("gdc", GradientBoostingClassifier()))
                level0.append(("cart", DecisionTreeClassifier(max_depth=5)))
                level0.append(("abc", AdaBoostClassifier()))
                level0.append(("qda", QuadraticDiscriminantAnalysis()))
                level0.append(("rid", RidgeClassifier(class_weight="balanced")))
                level0.append(("svc", SVC(class_weight="balanced")))
                level0.append(
                    (
                        "rdf",
                        RandomForestClassifier(max_depth=5, class_weight="balanced"),
                    )
                )
                # define meta learner model
                level1 = GradientBoostingClassifier()
                # define the stacking ensemble
                model = StackingClassifier(
                    estimators=level0, final_estimator=level1, cv=5, n_jobs=-1
                )

            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass
            model.fit(X_train, Y_train)
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def sklearn_ensemble_predict(
        self, feat_importance=True, importance_alg="permutation"
    ):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with sklearn ensemble")
        algorithm = "sklearn_ensemble"
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )
            self.predicted_probs[f"{algorithm}"] = {}
            self.predicted_classes[f"{algorithm}"] = {}
            self.predicted_probs[f"{algorithm}"] = predicted_probs
            self.predicted_classes[f"{algorithm}"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in partial_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                        explainer="kernel",
                    )
                except Exception:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test,
                        cols=X_test.columns,
                        explainer="kernel",
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model,
                    X_test,
                    Y_test.astype(int),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
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

    def ngboost_train(self, tune_mode="accurate"):  # noqa: C901
        """
        Trains an Ngboost regressor.
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task="Train Ngboost")
        algorithm = "ngboost"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            classes_weights = class_weight.compute_sample_weight(
                class_weight="balanced", y=Y_train
            )
            x_train, y_train = self.get_hyperparameter_tuning_sample_df()
            classes_weights_sample = class_weight.compute_sample_weight(
                class_weight="balanced", y=y_train
            )
            nb_classes = k_categorical(Y_train.nunique())
            try:
                Y_train = Y_train.astype(int)
                y_train = y_train.astype(int)
                Y_test = Y_test.astype(int)
            except Exception:
                y_train = np.int(y_train)
                Y_test = np.int(Y_test)

            def objective(trial):
                base_learner_choice = trial.suggest_categorical(
                    "base_learner",
                    [
                        "DecTree_depth2",
                        "DecTree_depth5",
                        "DecTree_depthNone",
                        "GradientBoost_depth2",
                        "GradientBoost_depth5",
                    ],
                )
                if base_learner_choice == "DecTree_depth2":
                    base_learner_choice = DecisionTreeRegressor(max_depth=2)
                elif base_learner_choice == "DecTree_depth5":
                    base_learner_choice = DecisionTreeRegressor(max_depth=5)
                elif base_learner_choice == "DecTree_depthNone":
                    base_learner_choice = DecisionTreeRegressor(max_depth=None)
                elif base_learner_choice == "GradientBoost_depth2":
                    base_learner_choice = GradientBoostingRegressor(
                        max_depth=2,
                        n_estimators=1000,
                        n_iter_no_change=10,
                        random_state=42,
                    )
                elif base_learner_choice == "GradientBoost_depth5":
                    base_learner_choice = GradientBoostingRegressor(
                        max_depth=5,
                        n_estimators=10000,
                        n_iter_no_change=10,
                        random_state=42,
                    )

                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 2, 50000),
                    "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.4, 1.0),
                    "learning_rate": trial.suggest_loguniform(
                        "learning_rate", 1e-3, 0.1
                    ),
                }
                if tune_mode == "simple":
                    model = NGBClassifier(
                        n_estimators=param["n_estimators"],
                        minibatch_frac=param["minibatch_frac"],
                        Dist=nb_classes,
                        Base=base_learner_choice,
                        learning_rate=param["learning_rate"],
                    ).fit(
                        x_train,
                        y_train,
                        X_val=X_test,
                        Y_val=Y_test,
                        sample_weight=classes_weights_sample,
                        early_stopping_rounds=10,
                    )
                    pred_labels = model.predict(X_test)
                    try:
                        matthew = matthews_corrcoef(Y_test, pred_labels)
                    except Exception:
                        matthew = 0
                    return matthew
                else:
                    model = NGBClassifier(
                        n_estimators=param["n_estimators"],
                        minibatch_frac=param["minibatch_frac"],
                        Dist=nb_classes,
                        Base=base_learner_choice,
                        learning_rate=param["learning_rate"],
                        random_state=42,
                    )
                    try:
                        scores = cross_val_score(
                            model,
                            x_train,
                            y_train,
                            cv=10,
                            scoring="f1_weighted",
                            fit_params={
                                "X_val": X_test,
                                "Y_val": Y_test,
                                "sample_weight": classes_weights_sample,
                                "early_stopping_rounds": 10,
                            },
                        )
                        mae = np.mean(scores)
                    except Exception:
                        mae = 0
                    return mae

            algorithm = "ngboost"
            if tune_mode == "simple":
                study = optuna.create_study(
                    direction="maximize", study_name=f"{algorithm} tuning"
                )
            else:
                study = optuna.create_study(
                    direction="maximize", study_name=f"{algorithm} tuning"
                )
            study.optimize(
                objective,
                n_trials=self.hyperparameter_tuning_rounds["ngboost"],
                timeout=self.hyperparameter_tuning_max_runtime_secs["ngboost"],
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

            lgbm_best_param = study.best_trial.params

            if lgbm_best_param["base_learner"] == "DecTree_depth2":
                base_learner_choice = DecisionTreeRegressor(max_depth=2)
            elif lgbm_best_param["base_learner"] == "DecTree_depth5":
                base_learner_choice = DecisionTreeRegressor(max_depth=5)
            elif lgbm_best_param["base_learner"] == "DecTree_depthNone":
                base_learner_choice = DecisionTreeRegressor(max_depth=None)
            elif lgbm_best_param["base_learner"] == "GradientBoost_depth2":
                base_learner_choice = GradientBoostingRegressor(
                    max_depth=2, n_estimators=1000, n_iter_no_change=10, random_state=42
                )
            elif lgbm_best_param["base_learner"] == "GradientBoost_depth5":
                base_learner_choice = GradientBoostingRegressor(
                    max_depth=5,
                    n_estimators=10000,
                    n_iter_no_change=10,
                    random_state=42,
                )
            param = {
                "Dist": nb_classes,
                "n_estimators": lgbm_best_param["n_estimators"],
                "minibatch_frac": lgbm_best_param["minibatch_frac"],
                "learning_rate": lgbm_best_param["learning_rate"],
            }
            try:
                X_train = X_train.drop(self.target_variable, axis=1)
            except Exception:
                pass
            model = NGBClassifier(
                n_estimators=param["n_estimators"],
                minibatch_frac=param["minibatch_frac"],
                Dist=nb_classes,
                Base=base_learner_choice,
                learning_rate=param["learning_rate"],
                random_state=42,
            ).fit(
                X_train,
                Y_train,
                X_val=X_test,
                Y_val=Y_test,
                sample_weight=classes_weights,
                early_stopping_rounds=10,
            )
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def ngboost_predict(self, feat_importance=True, importance_alg="SHAP"):
        """
        Loads the pretrained model from the class itself and predicts on new data.
        :param feat_importance: Set True, if feature importance shall be calculated.
        :param importance_alg: Chose 'permutation' (recommended on CPU) or 'SHAP' (recommended when model uses
        GPU acceleration). (Default: 'permutation')
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task="Predict with Ngboost")
        algorithm = "ngboost"
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_probs = np.asarray([line[1] for line in partial_probs])
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
                predicted_classes = predicted_classes.astype(int)
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in partial_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            partial_probs = model.predict_proba(X_test)
            predicted_probs = np.asarray([line[1] for line in partial_probs])
            if self.class_problem == "binary":
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

            if feat_importance and importance_alg == "SHAP":
                self.runtime_warnings(warn_about="shap_cpu")
                try:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test.sample(10000, random_state=42),
                        cols=X_test.columns,
                        explainer="kernel",
                    )
                except Exception:
                    self.shap_explanations(
                        model=model,
                        test_df=X_test,
                        cols=X_test.columns,
                        explainer="kernel",
                    )
            elif feat_importance and importance_alg == "permutation":
                result = permutation_importance(
                    model,
                    X_test,
                    Y_test.astype(int),
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                permutation_importances = pd.Series(
                    result.importances_mean, index=X_test.columns
                )
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

    def deesc_train(self):  # noqa: C901
        """
        Trains an DEESC classifier
        :return: Updates class attributes by its predictions.
        """
        self.get_current_timestamp(task="Train DEESC")
        algorithm = "deesc"
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            from e2eml.model_utils.deesc_classifier import DEESCClassifier

            model = DEESCClassifier(X_train, X_test, Y_train, Y_test)
            model.fit()
            self.trained_models[f"{algorithm}"] = {}
            self.trained_models[f"{algorithm}"] = model
            del model
            _ = gc.collect()
            return self.trained_models

    def deesc_predict(self):
        self.get_current_timestamp(task="Predict with DEESC")
        algorithm = "deesc"
        model = self.trained_models[f"{algorithm}"]
        if self.prediction_mode:
            X_test = self.dataframe
            partial_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                predicted_classes = (
                    partial_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
                predicted_classes = predicted_classes.astype(int)
                predicted_probs = partial_probs
            else:
                predicted_probs = partial_probs
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in partial_probs]
                )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            predicted_probs = model.predict_proba(X_test)
            if self.class_problem == "binary":
                self.threshold_refiner(predicted_probs, Y_test, algorithm)
                predicted_classes = (
                    predicted_probs
                    > self.preprocess_decisions["probability_threshold"][algorithm]
                )
            else:
                predicted_classes = np.asarray(
                    [np.argmax(line) for line in predicted_probs]
                )

        self.predicted_probs[f"{algorithm}"] = {}
        self.predicted_classes[f"{algorithm}"] = {}
        self.predicted_probs[f"{algorithm}"] = predicted_probs
        self.predicted_classes[f"{algorithm}"] = predicted_classes
        del model
        _ = gc.collect()
        return self.predicted_probs
