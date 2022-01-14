import math
import typing
from collections import Counter

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy.stats as ss
from boostaroota import BoostARoota
from lightgbm import LGBMClassifier
from numpy import dot
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class DEESCClassifier:
    """
    Base implementation for delta embedding extended stacking classifier.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train,
        Y_test,
        learning_rate=0.3,
        random_state=1000,
        use_long_warmup: bool = False,
        auto_select_features: bool = False,
        max_tuning_time_h: float = 2,
    ):
        if isinstance(Y_train, pd.Series):
            pass
        else:
            Y_train = pd.Series(Y_train)
        (
            self.X_train,
            self.X_train_2nd_layer,
            self.Y_train,
            self.Y_train_2nd_layer,
        ) = train_test_split(
            X_train, Y_train, test_size=0.70, random_state=random_state
        )
        self.X_test = X_test
        self.Y_test = Y_test
        self.original_columns = [str(col) for col in self.X_train.columns.to_list()]
        self.numerical_columns = self.X_train.select_dtypes(
            include=np.number
        ).columns.tolist()
        self.selected_feats = []  # type: typing.List[str]
        self.selected_feats_2nd_layer = []  # type: typing.List[str]
        self.unique_classes = self.Y_train.unique()
        self.delta_cols_mapping = {}  # type: typing.Dict[str, float]
        self.corr_cols_created = {}  # type: typing.Dict[float, list]
        self.cosine_cols_created = {}  # type: typing.Dict[float, list]
        self.class_corr_cols_created = []  # type: typing.List[str]
        self.class_cosine_cols_created = []  # type: typing.List[str]
        self.cosine_features_selected = {}  # type: typing.Dict[float, list]
        self.correlation_features_selected = {}  # type: typing.Dict[float, list]
        for ind_class in self.unique_classes:
            self.corr_cols_created[ind_class] = []
            self.cosine_cols_created[ind_class] = []
        self.meta_data = {}  # type: typing.Dict[float, pd.DataFrame]
        self.learning_rate = learning_rate
        self.weights_array = self.learning_rate * np.arange(-100, 100)
        self.n_jobs = 1
        self.max_tuning_time_h = max_tuning_time_h
        self.use_long_warmup = use_long_warmup
        self.auto_select_features = auto_select_features
        self.stats_to_consider = ["min", "25%", "50%", "75%", "mean", "max"]
        self.ext_stats_to_consider = [
            "min",
            "1%",
            "5%",
            "10%",
            "20%",
            "25%",
            "50%",
            "75%",
            "80%",
            "90%",
            "95%",
            "99%",
            "mean",
            "max",
        ]
        self.df_shapes = {
            "train_N": self.X_train.shape[0],
            "train_D": self.X_train.shape[1],
            "test_N": self.X_test.shape[0],
            "test_D": self.X_test.shape[1],
        }
        self.delta_cols_created = []  # type: typing.List[str]
        self.base_weights = {}  # type: typing.Dict[float, np.ndarray]
        for ind_class in self.unique_classes:
            self.base_weights[ind_class] = {}
            for stat in self.stats_to_consider:
                self.base_weights[ind_class][stat] = np.ones(self.df_shapes["train_D"])

        self.best_weights = {}  # type: typing.Dict[float, np.ndarray]
        for ind_class in self.unique_classes:
            self.best_weights[ind_class] = {}
            for stat in self.stats_to_consider:
                self.best_weights[ind_class][stat] = np.ones(self.df_shapes["train_D"])

        self.best_score = 0
        self.best_train_score = 0
        self.random_state = random_state
        self.lgbm_model = LGBMClassifier()
        self.ridge_model = RidgeClassifier()
        self.svm_model = SVC()
        self.lgbm_2nd_meta_model = None
        self.lgbm_cols_created = []  # type: typing.List[str]
        self.ridge_cols_created = []  # type: typing.List[str]
        self.svm_cols_created = []  # type: typing.List[str]
        self.lgbm_predictions_non_delta = []  # type: typing.List[str]
        self.ridge_predictions_non_delta = []  # type: typing.List[str]
        self.svm_predictions_non_delta = []  # type: typing.List[str]
        self.nearest_class_predictions = None
        self.class_correlation_predictions = None
        self.class_cosine_predictions = None
        self.lgbm_predictions = None
        self.ridge_predictions = None
        self.svm_predictions = None
        self.lgbm_final_predictions = []  # type: ignore
        self.scaler = MinMaxScaler()
        self.pca_encoder = PCA()
        self.pca_components = []  # type: typing.List[str]

    def add_weights(
        self, delta_matrix, weight: float, col_idx: int, lookup_class, stat
    ):
        """
        Takes a numpy array and adds weights to the specified column_index.
        :param weight: Float. Column value will be multiplied by this value.
        :param col_idx: Integer providing numeric column index.
        :return: Returns modified numpy array.
        """
        temp_weights = self.base_weights[lookup_class][stat]
        temp_weights = np.nan_to_num(temp_weights, neginf=0)
        temp_weights[col_idx : col_idx + 1] = (
            temp_weights[col_idx : col_idx + 1] * weight
        )
        delta_matrix = delta_matrix * temp_weights
        return delta_matrix, temp_weights

    def get_meta_data(self):
        """
        Loops through all classes of a DataFrame, slices it and stores the metadata in a class self.meta_data attribute.
        :return: Updates class attribute.
        """
        for ind_class in self.unique_classes:
            X_train_class_only = self.X_train[self.original_columns].iloc[
                np.where(self.Y_train == ind_class)[0]
            ]
            meta_data = X_train_class_only.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.10,
                    0.20,
                    0.25,
                    0.5,
                    0.75,
                    0.80,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            self.meta_data[ind_class] = meta_data

    def delta_to_stat(self, dataframe: pd.DataFrame, stat: str, lookup_class):
        """
        Takes the desired stat as string and the class to look up for. Takes the DataFrame and calculates the delta
        between each row and the class stat.
        :param stat: Str. Must be present in pd.DataFrame.describe() (i.e. "mean", "min", "max")
        :param lookup_class: Class that has been used as key in self.meta_data attribute
        :return: Returns numpy array with all deltas of shape equal to original DataFrame
        """
        delta_to_stat = (
            dataframe.iloc[:, :].values
            - self.meta_data[lookup_class].loc[stat, :].values
        )
        return delta_to_stat

    def get_matthews(self, y_original, preds):
        matthews = matthews_corrcoef(y_original, preds)
        return matthews

    def corr2_coeff_rowwise(self, A, B):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        # Sum of squares across rows
        ssA = (A_mA ** 2).sum(1)
        ssB = (B_mB ** 2).sum(1)
        # Finally get corr coeff
        return np.einsum("ij,ij->i", A_mA, B_mB) / np.sqrt(ssA * ssB)

    def cosine_similarity(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def conditional_entropy(self, x, y):
        # entropy of x given y
        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x, y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y / p_xy)
        return entropy

    def theil_u(self, x, y):
        s_xy = self.conditional_entropy(x, y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
        s_x = ss.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x

    def theil_u_based_feature_selection(self):
        # transform numerical columns into bins to categorize them
        X_train_temp = self.X_train.copy()
        for col in self.X_train[self.numerical_columns].columns.to_list():
            X_train_temp[str(col)] = pd.qcut(
                X_train_temp[col].rank(method="first"), 10, labels=False
            )
        # get Theil U values
        X_train_temp["target"] = self.Y_train
        theilu = pd.DataFrame(index=["target"], columns=X_train_temp.columns)
        columns = X_train_temp.columns.to_list()
        for j in range(0, len(columns)):
            u = self.theil_u(
                X_train_temp["target"].tolist(), X_train_temp[columns[j]].tolist()
            )
            theilu.loc[:, columns[j]] = u
        theilu.fillna(value=np.nan, inplace=True)

        # feature selection based on Theil U
        for col in theilu.columns.to_list():
            if theilu.loc[:, col].values[0] >= 0.005 and col != "target":
                self.selected_feats.append(col)

    def feature_selection(self):
        br = BoostARoota(metric="mlogloss")
        br.fit(self.X_train, self.Y_train)
        selected = br.keep_vars_.values.tolist()
        selected = [str(col) for col in selected]
        self.selected_feats = selected

    def last_layer_feature_selection(self):
        br = BoostARoota(metric="mlogloss")
        br.fit(self.X_train_2nd_layer, self.Y_train_2nd_layer)
        selected = br.keep_vars_
        selected = selected.values.tolist()
        selected = [str(col) for col in selected]
        self.selected_feats_2nd_layer = selected

    def fit_scale_data(self, dataframe: pd.DataFrame):
        standard_scaler = MinMaxScaler()
        columns = dataframe.columns
        dataframe = pd.DataFrame(
            standard_scaler.fit_transform(dataframe), columns=columns
        )
        self.scaler = standard_scaler
        return dataframe

    def predict_scale_data(self, dataframe: pd.DataFrame):
        standard_scaler = self.scaler
        columns = dataframe.columns
        dataframe = pd.DataFrame(standard_scaler.transform(dataframe), columns=columns)
        return dataframe

    def fit_delta_embeddings_pca(self, dataframe: pd.DataFrame):
        dataframe = self.fit_scale_data(dataframe)
        pca = PCA(
            n_components=0.95,
            random_state=self.random_state,
        )
        train_comps = pca.fit_transform(dataframe)
        new_cols = [f"PCA_{i}" for i in range(train_comps.shape[1])]

        self.pca_encoder = pca
        self.pca_components = new_cols

    def predict_delta_embeddings_pca(self, dataframe: pd.DataFrame):
        dataframe = self.predict_scale_data(dataframe)
        pca = self.pca_encoder
        new_cols = self.pca_components
        components = pca.transform(dataframe)
        reduced_df = pd.DataFrame(components, columns=new_cols)
        return reduced_df

    def create_baseline_score(self):
        self.get_meta_data()
        # initial filling & base line score
        for lookup_class in self.unique_classes:
            all_non_stat_delta_train = np.ones(
                (self.df_shapes["train_N"], self.df_shapes["train_D"]), dtype="float64"
            )
            all_non_stat_delta_test = np.ones(
                (self.df_shapes["test_N"], self.df_shapes["test_D"]), dtype="float64"
            )
            for stat in self.stats_to_consider:
                stat_delta = self.delta_to_stat(
                    self.X_train[self.original_columns], stat, lookup_class
                )
                stat_delta = np.nan_to_num(stat_delta, neginf=0)
                all_non_stat_delta_train += stat_delta
                total_delta_matrix = stat_delta + all_non_stat_delta_train
                total_delta_df = pd.DataFrame(total_delta_matrix)
                self.X_train[
                    f"total_delta_sum_to_class_{lookup_class}"
                ] = total_delta_df.sum(axis=1).values

                stat_delta = self.delta_to_stat(
                    self.X_test[self.original_columns], stat, lookup_class
                )
                all_non_stat_delta_test += stat_delta
                total_delta_matrix = stat_delta + all_non_stat_delta_test
                total_delta_df = pd.DataFrame(total_delta_matrix)
                self.X_test[
                    f"total_delta_sum_to_class_{lookup_class}"
                ] = total_delta_df.sum(axis=1).values
                self.delta_cols_created.append(
                    f"total_delta_sum_to_class_{lookup_class}"
                )
                self.delta_cols_mapping[
                    f"total_delta_sum_to_class_{lookup_class}"
                ] = lookup_class

        self.X_train["supervised_distance_nearest_class"] = (
            self.X_train[self.delta_cols_created].idxmin(axis=1).values
        )
        self.X_test["supervised_distance_nearest_class"] = (
            self.X_test[self.delta_cols_created].idxmin(axis=1).values
        )

        self.X_train["supervised_distance_nearest_class"] = self.X_train[
            "supervised_distance_nearest_class"
        ].map(self.delta_cols_mapping)

        self.X_test["supervised_distance_nearest_class"] = self.X_test[
            "supervised_distance_nearest_class"
        ].map(self.delta_cols_mapping)

        self.best_score = self.get_matthews(
            self.Y_test, self.X_test["supervised_distance_nearest_class"]
        )
        print(f"Baseline score with default weights is {self.best_score}.")

    def warmup_weights(self):
        # optimization loop
        trials = 0
        total_trials = (
            len(self.unique_classes)
            * len(self.weights_array)
            * self.df_shapes["train_D"]
        )
        print(f"Start warmup. Total trials will be: {total_trials}:")
        for lookup_class in self.unique_classes:
            print(f"Start loop for class {lookup_class}...")
            for stat in ["min"]:
                print(f"Start loop for stat {stat}...")
                # print(f"Start loop for stat {stat}")
                non_sel = self.stats_to_consider.copy()
                non_sel.remove(stat)
                for col in range(self.df_shapes["train_D"]):
                    # print(f"Start loop for col {col}")
                    for weight in self.weights_array:
                        stat_delta = self.delta_to_stat(
                            self.X_train[self.original_columns], stat, lookup_class
                        )
                        stat_delta, temp_weights = self.add_weights(
                            stat_delta, weight, col, lookup_class, stat
                        )
                        stat_delta = np.nan_to_num(stat_delta, neginf=0)

                        all_non_stat_delta = np.ones(
                            (self.df_shapes["train_N"], self.df_shapes["train_D"]),
                            dtype="float64",
                        )
                        for non_sel_stat in non_sel:
                            non_sel_stat_delta = self.delta_to_stat(
                                self.X_train[self.original_columns],
                                non_sel_stat,
                                lookup_class,
                            )
                            # non_sel_stat_delta, temp_weights_non = self.add_weights(non_sel_stat_delta, 1, col, lookup_class, non_sel_stat)
                            non_sel_stat_delta = np.nan_to_num(
                                non_sel_stat_delta, neginf=0
                            )
                            all_non_stat_delta += non_sel_stat_delta
                        total_delta_matrix = stat_delta + all_non_stat_delta
                        total_delta_df = pd.DataFrame(total_delta_matrix)
                        self.X_train[
                            f"total_delta_sum_to_class_{lookup_class}"
                        ] = total_delta_df.sum(axis=1).values

                        stat_delta = self.delta_to_stat(
                            self.X_test[self.original_columns], stat, lookup_class
                        )
                        stat_delta, temp_weights_test = self.add_weights(
                            stat_delta, weight, col, lookup_class, stat
                        )
                        stat_delta = np.nan_to_num(stat_delta, neginf=0)
                        all_non_stat_delta = np.ones(
                            (self.df_shapes["test_N"], self.df_shapes["test_D"]),
                            dtype="float64",
                        )
                        all_non_stat_delta = np.nan_to_num(all_non_stat_delta, neginf=0)
                        for non_sel_stat in non_sel:
                            non_sel_stat_delta = self.delta_to_stat(
                                self.X_test[self.original_columns],
                                non_sel_stat,
                                lookup_class,
                            )
                            # non_sel_stat_delta, temp_weights_non = self.add_weights(non_sel_stat_delta, 1, col, lookup_class, non_sel_stat)
                            non_sel_stat_delta = np.nan_to_num(
                                non_sel_stat_delta, neginf=0
                            )
                            all_non_stat_delta += non_sel_stat_delta

                        total_delta_matrix = stat_delta + all_non_stat_delta
                        total_delta_df = pd.DataFrame(total_delta_matrix)
                        self.X_test[
                            f"total_delta_sum_to_class_{lookup_class}"
                        ] = total_delta_df.sum(axis=1).values

                        self.X_train["supervised_distance_nearest_class"] = (
                            self.X_train[self.delta_cols_created].idxmin(axis=1).values
                        )
                        self.X_test["supervised_distance_nearest_class"] = (
                            self.X_test[self.delta_cols_created].idxmin(axis=1).values
                        )

                        self.X_train[
                            "supervised_distance_nearest_class"
                        ] = self.X_train["supervised_distance_nearest_class"].map(
                            self.delta_cols_mapping
                        )

                        self.X_test["supervised_distance_nearest_class"] = self.X_test[
                            "supervised_distance_nearest_class"
                        ].map(self.delta_cols_mapping)

                        trials += 1
                        progress = round((trials / total_trials), 2)

                        matthews_train = self.get_matthews(
                            self.Y_train,
                            self.X_train["supervised_distance_nearest_class"],
                        )
                        matthews = self.get_matthews(
                            self.Y_test,
                            self.X_test["supervised_distance_nearest_class"],
                        )

                        if matthews_train > self.best_train_score:
                            print(
                                f"New best train score after trial {trials} is {matthews_train}."
                            )
                            self.best_train_score = matthews_train

                        if matthews > self.best_score:
                            print(
                                f"New best score after trial {trials} is {matthews}. Best training set score is {self.best_train_score}."
                            )
                            self.best_weights[lookup_class][stat] = temp_weights
                            self.best_score = matthews

                        if trials % 1000 == 0:
                            print(
                                f"Reached progress of {progress * 100}% with current best score of {self.best_score}."
                            )

    def fit_delta_weights(self):
        if self.auto_select_features:
            self.feature_selection()
            if len(self.selected_feats) > 0:
                self.X_train = self.X_train[self.selected_feats].copy()
                self.X_test = self.X_test[self.selected_feats].copy()
                self.original_columns = self.selected_feats
                self.df_shapes = {
                    "train_N": self.X_train.shape[0],
                    "train_D": self.X_train.shape[1],
                    "test_N": self.X_test.shape[0],
                    "test_D": self.X_test.shape[1],
                }
                self.base_weights = {}  # type: typing.Dict[float, np.ndarray]
                for ind_class in self.unique_classes:
                    self.base_weights[ind_class] = {}
                    for stat in self.stats_to_consider:
                        self.base_weights[ind_class][stat] = np.ones(
                            self.df_shapes["train_D"]
                        )

                self.best_weights = {}  # type: typing.Dict[float, np.ndarray]
                for ind_class in self.unique_classes:
                    self.best_weights[ind_class] = {}
                    for stat in self.stats_to_consider:
                        self.best_weights[ind_class][stat] = np.ones(
                            self.df_shapes["train_D"]
                        )
            else:
                pass

        self.create_baseline_score()
        if self.use_long_warmup:
            self.warmup_weights()

        algorithm = "delta_classifier"
        max_trials = (
            len(self.unique_classes)
            * len(self.stats_to_consider)
            * self.df_shapes["train_D"]
            * 100
        )
        nb_trials = int(max_trials / 10)
        warm_up_trials = int(max_trials / 100)
        if warm_up_trials < 1:
            warm_up_trials = 10
        else:
            pass
        print(
            f"The maximum number of trials planned is {nb_trials} with {warm_up_trials} random warmup trials at start."
        )

        def objective(trial):
            lookup_class = trial.suggest_categorical(
                "lookup_class", [cls for cls in self.unique_classes]
            )
            weight = trial.suggest_uniform("weight", -100, 100)
            col = trial.suggest_int(
                "col", 0, len(self.X_train[self.original_columns].columns)
            )
            stat = trial.suggest_categorical("stat", self.stats_to_consider)
            non_sel = self.stats_to_consider.copy()
            non_sel.remove(str(stat))
            stat_delta = self.delta_to_stat(
                self.X_train[self.original_columns], stat, lookup_class
            )
            stat_delta, temp_weights = self.add_weights(
                stat_delta, weight, col, lookup_class, stat
            )
            stat_delta = np.nan_to_num(stat_delta, neginf=0)
            all_non_stat_delta = np.ones(
                (self.df_shapes["train_N"], self.df_shapes["train_D"]), dtype="float64"
            )
            for non_sel_stat in non_sel:
                non_sel_stat_delta = self.delta_to_stat(
                    self.X_train[self.original_columns], non_sel_stat, lookup_class
                )
                # non_sel_stat_delta, temp_weights_non = self.add_weights(non_sel_stat_delta, 1, col, lookup_class, non_sel_stat)
                non_sel_stat_delta = np.nan_to_num(non_sel_stat_delta, neginf=0)
                all_non_stat_delta += non_sel_stat_delta
            total_delta_matrix = stat_delta + all_non_stat_delta
            total_delta_df = pd.DataFrame(total_delta_matrix)
            self.X_train[
                f"total_delta_sum_to_class_{lookup_class}"
            ] = total_delta_df.sum(axis=1).values
            stat_delta = self.delta_to_stat(
                self.X_test[self.original_columns], stat, lookup_class
            )
            stat_delta, temp_weights_test = self.add_weights(
                stat_delta, weight, col, lookup_class, stat
            )
            stat_delta = np.nan_to_num(stat_delta, neginf=0)
            all_non_stat_delta = np.ones(
                (self.df_shapes["test_N"], self.df_shapes["test_D"]), dtype="float64"
            )
            all_non_stat_delta = np.nan_to_num(all_non_stat_delta, neginf=0)
            for non_sel_stat in non_sel:
                non_sel_stat_delta = self.delta_to_stat(
                    self.X_test[self.original_columns], non_sel_stat, lookup_class
                )
                # non_sel_stat_delta, temp_weights_non = self.add_weights(non_sel_stat_delta, 1, col, lookup_class, non_sel_stat)
                non_sel_stat_delta = np.nan_to_num(non_sel_stat_delta, neginf=0)
                all_non_stat_delta += non_sel_stat_delta
            total_delta_matrix = stat_delta + all_non_stat_delta
            total_delta_df = pd.DataFrame(total_delta_matrix)
            self.X_test[
                f"total_delta_sum_to_class_{lookup_class}"
            ] = total_delta_df.sum(axis=1).values
            self.X_train["supervised_distance_nearest_class"] = (
                self.X_train[self.delta_cols_created].idxmin(axis=1).values
            )
            self.X_test["supervised_distance_nearest_class"] = (
                self.X_test[self.delta_cols_created].idxmin(axis=1).values
            )
            self.X_train["supervised_distance_nearest_class"] = self.X_train[
                "supervised_distance_nearest_class"
            ].map(self.delta_cols_mapping)
            self.X_test["supervised_distance_nearest_class"] = self.X_test[
                "supervised_distance_nearest_class"
            ].map(self.delta_cols_mapping)
            matthews_train = self.get_matthews(
                self.Y_train, self.X_train["supervised_distance_nearest_class"]
            )
            matthews = self.get_matthews(
                self.Y_test, self.X_test["supervised_distance_nearest_class"]
            )
            if matthews_train > self.best_train_score:
                self.best_train_score = matthews_train
            if matthews > self.best_score:
                self.best_weights[lookup_class][stat] = temp_weights
                self.best_score = matthews
            return matthews

        sampler = optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=20, seed=42
        )
        study = optuna.create_study(
            direction="maximize", study_name=f"{algorithm} tuning", sampler=sampler
        )

        study.optimize(
            objective,
            n_trials=nb_trials,
            timeout=self.max_tuning_time_h * 60 * 60,
            gc_after_trial=True,
            show_progress_bar=True,
            n_jobs=self.n_jobs,
        )

        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()
        except ZeroDivisionError:
            print(
                "Plotting of hyperparameter performances failed. This usually implicates an error during training."
            )

        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
        except ZeroDivisionError:
            print(
                "Plotting of hyperparameter performances failed. This usually implicates an error during training."
            )

    def delta_correlations(
        self, dataframe: pd.DataFrame, delta_array, stat, lookup_class
    ):
        df1 = dataframe.rank(axis=1)
        dataframe[f"corr_{lookup_class}_{stat}"] = df1.corrwith(
            pd.Series(delta_array, index=df1.columns).rank(), axis=1
        )
        self.corr_cols_created[lookup_class].append(f"corr_{lookup_class}_{stat}")
        return dataframe[f"corr_{lookup_class}_{stat}"]

    def fit_cosine_similarity(self):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        cosine_cols_wo_class = []
        cosine_cols_to_full_name = {}
        for lookup_class in self.unique_classes:
            for stat in self.ext_stats_to_consider:
                X_train[
                    f"cosine_similarity_{lookup_class}_{stat}"
                ] = self.cosine_similarity(
                    X_train[self.original_columns].values,
                    self.meta_data[lookup_class].loc[stat, :].values,
                )
                X_test[
                    f"cosine_similarity_{lookup_class}_{stat}"
                ] = self.cosine_similarity(
                    X_test[self.original_columns].values,
                    self.meta_data[lookup_class].loc[stat, :].values,
                )
                self.cosine_cols_created[lookup_class].append(
                    f"cosine_similarity_{lookup_class}_{stat}"
                )
                if f"cosine_similarity_{stat}" not in cosine_cols_wo_class:
                    cosine_cols_wo_class.append(f"cosine_similarity_{stat}")
                cosine_cols_to_full_name[
                    f"cosine_similarity_{lookup_class}_{stat}"
                ] = f"cosine_similarity_{stat}"

        def objective(trial):
            param = {}
            # we go crazy here and allow Optuna to chose the columns to consider for clustering
            for col in self.ext_stats_to_consider:
                param[col] = trial.suggest_int(col, 0, 1)

            temp_features_dic = {}
            for lookup_class in self.unique_classes:
                for k, v in param.items():
                    if v == 1:
                        temp_features_dic[
                            lookup_class
                        ] = f"cosine_similarity_{lookup_class}_{k}"
                    else:
                        pass

            temp_mean_cols_mapping = {}
            temp_mean_cols = []
            for lookup_class in self.unique_classes:
                temp_features = [
                    v for k, v in temp_features_dic.items() if k == lookup_class
                ]
                X_train[f"test_score_{lookup_class}"] = X_train[temp_features].mean(
                    axis=1
                )
                X_test[f"test_score_{lookup_class}"] = X_test[temp_features].mean(
                    axis=1
                )
                temp_mean_cols_mapping[f"test_score_{lookup_class}"] = lookup_class
                temp_mean_cols.append(f"test_score_{lookup_class}")

            X_train["final_test_score"] = X_train[temp_mean_cols].idxmax(axis=1).values
            X_train["final_test_score"] = X_train["final_test_score"].map(
                temp_mean_cols_mapping
            )
            X_test["final_test_score"] = X_test[temp_mean_cols].idxmax(axis=1).values
            X_test["final_test_score"] = X_test["final_test_score"].map(
                temp_mean_cols_mapping
            )

            X_train["final_test_score"] = X_train["final_test_score"].fillna(0)
            X_test["final_test_score"] = X_test["final_test_score"].fillna(0)
            matthews_train = matthews_corrcoef(
                self.Y_train, X_train["final_test_score"]
            )
            matthews_test = matthews_corrcoef(self.Y_test, X_test["final_test_score"])
            matthews = matthews_train - (matthews_train - matthews_test) ** 2
            return matthews

        sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name="tune_cosine"
        )
        study.optimize(
            objective,
            n_trials=500,
            gc_after_trial=True,
            show_progress_bar=True,
            timeout=1 * 60 * 60,
        )

        final_cosine_features_dic = {}
        for lookup_class in self.unique_classes:
            final_cosine_features_dic[lookup_class] = []
            for k, v in study.best_trial.params.items():
                if v == 1:
                    final_cosine_features_dic[lookup_class].append(
                        f"cosine_similarity_{lookup_class}_{k}"
                    )
                else:
                    pass

        self.cosine_features_selected = final_cosine_features_dic

    def fit_correlations(self):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        corr_cols_wo_class = []
        corr_cols_to_full_name = {}
        for lookup_class in self.unique_classes:
            for stat in self.ext_stats_to_consider:
                X_train[f"corr_{lookup_class}_{stat}"] = self.delta_correlations(
                    X_train[self.original_columns],
                    self.meta_data[lookup_class].loc[stat, :].values,
                    stat,
                    lookup_class,
                )
                X_test[f"corr_{lookup_class}_{stat}"] = self.delta_correlations(
                    X_test[self.original_columns],
                    self.meta_data[lookup_class].loc[stat, :].values,
                    stat,
                    lookup_class,
                )
                self.corr_cols_created[lookup_class].append(
                    f"corr_{lookup_class}_{stat}"
                )
                if f"corr_{lookup_class}_{stat}" not in corr_cols_wo_class:
                    corr_cols_wo_class.append(f"corr_{lookup_class}_{stat}")
                corr_cols_to_full_name[f"corr_{lookup_class}_{stat}"] = f"corr_{stat}"

        def objective(trial):
            param = {}
            # we go crazy here and allow Optuna to chose the columns to consider for clustering
            for col in self.ext_stats_to_consider:
                param[col] = trial.suggest_int(col, 0, 1)

            temp_features_dic = {}
            for lookup_class in self.unique_classes:
                for k, v in param.items():
                    if v == 1:
                        temp_features_dic[lookup_class] = f"corr_{lookup_class}_{k}"
                    else:
                        pass

            temp_mean_cols_mapping = {}
            temp_mean_cols = []
            for lookup_class in self.unique_classes:
                temp_features = [
                    v for k, v in temp_features_dic.items() if k == lookup_class
                ]
                X_train[f"test_score_{lookup_class}"] = X_train[temp_features].mean(
                    axis=1
                )
                X_test[f"test_score_{lookup_class}"] = X_test[temp_features].mean(
                    axis=1
                )
                temp_mean_cols_mapping[f"test_score_{lookup_class}"] = lookup_class
                temp_mean_cols.append(f"test_score_{lookup_class}")

            X_train["final_test_score"] = X_train[temp_mean_cols].idxmax(axis=1).values
            X_train["final_test_score"] = X_train["final_test_score"].map(
                temp_mean_cols_mapping
            )
            X_test["final_test_score"] = X_test[temp_mean_cols].idxmax(axis=1).values
            X_test["final_test_score"] = X_test["final_test_score"].map(
                temp_mean_cols_mapping
            )
            X_train["final_test_score"] = X_train["final_test_score"].fillna(0)
            X_test["final_test_score"] = X_test["final_test_score"].fillna(0)
            matthews_train = matthews_corrcoef(
                self.Y_train, X_train["final_test_score"]
            )
            matthews_test = matthews_corrcoef(self.Y_test, X_test["final_test_score"])
            matthews = matthews_train - (matthews_train - matthews_test) ** 2
            return matthews

        sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name="tune_correlations"
        )
        study.optimize(
            objective,
            n_trials=500,
            gc_after_trial=True,
            show_progress_bar=True,
            timeout=1 * 60 * 60,
        )

        final_corr_features_dic = {}
        for lookup_class in self.unique_classes:
            final_corr_features_dic[lookup_class] = []
            for k, v in study.best_trial.params.items():
                if v == 1:
                    final_corr_features_dic[lookup_class].append(
                        f"cosine_similarity_{lookup_class}_{k}"
                    )
                else:
                    pass

        self.correlation_features_selected = final_corr_features_dic

    def fit_base_models(self):
        X_train = self.X_train.copy()

        total_delta_dfs = []
        for lookup_class in self.unique_classes:
            total_delta_matrix = np.ones(
                (
                    X_train[self.original_columns].shape[0],
                    X_train[self.original_columns].shape[1],
                ),
                dtype="float64",
            )
            stats_dfs = []
            for stat in self.stats_to_consider:
                stat_delta = self.delta_to_stat(
                    X_train[self.original_columns], stat, lookup_class
                )

                temp_weights = self.best_weights[lookup_class][stat]
                temp_weights = np.nan_to_num(temp_weights, neginf=0)
                stat_delta = stat_delta * temp_weights
                stat_delta = np.nan_to_num(stat_delta, neginf=0)
                total_delta_matrix += stat_delta
                # get sums of all deltas per class
                total_delta_df = pd.DataFrame(total_delta_matrix)
                stats_dfs.append(total_delta_df)
            for i in range(len(stats_dfs)):
                if i == 0:
                    stats_df = stats_dfs[i]
                else:
                    stats_df = stats_df.merge(
                        stats_dfs[i], left_index=True, right_index=True
                    )
            total_delta_dfs.append(stats_df)

        for i in range(len(total_delta_dfs)):
            if i == 0:
                embeddings_df = total_delta_dfs[i]
            else:
                embeddings_df = embeddings_df.merge(
                    total_delta_dfs[i], left_index=True, right_index=True
                )

        N = embeddings_df.shape[0]
        train_sample = embeddings_df.sample(int(N / 2), random_state=self.random_state)
        # val_sample = embeddings_df[(~embeddings_df.index.isin(train_sample.index))]
        embeddings_df.columns = range(embeddings_df.shape[1])
        self.fit_delta_embeddings_pca(train_sample)
        embeddings_df = self.predict_delta_embeddings_pca(embeddings_df)
        lgbm_model = LGBMClassifier(random_state=self.random_state)
        lgbm_model.fit(embeddings_df, self.Y_train)
        self.lgbm_model = lgbm_model

        ridge_model = RidgeClassifier(random_state=self.random_state)
        ridge_model.fit(embeddings_df, self.Y_train)
        self.ridge_model = ridge_model

        svm_model = SVC(random_state=self.random_state, probability=True)
        svm_model.fit(embeddings_df, self.Y_train)
        self.svm_model = svm_model

    def fit_1st_layer(self):
        self.fit_delta_weights()
        self.fit_cosine_similarity()
        self.fit_correlations()
        self.fit_base_models()

    def predict_cosine_similarity(self, dataframe: pd.DataFrame):
        for lookup_class in self.unique_classes:
            features_to_use = self.cosine_features_selected[lookup_class]
            for stat in self.ext_stats_to_consider:
                dataframe[
                    f"cosine_similarity_{lookup_class}_{stat}"
                ] = self.cosine_similarity(
                    dataframe[self.original_columns].values,
                    self.meta_data[lookup_class].loc[stat, :].values,
                )
                self.cosine_cols_created[lookup_class].append(
                    f"cosine_similarity_{lookup_class}_{stat}"
                )
            dataframe[f"mean_stat_delta_cosine_class_{lookup_class}"] = (
                dataframe[features_to_use].mean(axis=1).values
            )
            self.class_cosine_cols_created.append(
                f"mean_stat_delta_cosine_class_{lookup_class}"
            )

            self.delta_cols_mapping[
                f"mean_stat_delta_cosine_class_{lookup_class}"
            ] = lookup_class

        # get final predictions based on highest cosine to delta
        dataframe["cosine_strongest_class"] = (
            dataframe[self.class_cosine_cols_created].idxmax(axis=1).values
        )
        dataframe["cosine_strongest_class"] = dataframe["cosine_strongest_class"].map(
            self.delta_cols_mapping
        )

        self.class_cosine_predictions = dataframe["cosine_strongest_class"]
        return dataframe

    def predict_correlations(self, dataframe: pd.DataFrame):
        for lookup_class in self.unique_classes:
            features_to_use = self.correlation_features_selected[lookup_class]
            for stat in self.ext_stats_to_consider:
                dataframe[f"corr_{lookup_class}_{stat}"] = self.delta_correlations(
                    dataframe[self.original_columns],
                    self.meta_data[lookup_class].loc[stat, :].values,
                    stat,
                    lookup_class,
                )
                self.corr_cols_created[lookup_class].append(
                    f"corr_{lookup_class}_{stat}"
                )
            dataframe[f"mean_stat_delta_correlation_class_{lookup_class}"] = (
                dataframe[features_to_use].mean(axis=1).values
            )
            self.class_corr_cols_created.append(
                f"mean_stat_delta_correlation_class_{lookup_class}"
            )

            self.delta_cols_mapping[
                f"mean_stat_delta_correlation_class_{lookup_class}"
            ] = lookup_class

        # get final predictions based on highest cosine to delta
        dataframe["correlations_strongest_class"] = (
            dataframe[self.class_corr_cols_created].idxmax(axis=1).values
        )
        dataframe["correlations_strongest_class"] = dataframe[
            "correlations_strongest_class"
        ].map(self.delta_cols_mapping)

        self.class_correlation_predictions = dataframe["correlations_strongest_class"]
        return dataframe

    def get_classes_from_probs(self, probs):
        if len(self.unique_classes) == 2:
            predicted_probs = np.asarray([line[1] for line in probs])
            predicted_classes = predicted_probs > 0.5
        else:
            predicted_probs = probs
            predicted_classes = np.asarray(
                [np.argmax(line) for line in predicted_probs]
            )
        return predicted_classes

    def predict_base_models(self, dataframe: pd.DataFrame):
        total_delta_dfs = []
        for lookup_class in self.unique_classes:
            # predict loop
            total_delta_matrix = np.ones(
                (
                    dataframe[self.original_columns].shape[0],
                    dataframe[self.original_columns].shape[1],
                ),
                dtype="float64",
            )
            stats_dfs = []
            for stat in self.stats_to_consider:
                stat_delta = self.delta_to_stat(
                    dataframe[self.original_columns], stat, lookup_class
                )

                temp_weights = self.best_weights[lookup_class][stat]
                temp_weights = np.nan_to_num(temp_weights, neginf=0)
                stat_delta = stat_delta * temp_weights
                stat_delta = np.nan_to_num(stat_delta, neginf=0)
                total_delta_matrix += stat_delta
                # get sums of all deltas per class
                total_delta_df = pd.DataFrame(total_delta_matrix)
                stats_dfs.append(total_delta_df)
            for i in range(len(stats_dfs)):
                if i == 0:
                    stats_df = stats_dfs[i]
                else:
                    stats_df = stats_df.merge(
                        stats_dfs[i], left_index=True, right_index=True
                    )
            total_delta_dfs.append(stats_df)

        for i in range(len(total_delta_dfs)):
            if i == 0:
                embeddings_df = total_delta_dfs[i]
            else:
                embeddings_df = embeddings_df.merge(
                    total_delta_dfs[i], left_index=True, right_index=True
                )
        embeddings_df.columns = range(embeddings_df.shape[1])
        embeddings_df = self.predict_delta_embeddings_pca(embeddings_df)
        lgbm_model = self.lgbm_model
        temp_preds = lgbm_model.predict_proba(embeddings_df)
        for lookup_class in range(self.unique_classes.shape[0]):
            dataframe[f"lgbm_predictions_proba_{lookup_class}"] = temp_preds[
                :, lookup_class
            ].reshape(-1, 1)
        dataframe["lgbm_predictions"] = self.get_classes_from_probs(temp_preds)
        self.lgbm_predictions = dataframe["lgbm_predictions"]
        self.lgbm_cols_created.append("lgbm_predictions")

        ridge_model = self.ridge_model
        dataframe["ridge_predictions"] = ridge_model.predict(embeddings_df)
        self.ridge_predictions = dataframe["ridge_predictions"]
        self.ridge_cols_created.append("ridge_predictions")

        svm_model = self.svm_model
        temp_preds = svm_model.predict_proba(embeddings_df)
        for lookup_class in range(self.unique_classes.shape[0]):
            dataframe[f"svm_predictions_proba_{lookup_class}"] = temp_preds[
                :, lookup_class
            ].reshape(-1, 1)
        dataframe["svm_predictions"] = self.get_classes_from_probs(temp_preds)
        self.svm_predictions = dataframe["svm_predictions"]
        self.svm_cols_created.append("svm_predictions")
        return dataframe

    def predict_1st_layer(self, dataframe: pd.DataFrame):
        dataframe = dataframe[self.original_columns].copy()

        dataframe = self.predict_cosine_similarity(dataframe)
        dataframe = self.predict_correlations(dataframe)
        dataframe = self.predict_base_models(dataframe)

        for lookup_class in self.unique_classes:
            total_delta_matrix = np.ones(
                (
                    dataframe[self.original_columns].shape[0],
                    dataframe[self.original_columns].shape[1],
                ),
                dtype="float64",
            )
            for stat in self.stats_to_consider:
                stat_delta = self.delta_to_stat(
                    dataframe[self.original_columns], stat, lookup_class
                )

                temp_weights = self.best_weights[lookup_class][stat]
                temp_weights = np.nan_to_num(temp_weights, neginf=0)
                stat_delta = stat_delta * temp_weights
                stat_delta = np.nan_to_num(stat_delta, neginf=0)
                total_delta_matrix += stat_delta
            # get sums of all deltas per class
            total_delta_df = pd.DataFrame(total_delta_matrix)
            dataframe[f"total_delta_sum_to_class_{lookup_class}"] = total_delta_df.sum(
                axis=1
            ).values

        # get final predictions based on delta distances and weights
        dataframe["supervised_distance_nearest_class"] = (
            dataframe[self.delta_cols_created].idxmin(axis=1).values
        )
        dataframe["supervised_distance_nearest_class"] = dataframe[
            "supervised_distance_nearest_class"
        ].map(self.delta_cols_mapping)

        self.nearest_class_predictions = dataframe["supervised_distance_nearest_class"]

        final_predictors = [
            "correlations_strongest_class",
            "cosine_strongest_class",
            "supervised_distance_nearest_class",
            # "lgbm_predictions",
            # "ridge_predictions",
            # "svm_predictions"
        ]

        dataframe = dataframe.fillna(0)
        dataframe["delta_based_predictions"] = dataframe[final_predictors].mode(axis=1)[
            0
        ]
        return dataframe

    def fit(self):
        self.fit_1st_layer()
        self.last_layer_feature_selection()
        dataframe = self.predict_1st_layer(self.X_train_2nd_layer)

        X_train = dataframe[self.selected_feats_2nd_layer].copy()
        Y_train = self.Y_train_2nd_layer.copy()
        X_train["target"] = self.Y_train_2nd_layer

        length = len(X_train.index)

        x_train = X_train.sample(
            int(length / 10 * 8), random_state=self.random_state
        ).copy()
        y_train = x_train["target"]

        X_test = X_train[(~X_train.index.isin(x_train.index))].copy()
        Y_test = X_test["target"]

        x_train = x_train.drop("target", axis=1)
        X_test = X_test.drop("target", axis=1)
        X_train = X_train.drop("target", axis=1)

        dtrain = lgb.Dataset(x_train, label=y_train)

        if len(self.unique_classes) == 2:
            # weights_for_lgb = self.calc_scale_pos_weight()

            def objective(trial):
                param = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    # 'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1e3),
                    "num_boost_round": trial.suggest_int("num_boost_round", 100, 50000),
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
                    "gpu_use_dp": False,
                }

                pruning_callback = optuna.integration.LightGBMPruningCallback(
                    trial, "binary_logloss"
                )
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
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=f"{algorithm} tuning",
            )

            study.optimize(
                objective,
                n_trials=500,
                timeout=2 * 60 * 60,
                gc_after_trial=True,
                show_progress_bar=True,
            )

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
                "feature_fraction_bynode": lgbm_best_param["feature_fraction_bynode"],
                # 'bagging_freq': lgbm_best_param["bagging_freq"],
                # 'min_child_samples': lgbm_best_param["min_child_samples"],
                "learning_rate": lgbm_best_param["learning_rate"],
                "verbose": -1,
                "gpu_use_dp": False,
            }
            Dtrain = lgb.Dataset(X_train, label=Y_train)
            Dtest = lgb.Dataset(X_test, label=Y_test)
            model = lgb.train(
                param,
                Dtrain,
                valid_sets=[Dtrain, Dtest],
                valid_names=["train", "valid"],
                early_stopping_rounds=10,
            )
            self.lgbm_2nd_meta_model = model
        else:
            nb_classes = len(self.unique_classes)

            def objective(trial):
                param = {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "boosting": "dart",
                    "drop_rate": trial.suggest_uniform("drop_rate", 0.1, 1.0),
                    "skip_drop": trial.suggest_uniform("skip_drop", 0.1, 1.0),
                    "num_boost_round": trial.suggest_int("num_boost_round", 100, 50000),
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
                    "gpu_use_dp": False,
                }

                pruning_callback = optuna.integration.LightGBMPruningCallback(
                    trial, "multi_logloss"
                )
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
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=f"{algorithm} tuning",
            )
            study.optimize(
                objective,
                n_trials=500,
                timeout=2 * 60 * 60,
                gc_after_trial=True,
                show_progress_bar=True,
            )

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
                "feature_fraction_bynode": lgbm_best_param["feature_fraction_bynode"],
                # 'bagging_freq': lgbm_best_param["bagging_freq"],
                # 'min_child_samples': lgbm_best_param["min_child_samples"],
                "min_gain_to_split": lgbm_best_param["min_gain_to_split"],
                "learning_rate": lgbm_best_param["learning_rate"],
                "verbose": -1,
                "gpu_use_dp": False,
            }
            Dtrain = lgb.Dataset(X_train, label=Y_train)
            Dtest = lgb.Dataset(X_test, label=Y_test)
            model = lgb.train(
                param,
                Dtrain,
                valid_sets=[Dtrain, Dtest],
                valid_names=["train", "valid"],
                early_stopping_rounds=10,
            )
            self.lgbm_2nd_meta_model = model

    def predict(self, dataframe: pd.DataFrame, return_df=False):
        dataframe = self.predict_1st_layer(dataframe)
        model = self.lgbm_2nd_meta_model
        y_hat = model.predict(dataframe[self.selected_feats_2nd_layer])  # type:ignore
        self.lgbm_final_predictions = y_hat  # type:ignore
        if return_df:
            for lookup_class in range(self.unique_classes.shape[0]):
                dataframe[
                    f"lgbm_final_prediction_proba_{lookup_class}"
                ] = self.lgbm_final_predictions[
                    :, lookup_class
                ].reshape(  # type:ignore
                    -1, 1
                )  # type: ignore
            return dataframe
        else:
            return y_hat

    def predict_proba(self, dataframe: pd.DataFrame, return_df=False):
        dataframe = self.predict_1st_layer(dataframe)
        model = self.lgbm_2nd_meta_model
        y_hat = model.predict(dataframe[self.selected_feats_2nd_layer])  # type:ignore
        self.lgbm_final_predictions = y_hat
        if return_df:
            for lookup_class in range(self.unique_classes.shape[0]):
                dataframe[
                    f"lgbm_final_prediction_proba_{lookup_class}"
                ] = self.lgbm_final_predictions[
                    :, lookup_class
                ].reshape(  # type:ignore
                    -1, 1
                )  # type:ignore
            return dataframe
        else:
            return y_hat
