import numpy as np
import optuna
import pandas as pd


class AutoDataSlicer:
    def __init__(
        self,
        df,
        target_col,
        objective="binary",
        use_percentages_as_metric=True,
        study_direction="maximize",
        study_name="dataframe slicer",
        random_state=1000,
        n_trials=None,
        remove_high_cardinality=True,
        max_cat_col_cardinality=50,
        supress_granular_prints=True,
        slice_one_category_only=True,
        pre_remove_slices=False,
        timeout=3 * 60 * 60,
    ):
        self.df = df
        self.target_col = target_col
        self.objective = objective
        self.use_percentages_as_metric = use_percentages_as_metric
        self.study_direction = study_direction
        self.study_name = study_name
        self.random_state = random_state
        if isinstance(n_trials, list):
            self.n_trials = n_trials
        else:
            self.n_trials = [500, 100, 100]
        self.remove_high_cardinality = remove_high_cardinality
        self.max_cat_col_cardinality = max_cat_col_cardinality
        self.supress_granular_prints = supress_granular_prints
        self.timeout = timeout
        self.slice_one_category_only = slice_one_category_only
        self.pre_remove_slices = (pre_remove_slices,)
        self.study = None
        self.cat_columns = None
        self.num_col_list = None
        self.numerical_meta_data = None
        self.categorical_filters = {}
        self.trial_counter = 0
        self.best_segments = None
        self.best_score = 1

    def get_categorical_columns(self):
        self.cat_columns = self.df.select_dtypes(include=["object"]).columns.to_list()
        if self.n_trials[0] > 0 and len(self.cat_columns) == 0:
            self.n_trials[0] = 0
            self.n_trials[2] = 0
            print(
                "Categorical columns could not be identified. Corrected n_trials attribute."
            )

    def get_numerical_columns(self):
        num_dtypes = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        self.num_col_list = []
        for vartype in num_dtypes:
            num_cols = self.df.select_dtypes(include=[vartype]).columns
            for col in num_cols:
                self.num_col_list.append(col)

        if self.n_trials[1] > 0 and len(self.num_col_list) == 0:
            self.n_trials[1] = 0
            self.n_trials[2] = 0
            print(
                "Numerical columns could not be identified. Corrected n_trials attribute."
            )

    def get_numerical_meta_data(self):
        self.numerical_meta_data = self.df[self.num_col_list].describe()

    def create_categorical_filters(self):
        for col in self.cat_columns:
            self.categorical_filters[col] = {}
            options = self.df[col].unique()
            for option in options:
                self.categorical_filters[col][option] = 1

    def remove_cardinal_cols(self):
        uniques = self.df[self.cat_columns].nunique()
        for col in self.cat_columns:
            if uniques[col] > self.max_cat_col_cardinality:
                self.cat_columns.remove(col)
                self.df = self.df.drop(col, axis=1)
                print(f"Removed column {col} due to high cardinality.")

    def pre_slice_dataframe(self):  # noqa: B007
        # build slices based on categories
        for col, val_list in self.categorical_filters.items():  # noqa: B007
            for cat, value in self.categorical_filters[col].items():  # noqa: B007
                temp_df = self.df[(self.df[col] == cat)]
                try:
                    has_true = (
                        temp_df[self.target_col]
                        .value_counts(normalize=self.use_percentages_as_metric)
                        .loc[True]
                    )
                    # exclude each category that does not include any representatives of target class
                    if not has_true:
                        self.df = self.df[(self.df[col] != cat)].copy()
                        print(
                            f"Excluded from feature {col} sub category {cat} due to missing data points of target class."
                        )
                except KeyError:
                    self.df = self.df[(self.df[col] != cat)].copy()
                    print(
                        f"Excluded from feature {col} sub category {cat} due to missing data points of target class."
                    )

    def pre_slice_numerical_dataframe(self):
        for col in self.num_col_list:
            perc_25th = self.numerical_meta_data[col]["25%"]
            perc_75th = self.numerical_meta_data[col]["75%"]
            temp_df = self.df[(self.df[col] < perc_25th)]
            try:
                has_true = (
                    temp_df[self.target_col]
                    .value_counts(normalize=self.use_percentages_as_metric)
                    .loc[True]
                )
                # exclude each category that does not include any representatives of target class
                if not has_true:
                    self.df = self.df[(self.df[col] >= perc_25th)].copy()
                    print(
                        f"Excluded from feature {col} values below {perc_25th} due to missing data points of target class."
                    )
            except KeyError:
                self.df = self.df[(self.df[col] >= perc_25th)].copy()
                print(
                    f"Excluded from feature {col} values below {perc_25th} due to missing data points of target class."
                )

            temp_df = self.df[(self.df[col] > perc_75th)]
            try:
                has_true = (
                    temp_df[self.target_col]
                    .value_counts(normalize=self.use_percentages_as_metric)
                    .loc[True]
                )
                # exclude each category that does not include any representatives of target class
                if not has_true:
                    self.df = self.df[(self.df[col] <= perc_75th)].copy()
                    print(
                        f"Excluded from feature {col} values above {perc_75th} due to missing data points of target class."
                    )
            except KeyError:
                self.df = self.df[(self.df[col] <= perc_75th)].copy()
                print(
                    f"Excluded from feature {col} values above {perc_75th} due to missing data points of target class."
                )

    def append_results(self, segment_dict, numerical_segment_dict, score, df_len, mode):
        if isinstance(self.best_segments, pd.DataFrame):
            new_res = pd.DataFrame(
                {
                    "score": score,
                    "segments_chosen": [segment_dict],
                    "numerical_segments_chosen": [numerical_segment_dict],
                    "no_rows_considered": df_len,
                    "mode": mode,
                }
            )
            self.best_segments = pd.concat([self.best_segments, new_res])
            self.best_segments = self.best_segments.sort_values(
                by=["score"], ascending=[False]
            )
        else:
            self.best_segments = pd.DataFrame(
                {
                    "score": score,
                    "segments_chosen": [segment_dict],
                    "numerical_segments_chosen": [numerical_segment_dict],
                    "no_rows_considered": df_len,
                    "mode": mode,
                }
            )

    def get_best_slice(self, min_rows_considered=20):
        best_df = self.best_segments
        best_df = best_df[(best_df["no_rows_considered"] >= min_rows_considered)]
        mode = best_df.head(1)["mode"].values[0]
        temp_df = self.df.copy()

        if mode in ["Categorical only", "Categorical and numerical"]:
            top_cat_slice_param = best_df["segments_chosen"]
            for col, val_list in (top_cat_slice_param.values[0]).items():  # noqa: B007
                for cat, value in (top_cat_slice_param.values[0][col]).items():
                    if value == 0:
                        temp_df = temp_df[(temp_df[col] != cat)]
                    else:
                        pass

        if mode in ["Numerical only", "Categorical and numerical"]:
            top_num_slice_param = best_df.head(1)["numerical_segments_chosen"]
            col = top_num_slice_param.values[0]["chosen_col"]
            min_val = top_num_slice_param.values[0][col]["min"]
            max_val = top_num_slice_param.values[0][col]["max"]
            temp_df = temp_df[(temp_df[col] >= min_val) & (temp_df[col] < max_val)]

        return temp_df

    def slice_df_binary(self):  # noqa: C901
        def binary_objective(trial):  # this does not allow additional parameters!

            self.trial_counter += 1
            print(f"Start trial {self.trial_counter}...")

            if self.trial_counter <= self.n_trials[0]:
                mode = "Categorical only"
            elif self.trial_counter > self.n_trials[0] and self.trial_counter <= np.sum(
                np.array(self.n_trials[:2])
            ):
                mode = "Numerical only"
            elif self.trial_counter > np.sum(np.array(self.n_trials[:2])):
                mode = "Categorical and numerical"

            if self.trial_counter <= self.n_trials[0] or self.trial_counter > np.sum(
                np.array(self.n_trials[:2])
            ):
                # store all slice choices
                param = {}

                if self.slice_one_category_only:
                    col = trial.suggest_categorical("chosen_cat_col", self.cat_columns)
                    param[col] = {}
                    for cat, value in self.categorical_filters[  # noqa: B007
                        col
                    ].items():  # noqa: B007
                        param[col][cat] = trial.suggest_int(f"{col}_{cat}", 0, 1)

                else:

                    # build slices based on categories
                    for col, val_list in self.categorical_filters.items():  # noqa: B007
                        param[col] = {}
                        for cat, value in self.categorical_filters[  # noqa: B007
                            col
                        ].items():  # noqa: B007
                            param[col][cat] = trial.suggest_int(f"{col}_{cat}", 0, 1)

                temp_df = self.df.copy()

                # slice df based on selected filters
                for col, val_list in param.items():  # noqa: B007
                    for cat, value in param[col].items():  # noqa: B007
                        if value == 0:
                            temp_df = temp_df[(temp_df[col] != cat)]
                        else:
                            pass

            if self.trial_counter > self.n_trials[0] and self.trial_counter <= np.sum(
                np.array(self.n_trials[:2])
            ):
                temp_df = self.df.copy()
                param = {}

            if self.trial_counter > self.n_trials[0]:
                # here starts sclicing of numerical columns
                numerical_param = {}
                numerical_param["chosen_col"] = trial.suggest_categorical(
                    "chosen_col", self.num_col_list
                )

                col = numerical_param["chosen_col"]
                numerical_param[col] = {}
                min_val = self.numerical_meta_data[col]["min"]
                perc_75th = self.numerical_meta_data[col]["75%"]
                max_val = self.numerical_meta_data[col]["max"]
                numerical_param[col]["min"] = trial.suggest_uniform(
                    f"{col}_min", min_val, perc_75th
                )
                numerical_param[col]["max"] = trial.suggest_uniform(
                    f"{col}_max", numerical_param[col]["min"], max_val
                )

                # slice df based on selected filters
                min_val = numerical_param[col]["min"]
                max_val = numerical_param[col]["max"]
                temp_df = temp_df[(temp_df[col] >= min_val) & (temp_df[col] < max_val)]
            else:
                numerical_param = {}

            df_len = len(temp_df.index)

            try:
                # print(temp_df[self.target_col].value_counts(normalize=self.use_percentages_as_metric))
                bin_true = (
                    temp_df[self.target_col]
                    .value_counts(normalize=self.use_percentages_as_metric)
                    .loc[True]
                )
                self.append_results(param, numerical_param, bin_true, df_len, mode)
            except KeyError:
                bin_true = 0

            if self.supress_granular_prints:
                print(
                    f"""Trial {self.trial_counter} finished with score {bin_true} with df length of {df_len}."""
                )

            if self.trial_counter == 0 or bin_true > self.best_score:
                self.best_score = bin_true

            del temp_df
            return bin_true, df_len

        if self.supress_granular_prints:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # create filters
        self.get_categorical_columns()

        if self.remove_high_cardinality:
            self.remove_cardinal_cols()
            self.get_categorical_columns()
        self.create_categorical_filters()

        if self.pre_remove_slices:
            self.pre_slice_dataframe()
        self.get_numerical_columns()
        self.get_numerical_meta_data()

        if self.pre_remove_slices:
            self.pre_slice_numerical_dataframe()
            self.get_numerical_meta_data()

        sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=self.random_state, warn_independent_sampling=False
        )
        study = optuna.create_study(
            directions=[self.study_direction, "maximize"],
            sampler=sampler,
            study_name=self.study_name,
        )
        study.optimize(
            binary_objective,
            n_trials=np.sum(np.array(self.n_trials)),
            gc_after_trial=True,
            show_progress_bar=True,
            timeout=self.timeout,
        )
        self.study = study

    def plot_study_optimization_history(self, target=lambda t: t.values[0]):
        try:
            fig = optuna.visualization.plot_optimization_history(
                self.study, target=target
            )
            fig.show()
        except ZeroDivisionError:
            pass

    def plot_study_parameter_importances(self, target=lambda t: t.values[0]):
        try:
            fig = optuna.visualization.plot_param_importances(self.study, target=target)
            fig.show()
        except ZeroDivisionError:
            pass


def analyse_results(
    holdout_df,
    y_true,
    y_hat,
    search_for="correct_pred",
    n_trials=None,
    random_state=1000,
    max_cat_col_cardinality=50,
):
    """
    Takes a Pandas DataFrame and tries to find slices with high quotas of
    search_for param. During optimization it will try to optimize for slices as big as possible as well.
    If only a certain class is of interest, just pass the slice containing this class.
    :param holdout_df: Pandas DataFrame without target or prediction column.
    :param y_true: Pandas Series holding real target values. Index must match holdout_df
    :param y_hat: Pandas Series holding predicted target values. Index must match holdout_df
    :param search_for: Expects a string of values "correct_pred", "incorrect_pred".
    :param n_trials: Expects a list containing three integers to define the number of trials for each part of the
    routine (default: [500, 100, 100]). Index 0 will slice using categories only. Index 1 will slice numerical
    columns only. Index 2 will slice categorical and numerical columns.
    :param random_state: Integer specifying random state to use.
    :param max_cat_col_cardinality: Expects integer. Columns with specified cardinality or higher will be removed.
    :return: Returns AutoDataSlicer object.
    """
    holdout_df["true_target"] = y_true
    holdout_df["pred_target"] = y_hat

    if search_for == "correct_pred":
        conditions = [holdout_df["true_target"] == holdout_df["pred_target"]]
    elif search_for == "incorrect_pred":
        conditions = [holdout_df["true_target"] != holdout_df["pred_target"]]
    else:
        print("No matching search_for param defined. Fallback to true_positives.")
        conditions = [holdout_df["true_target"] == holdout_df["pred_target"]]
    choices = [True]
    holdout_df[search_for] = np.select(conditions, choices, default=False)
    study_maker_fl = AutoDataSlicer(
        holdout_df,
        target_col=search_for,
        objective="binary",
        study_direction="maximize",
        study_name="dataframe slicer",
        random_state=random_state,
        n_trials=n_trials,
        remove_high_cardinality=True,
        max_cat_col_cardinality=max_cat_col_cardinality,
    )
    study_maker_fl.slice_df_binary()
    return study_maker_fl
