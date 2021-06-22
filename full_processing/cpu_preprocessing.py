from pandas.core.common import SettingWithCopyWarning
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from category_encoders import *
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from boostaroota import BoostARoota
import gc
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class PreProcessing:
    """
    The preprocessing base class expects a Pandas dataframe and the target variable at least.
    Date columns and categorical columns can be passed as lists additionally for respective preprocessing.
    A unique identifier (i.e. an ID column) can be passed as well to preserve this information for later processing.
    """
    def __init__(self, dataframe, target_variable, date_columns=None, categorical_columns=None, num_columns=None,
                 unique_identifier=None, selected_feats=None, cat_encoded=None, cat_encoder_model=None,
                 prediction_mode=False, preprocess_decisions=None, trained_model=None):
        self.dataframe = dataframe
        self.dataframe.columns = self.dataframe.columns.astype(str)
        try:
            if dataframe[target_variable].nunique() > 2:
                self.class_problem = 'multiclass'
                self.num_classes = dataframe[target_variable].nunique()
            elif dataframe[target_variable].nunique() == 2:
                self.class_problem = 'binary'
                self.num_classes = 2
            else:
                pass
        except Exception:
            if len(np.unique(np.array(target_variable))) > 2:
                self.class_problem = 'multiclass'
                self.num_classes = len(np.unique(np.array(target_variable)))
            elif len(np.unique(np.array(target_variable))) == 2:
                self.class_problem = 'binary'
                self.num_classes = 2
            else:
                pass
        self.date_columns = date_columns
        self.date_columns_created = None
        self.categorical_columns = categorical_columns
        self.cat_columns_encoded = None
        self.unique_identifier = unique_identifier
        self.target_variable = target_variable
        self.labels_encoded = False
        self.new_sin_cos_col_names = None
        self.df_dict = None
        # store chosen preprocessing settings
        if not preprocess_decisions:
            self.preprocess_decisions = {}
        else:
            self.preprocess_decisions = preprocess_decisions
        self.selected_feats = selected_feats
        self.cat_encoded = cat_encoded
        self.cat_encoder_model = cat_encoder_model
        self.data_scaled = False
        if not trained_model:
            self.trained_models = {}
        else:
            self.trained_models = trained_model
        self.optuna_studies = {}
        self.predicted_classes = {}
        self.predicted_probs = {}
        self.evaluation_scores = {}
        self.xg_boost_regression = None
        self.xgboost_objective = None
        self.prediction_mode = prediction_mode
        self.excluded = None
        self.num_dtypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not num_columns:
            num_col_list = []
            for vartype in self.num_dtypes:
                num_cols = self.dataframe.select_dtypes(include=[vartype]).columns
                for col in num_cols:
                    num_col_list.append(col)
            self.num_columns = num_col_list
        else:
            self.num_columns = num_columns

    def __repr__(self):
        return f"Central data class holding all information like dataframes, " \
               f"columns of certain data types, saved models and predictions." \
               f"Current target variable:'{self.target_variable}'"

    def __str__(self):
        return f"Current target: {self.target_variable}"

    def wrap_test_train_to_dict(self, X_train, X_test, Y_train, Y_test):
        """
        Takes in X_train & X_test parts and updates the class instance dictionary.
        :param X_train: Dataframe
        :param X_test: Dataframe
        :param Y_train: Pandas Series
        :param Y_test: Pandas Series
        :return: Class dictionary
        """
        if self.prediction_mode:
            pass
        else:
            self.df_dict = {'X_train': X_train,
                            'X_test': X_test,
                            'Y_train': Y_train,
                            'Y_test': Y_test}
            return self.df_dict

    def unpack_test_train_dict(self):
        """
        This function takes in the class dictionary holding test and train split and unpacks it.
        :return: X_train, X_test as dataframes. Y_train, Y_test as Pandas series.
        """
        X_train, X_test, Y_train, Y_test = self.df_dict["X_train"], self.df_dict["X_test"], self.df_dict["Y_train"], self.df_dict["Y_test"]
        return X_train, X_test, Y_train, Y_test

    def label_encoder_decoder(self, target, mode='fit', direction='encode'):
        if direction == 'encode' and mode == 'fit':
            le = preprocessing.LabelEncoder()
            le.fit(target)
            le.transform(target)
            self.preprocess_decisions["label_encoder"] = le
            self.labels_encoded = True
        elif direction == 'encode' and mode == 'predict':
            le = self.preprocess_decisions["label_encoder"]
            le.transform(target)
        else:
            le = self.preprocess_decisions["label_encoder"]
            le.inverse_transform(target)
        return target

    def data_scaling(self, scaling='minmax'):
        """
        Scales the data using the chosen scaling algorithm.
        :param scaling: Chose 'minmax'.
        :return:
        """
        if self.prediction_mode:
            dataframe_cols = self.dataframe.columns
            if scaling == 'minmax':
                scaler = self.preprocess_decisions["scaling"]
                scaler.fit(self.dataframe)
                scaler.transform(self.dataframe)
            self.dataframe = pd.DataFrame(self.dataframe, columns=dataframe_cols)
            self.data_scaled = True
            return self.dataframe, self.data_scaled
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train_cols = X_train.columns
            if scaling == 'minmax':
                scaler = MinMaxScaler()
                scaler.fit(X_train)
                scaler.transform(X_train)
                scaler.transform(X_test)
                self.preprocess_decisions["scaling"] = scaler
            X_train = pd.DataFrame(X_train, columns=X_train_cols)
            X_test = pd.DataFrame(X_test, columns=X_train_cols)
            self.data_scaled = True
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.data_scaled, self.preprocess_decisions

    def train_test_split(self, how='cross', split_by_col=None, split_date=None, train_size=0.80):
        """
        This method splits the dataframe either as a simple or as a time split.
        :param how: 'cross' for cross validation, 'time' for time validation.
        :param split_by_col: Chose column to be used for split. For time validation only.
        :param split_date: Chose exact date to split. Test dataframe is equal or greater than provided date.
        :param train_size: Chose how much percentage the train dataframe will have. For cross validation only.
        :return: X_train, X_test, Y_train, Y_test
        """
        if self.prediction_mode:
            pass
        elif how == 'cross':
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(self.dataframe,
                                                                                self.dataframe[self.target_variable],
                                                                                train_size=train_size)
            try:
                Y_train = Y_train.astype(float)
                Y_test = Y_test.astype(float)
            except Exception:
                Y_train = self.label_encoder_decoder(Y_train, mode='fit', direction='encode')
                Y_test = self.label_encoder_decoder(Y_train, mode='predict', direction='encode')
            del X_train[self.target_variable]
            del X_test[self.target_variable]
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif how == 'time':
            X_train = self.dataframe[(self.dataframe[split_by_col] < split_date)]
            X_test = self.dataframe[(self.dataframe[split_by_col] >= split_date)]
            Y_train = X_train[self.target_variable]
            Y_test = X_test[self.target_variable]
            del X_train[self.target_variable]
            del X_test[self.target_variable]
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        else:
            raise Exception("Please provide a split method.")

    def data_binning(self, nb_bins=10):
        """
        Takes numerical columns and splits them into desired number of bins. Bins will be attached as
        new columns to the dataframe.
        :param nb_bins: Takes a positive integer.
        :return:
        """
        def binning_on_data(dataframe, cols_to_bin=None):
            num_columns = cols_to_bin.select_dtypes(include=[vartype]).columns
            for col in num_columns:
                dataframe[str(col)+'_binned'] = pd.cut(dataframe[col], bins=nb_bins, labels=False)
                self.new_sin_cos_col_names.append(str(col)+'_binned')
            return dataframe

        if self.prediction_mode:
            for vartype in self.num_dtypes:
                filtered_columns = self.dataframe.loc[:, ~self.dataframe.columns.isin(self.new_sin_cos_col_names)]
            self.dataframe = binning_on_data(self.dataframe, cols_to_bin=filtered_columns)
            return self.dataframe

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            for vartype in self.num_dtypes:
                filtered_columns = X_train.loc[:, ~X_train.columns.isin(self.new_sin_cos_col_names)]

            X_train = binning_on_data(X_train, cols_to_bin=filtered_columns)
            X_test = binning_on_data(X_test, cols_to_bin=filtered_columns)
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def cardinality_remover(self, threshold=1000):
        """
        Loops through all columns and delete columns with cardinality higher than defined threshold.
        :param threshold: integer of any size
        :return:Cleaned dataframe.
        """
        def remove_high_cardinality(df, threshold=threshold, cols_to_delete=[]):
            deleted_columns = []
            if not len(cols_to_delete):
                pass
            elif not cols_to_delete:
                cat_columns = df.select_dtypes(include=['object']).columns
                for col in cat_columns:
                    cardinality = df[col].nunique()
                    if cardinality >= threshold:
                        del df[col]
                        deleted_columns.append(col)
                    else:
                        pass
            else:
                cat_columns = cols_to_delete
                for col in cat_columns:
                    del df[col]
                    deleted_columns.append(col)
            return df, deleted_columns

        if self.prediction_mode:
            threshold = self.preprocess_decisions["cardinality_threshold"]
            self.dataframe, self.preprocess_decisions["cardinality_deleted_columns"] = \
                remove_high_cardinality(self.dataframe, cols_to_delete=self.preprocess_decisions["cardinality_deleted_columns"])
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train, self.preprocess_decisions["cardinality_deleted_columns"] = remove_high_cardinality(X_train)
            X_test, self.preprocess_decisions["cardinality_deleted_columns"] = remove_high_cardinality(df=X_test,
                                                                                                       cols_to_delete=self.preprocess_decisions["cardinality_deleted_columns"])
            self.preprocess_decisions["cardinality_threshold"] = threshold
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def rare_feature_processor(self, threshold=0.03, mask_as='miscellaneous'):
        """
        Loops through categorical columns and identifies categories, which occur less than the
        given threshold. These features will be grouped together as defined by mask_as parameter.
        :param threshold: Minimum share of categories to be not grouped as misc. Takes a float between 0 and 1.
        :param mask_as: Group name of grouped rare features.
        :return:
        """
        def handle_rarity(all_data, threshold=threshold, mask_as=mask_as):
            cat_columns = all_data.select_dtypes(include=['category']).columns
            for col in cat_columns:
                frequencies = all_data[col].value_counts(normalize = True)
                condition = frequencies < threshold
                mask_obs = frequencies[condition].index
                mask_dict = dict.fromkeys(mask_obs, mask_as)
                all_data[col] = all_data[col].replace(mask_dict)  # or you could make a copy not to modify original data
            return all_data

        if self.prediction_mode:
            threshold = self.preprocess_decisions["rare_feature_threshold"]
            self.dataframe = handle_rarity(self.dataframe, threshold)
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = handle_rarity(X_train)
            X_test = handle_rarity(X_test)
            self.preprocess_decisions["rare_feature_threshold"] = threshold
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def clustering_as_a_feature(self, algorithm='dbscan', nb_clusters=2, eps=0.3, n_jobs=-1, min_samples=50):
        """
        Takes the numerical columns of a dataframe and performs clustering via the chosen algorithm.
        Appends the clusters as a new column.
        :param algorithm: Chose 'dbscan' or 'gaussian_clusters'
        :param nb_clusters: Takes an integer of 2 or higher.
        :param eps: Epsilon (only needed for DBSCAN). Defines the distance clusters can be apart from each other.
        :param n_jobs: How many cores to use. Chose -1 for all cores.
        :param min_samples: Minimum number of samples required to form a cluster.
        :return: Returns the modified dataframe.
        """
        def add_dbscan_clusters(dataframe, eps=eps, n_jobs=n_jobs, min_samples=min_samples):
            dataframe_red = dataframe.loc[:, dataframe.columns.isin(self.num_columns)].copy()
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(dataframe_red)
            labels = db.labels_
            dataframe['dbscan_cluster'] = labels
            return dataframe

        def add_gaussian_mixture_clusters(dataframe, n_components=nb_clusters):
            gaussian = GaussianMixture(n_components=n_components)
            gaussian.fit(dataframe)
            gaussian_clusters = gaussian.predict(dataframe)
            dataframe["gaussian_clusters"] = gaussian_clusters
            return dataframe

        if not self.data_scaled:
            self.data_scaling()
        if algorithm == 'dbscan':
            if self.prediction_mode:
                self.dataframe = add_dbscan_clusters(self.dataframe)
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train = add_dbscan_clusters(X_train)
                X_test = add_dbscan_clusters(X_test)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif algorithm == 'gaussian':
            if self.prediction_mode:
                self.dataframe = add_gaussian_mixture_clusters(self.dataframe)
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train = add_gaussian_mixture_clusters(X_train)
                X_test = add_gaussian_mixture_clusters(X_test)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def delete_high_null_cols(self, threshold=0.5):
        """
        Takes in a dataframe and removes columns, which have more NULLs than the given threshold.
        :param threshold: Maximum percentage of NULLs in a column allowed.
        :return: Updates test and train class attributes.
        """
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train.dropna(axis=1, thresh=int(threshold*len(X_train)))
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    # TODO: Check if parameters can be used via **kwargs argument
    def fill_nulls(self, how='imputation', selected_cols=None, fill_with=0, inplace=False, **parameters):
        """
        Takes in a dataframe and fills all NULLs with chosen value.
        :param fill_with: Define value to replace NULLs with.
        :param inplace: Chose True or False.
        :param how: Chose 'static' to define static fill values, 'iterative_imputation' for the sklearns iterative
        imputer.
        :param selected_cols: Provide list of columns to define where to replace NULLs
        :return: Returns modified dataframe
        """

        def static_filling(dataframe, columns=None):
            dataframe[columns] = dataframe[columns].fillna(fill_with, inplace=inplace)
            return dataframe

        def iterative_imputation(dataframe, params=None):
            dataframe_cols = dataframe.columns
            imp_mean = IterativeImputer(random_state=0)
            if not params:
                pass
            else:
                imp_mean.set_params(**parameters)
            imp_mean.fit(dataframe)
            imp_mean.transform(dataframe)
            dataframe = pd.DataFrame(dataframe, columns=dataframe_cols)
            return dataframe

        if self.prediction_mode:
            if not selected_cols:
                cols = self.dataframe.columns.to_list()
            else:
                cols = selected_cols

            if not how:
                how = self.preprocess_decisions[f"fill_nulls_how"]
            else:
                pass

            if how == 'static':
                self.dataframe = static_filling(self.dataframe, cols)
            elif how == 'iterative_imputation':
                self.dataframe = iterative_imputation(self.dataframe, params=self.preprocess_decisions[f"fill_nulls_params"])
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if not selected_cols:
                cols = X_train.columns.to_list()
            else:
                cols = selected_cols

            if how == 'static':
                X_train = static_filling(X_train, cols)
                X_test = static_filling(X_test, cols)
                self.preprocess_decisions[f"fill_nulls_how"] = how
                self.preprocess_decisions[f"fill_nulls_params"] = cols
            elif how == 'iterative_imputation':
                X_train = iterative_imputation(X_train, params=parameters)
                X_test = iterative_imputation(X_test, params=parameters)
                self.preprocess_decisions[f"fill_nulls_how"] = how
                self.preprocess_decisions[f"fill_nulls_params"] = cols
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def isolation_forest_identifier(self, how='append', threshold=0):
        """
        Takes a dataframe and runs isolation forest to either flag or delete outliers.
        :param how: Chose if outlier scores shall be 'append' or 'delete'.
        :param threshold: Threshold responsible for outlier deletion. Samples under this threshold will be deleted.
        :return: Returns modified dataframe.
        """
        if self.prediction_mode:
            if self.preprocess_decisions[f"isolation_forest"]["how"] == 'append':
                outlier_detector = self.preprocess_decisions[f"isolation_forest"]["model"]
                outlier_predictions = outlier_detector.decision_function(self.dataframe)
                outlier_predictions_class = outlier_predictions*-1
                self.dataframe["isolation_probs"] = outlier_predictions
                self.dataframe["isolation_class"] = outlier_predictions_class
                return self.dataframe
            else:
                pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            outlier_detector = IsolationForest(contamination=0.7)
            if how == 'append':
                outlier_detector.fit(X_train)
                outlier_predictions_train = outlier_detector.decision_function(X_train)
                outlier_predictions_class_train = outlier_predictions_train*-1
                X_train["isolation_probs"] = outlier_predictions_train
                X_train["isolation_class"] = outlier_predictions_class_train
                outlier_predictions_test = outlier_detector.decision_function(X_test)
                outlier_predictions_class_test = outlier_predictions_test*-1
                X_test["isolation_probs"] = outlier_predictions_test
                X_test["isolation_class"] = outlier_predictions_class_test
                self.preprocess_decisions[f"isolation_forest"] = {}
                self.preprocess_decisions[f"isolation_forest"]["model"] = outlier_detector
                self.preprocess_decisions[f"isolation_forest"]["how"] = how
            elif how == 'delete':
                outlier_detector.fit(X_train)
                outlier_predictions_train = outlier_detector.decision_function(X_train)
                X_train["isolation_probs"] = outlier_predictions_train
                X_train = X_train[(X_train["isolation_probs"] < threshold)]
                self.preprocess_decisions[f"isolation_forest"] = {}
                self.preprocess_decisions[f"isolation_forest"]["model"] = outlier_detector
                self.preprocess_decisions[f"isolation_forest"]["how"] = how
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def iqr_remover(self, df, column, threshold=1.5):
        """Remove outliers from a dataframe by column, including optional
           whiskers, removing rows for which the column value are
           less than Q1-1.5IQR or greater than Q3+1.5IQR.
        Args:
            df (`:obj:pd.DataFrame`): A pandas dataframe to subset
            column (str): Name of the column to calculate the subset from.
            whisker_width (float): Optional, loosen the IQR filter by a
                                   factor of `whisker_width` * IQR.
        Returns:
            (`:obj:pd.DataFrame`): Filtered dataframe
        """
        if self.prediction_mode:
            pass
        else:
            whisker_width=threshold
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            dataframe_red = X_train.loc[:, X_train.columns.isin(self.num_columns)].copy()
            dataframe_red[self.target_variable] = Y_train
            for col in dataframe_red.columns:
                # Calculate Q1, Q2 and IQR
                q1 = dataframe_red[col].quantile(0.25)
                q3 = dataframe_red[col].quantile(0.75)
                iqr = q3 - q1
                # Apply filter with respect to IQR, including optional whiskers
                filter = (dataframe_red[col] > q1 - whisker_width*iqr) & (dataframe_red[col] < q3 + whisker_width*iqr)
                dataframe_red = dataframe_red.loc[filter]
            X_train = dataframe_red
            Y_train = dataframe_red[self.target_variable]
            del dataframe_red[self.target_variable]
            del dataframe_red
            _ = gc.collect()
        return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def outlier_care(self, method='isolation', how='append', threshold=None):
        """
        This method handles outliers isolation forest only currently.
        :param method: Chose the method of outlier detection. Either 'IQR', 'z_avg or 'iqr_avg'.
        :param how: Chose 'adjust' to correct outliers by adjusting them to IQR (for IQR only), 'delete' to delete all
        rows with outliers or 'append' to append outlier scores.
        :param threshold: Define by how many range an outlier has to be off to be interpreted as outlier.
        :return: Returns instantiated dataframe object.
        """
        if method == 'isolation' and how == 'append':
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == 'isolation' and how == 'delete':
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == 'iqr':
            return self.iqr_remover(threshold=1.5)

    def datetime_converter(self, datetime_handling='all'):
        """
        Takes in a dataframe and processes date and datetime columns by categorical and/or cyclic transformation.
        Tries to identify datetime columns automatically, if no date columns have been provided during class
        instantiation.
        :param datetime_handling: Chose '
        :return:
        """
        if self.prediction_mode:
            if not self.date_columns:
                date_columns = []
                # convert date columns from object to datetime type
                for col in self.dataframe.columns:
                    if col not in self.num_columns:
                        try:
                            self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True)
                            date_columns.append(col)
                        except Exception:
                            self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True, errors='coerce')
                            date_columns.append(col)
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True)

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if not self.date_columns:
                date_columns = []
                # convert date columns from object to datetime type
                for col in X_train.columns:
                    if col not in self.num_columns:
                        try:
                            X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True)
                            X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True)
                            date_columns.append(col)
                        except Exception:
                            X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True, errors='coerce')
                            X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True, errors='coerce')
                            date_columns.append(col)
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True, errors='coerce')
                    X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True, errors='coerce')

        self.date_columns_created = {}
        self.new_sin_cos_col_names = [] # used to filter out these columns from binning

        def date_converter(dataframe):
            """
            Takes in a dataframe and loops through datetime columns to and extracts the date parts month, day, dayofweek
            and hour and adds them as additional columns.
            :param dataframe:
            :return: Returns modified dataframe.
            """
            for c in date_columns:
                if c in dataframe.columns:
                    if dataframe[c].dt.month.nunique() > 0:
                        dataframe[c + '_month'] = dataframe[c].dt.month
                        self.date_columns_created[c + '_month'] = 'month'
                    if dataframe[c].dt.day.nunique() > 0:
                        dataframe[c + '_day'] = dataframe[c].dt.day
                        self.date_columns_created[c + '_day'] = 'day'
                    if dataframe[c].dt.dayofweek.nunique() > 0:
                        dataframe[c + '_dayofweek'] = dataframe[c].dt.dayofweek
                        self.date_columns_created[c + '_dayofweek'] = 'dayofweek'
                    if dataframe[c].dt.hour.nunique() > 0:
                        dataframe[c + '_hour'] = dataframe[c].dt.hour
                        self.date_columns_created[c + '_hour'] = 'hour'
            return dataframe

        def cos_sin_transformation(dataframe):
            """
            Takes in a dataframe and loops through date columns. Create sine and cosine features and appends them
            as new columns.
            :param dataframe:
            :return: Returns modified dataframe.
            """
            for c in self.date_columns_created:
                if c in dataframe.columns:
                    if self.date_columns_created[c] == 'month':
                        dataframe[c + '_sin'] = np.sin(dataframe[c] * (2. * np.pi / 12))
                        dataframe[c + '_cos'] = np.cos(dataframe[c] * (2. * np.pi / 12))
                        self.new_sin_cos_col_names.append(c + '_sin')
                        self.new_sin_cos_col_names.append(c + '_cos')
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == 'day':
                        dataframe[c + '_sin'] = np.sin(dataframe[c] * (2. * np.pi / 31))
                        dataframe[c + '_cos'] = np.cos(dataframe[c] * (2. * np.pi / 31))
                        self.new_sin_cos_col_names.append(c + '_sin')
                        self.new_sin_cos_col_names.append(c + '_cos')
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == 'dayofweek':
                        dataframe[c + '_sin'] = np.sin((dataframe[c] + 1) * (2. * np.pi / 7))
                        dataframe[c + '_cos'] = np.cos((dataframe[c] + 1) * (2. * np.pi / 7))
                        self.new_sin_cos_col_names.append(c + '_sin')
                        self.new_sin_cos_col_names.append(c + '_cos')
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == 'hour':
                        dataframe[c + '_sin'] = np.sin(dataframe[c] * (2. * np.pi / 24))
                        dataframe[c + '_cos'] = np.cos(dataframe[c] * (2. * np.pi / 24))
                        self.new_sin_cos_col_names.append(c + '_sin')
                        self.new_sin_cos_col_names.append(c + '_cos')
                        dataframe.drop(c, axis=1, inplace=True)
                    elif self.date_columns_created[c] == 'dayofyear':
                        dataframe[c + '_sin'] = np.sin(dataframe[c] * (2. * np.pi / 365))
                        dataframe[c + '_cos'] = np.cos(dataframe[c] * (2. * np.pi / 365))
                        self.new_sin_cos_col_names.append(c + '_sin')
                        self.new_sin_cos_col_names.append(c + '_cos')
                        dataframe.drop(c, axis=1, inplace=True)
            return dataframe

        if self.prediction_mode:
            datetime_handling = self.preprocess_decisions["datetime_handling"]
            if datetime_handling == 'cyclic':
                self.dataframe = cos_sin_transformation(self.dataframe)
            elif datetime_handling == 'categorical':
                self.dataframe = date_converter(self.dataframe)
            elif datetime_handling == 'all':
                self.dataframe = date_converter(self.dataframe)
                self.dataframe = cos_sin_transformation(self.dataframe)
            else:
                pass

        elif datetime_handling == 'cyclic':
            X_train = cos_sin_transformation(X_train)
            X_test = cos_sin_transformation(X_test)
        elif datetime_handling == 'categorical':
            X_train = date_converter(X_train)
            X_test = date_converter(X_test)
        elif datetime_handling == 'all':
            X_train = date_converter(X_train)
            X_train = cos_sin_transformation(X_train)
            X_test = date_converter(X_test)
            X_test = cos_sin_transformation(X_test)
        else:
            pass

        if self.prediction_mode:
            # drop initial date columns
            for dates in date_columns:
                if dates in self.dataframe.columns:
                    # safe_copy = all_data[dates].copy()
                    self.dataframe.drop(dates, axis=1, inplace=True)
            return self.dataframe
        else:
            # drop initial date columns
            for dates in date_columns:
                if dates in X_train.columns:
                    # safe_copy = all_data[dates].copy()
                    X_train.drop(dates, axis=1, inplace=True)
                    X_test.drop(dates, axis=1, inplace=True)
            self.preprocess_decisions["datetime_handling"] = datetime_handling
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.date_columns_created

    def category_encoding(self, algorithm='target'):
        """
        Takes in a dataframe and applies the chosen category encoding algorithm to categorical columns.
        :param algorithm:
        :return: Returns modified dataframe.
        """
        if self.prediction_mode:
            cat_columns = self.cat_columns_encoded
            enc = self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"]
            self.dataframe[cat_columns] = enc.transform(self.dataframe[cat_columns])
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            cat_columns = X_train.select_dtypes(include=['object']).columns
            self.cat_columns_encoded = cat_columns
            self.preprocess_decisions[f"category_encoders"] = {}
            if algorithm == 'target':
                enc = TargetEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            elif algorithm == 'woee':
                enc = WOEEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            elif algorithm == 'GLMM':
                enc = GLMMEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            elif algorithm == 'ordinal':
                enc = OrdinalEncoder(cols=cat_columns)
                X_train = enc.fit_transform(X_train, Y_train)
                X_test = enc.transform(X_test)
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            elif algorithm == 'leaveoneout':
                enc = LeaveOneOutEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def remove_collinearity(self, threshold=0.8):
        """
        Loops through all columns and checks, if features are highly positively correlated.
        If correlation is above given threshold, then only one column is kept.
        :param threshold: Maximum allowed correlation. Expects a float from -1 to +1.
        :return: Returns modified dataframe.
        """
        def correlation(dataset, threshold=threshold):
            col_corr = set() # Set of all the names of deleted columns
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i] # getting the name of column
                        col_corr.add(colname)
                        del_corr.append(colname)
                        if colname in dataset.columns:
                            del dataset[colname] # deleting the column from the dataset
            return dataset

        if self.prediction_mode:
            threshold = self.preprocess_decisions[f"remove_collinearity_threshold"]
            self.dataframe = self.dataframe.drop(self.excluded, axis=1)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            del_corr = []
            X_train = correlation(X_train, 0.8)
            X_test = X_test.drop(del_corr, axis=1)
            self.excluded = del_corr
            self.preprocess_decisions[f"remove_collinearity_threshold"] = threshold
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.excluded

    def smote_data(self):
        """
        Applies vanilla form of Synthetical Minority Over-Sampling Technique.
        :return: Returns modifie dataframe.
        """
        if self.prediction_mode:
            pass
        else:
            oversample = SMOTE(n_jobs=-1)
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train_cols = X_train.columns
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)
            X_train = pd.DataFrame(X_train, columns=X_train_cols)
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def automated_feature_selection(self, metric='logloss'):
        """
        Uses boostaroota algorithm to automatically chose best features. boostaroota choses XGboost under
        the hood.
        :param metric: Metric to evaluate strength of features.
        :return: Returns reduced dataframe.
        """
        if self.prediction_mode:
            self.dataframe = self.dataframe[self.selected_feats]
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            br = BoostARoota(metric=metric)
            br.fit(X_train, Y_train)
            selected = br.keep_vars_
            X_train = X_train[selected]
            X_test = X_test[selected]
            self.selected_feats = selected
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.selected_feats

