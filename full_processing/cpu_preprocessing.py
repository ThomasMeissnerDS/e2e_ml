from pandas.core.common import SettingWithCopyWarning
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from category_encoders import *
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from boostaroota import BoostARoota
import gc
import warnings
import logging
import pickle
import os
import psutil
import time

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class PreProcessing:
    """
    The preprocessing base class expects a Pandas dataframe and the target variable at least.
    Date columns and categorical columns can be passed as lists additionally for respective preprocessing.
    A unique identifier (i.e. an ID column) can be passed as well to preserve this information for later processing.
    """

    def __init__(self, datasource, target_variable, date_columns=None, categorical_columns=None, num_columns=None,
                 unique_identifier=None, selected_feats=None, cat_encoded=None, cat_encoder_model=None, nlp_columns=None,
                 prediction_mode=False, preferred_training_mode='cpu', preprocess_decisions=None, trained_model=None, ml_task=None,
                 logging_file_path=None, low_memory_mode=False, save_models_path=None, train_split_type='cross'):

        self.dataframe = datasource
        self.low_memory_mode = low_memory_mode
        self.save_models_path = save_models_path
        self.logging_file_path = logging_file_path
        logging.basicConfig(filename=f'{self.logging_file_path}.log', format='%(asctime)s %(message)s',
                            level=logging.DEBUG)
        logging.info('Class instance created.')

        # check which type the data source is
        if isinstance(datasource, np.ndarray):
            self.source_format = 'numpy array'
        elif isinstance(datasource, pd.DataFrame):
            self.source_format = 'Pandas dataframe'
            self.dataframe.columns = self.dataframe.columns.astype(str)
        else:
            self.source_format = 'Unknown, not recommened'

        # check if we face a classification problem and check how many classes we have
        if not ml_task:
            try:
                if datasource[target_variable].nunique() > 2:
                    self.class_problem = 'multiclass'
                    self.num_classes = datasource[target_variable].nunique()
                elif datasource[target_variable].nunique() == 2:
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
        else:
            self.class_problem = ml_task

        if preferred_training_mode == 'cpu' or preferred_training_mode == 'gpu':
            self.preferred_training_mode = preferred_training_mode
        else:
            self.preferred_training_mode = 'cpu'
        self.train_split_type = train_split_type
        self.date_columns = date_columns
        self.date_columns_created = None
        self.categorical_columns = categorical_columns
        self.nlp_columns = nlp_columns
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
        self.predicted_values = {}
        self.evaluation_scores = {}
        self.xg_boost_regression = None
        self.xgboost_objective = None
        self.prediction_mode = prediction_mode
        self.best_model = None
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
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')

    def __repr__(self):
        return f"Central data class holding all information like dataframes, " \
               f"columns of certain data types, saved models and predictions." \
               f"Current target variable:'{self.target_variable}'"

    def __str__(self):
        return f"Current target: {self.target_variable}"

    def get_current_timestamp(self, task=None):
        t = time.localtime()
        if task:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"Started {task} at {current_time}.")
        else:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"{current_time}")
        return current_time

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
            logging.info('Skipped wrapping dataframe dict due to prediction mode.')
            pass
        else:
            logging.info('Start wrapping dataframe dictionary')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.df_dict = {'X_train': X_train,
                            'X_test': X_test,
                            'Y_train': Y_train,
                            'Y_test': Y_test}
            logging.info('Finished wrapping dataframe dictionary')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            del X_train,
            del X_test,
            del Y_train,
            del Y_test
            _ = gc.collect()
            return self.df_dict

    def unpack_test_train_dict(self):
        """
        This function takes in the class dictionary holding test and train split and unpacks it.
        :return: X_train, X_test as dataframes. Y_train, Y_test as Pandas series.
        """
        logging.info('Start unpacking data dictionary')
        X_train, X_test, Y_train, Y_test = self.df_dict["X_train"], self.df_dict["X_test"], self.df_dict["Y_train"], \
                                           self.df_dict["Y_test"]
        logging.info('Unpacking of data dictionary finished.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        return X_train, X_test, Y_train, Y_test

    def np_array_wrap_test_train_to_dict(self, Y_train, Y_test):
        """
        Takes in X_train & X_test parts and updates the class instance dictionary.
        :param Y_train: Numpy array
        :param Y_test: Numpy array
        :return: Class dictionary
        """
        if self.prediction_mode:
            logging.info('Wrapping Numpy dict skipped due to prediction mode.')
            pass
        else:
            logging.info('Start wrapping Numpy dict.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.df_dict = {'Y_train': Y_train,
                            'Y_test': Y_test}
            logging.info('Finished wrapping Numpy dict.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.df_dict

    def np_array_unpack_test_train_dict(self):
        """
        This function takes in the class dictionary holding test and train split and unpacks it.
        :return: X_train, X_test as dataframes. Y_train, Y_test as numpy array.
        """
        logging.info('Start unpacking Numpy dict.')
        Y_train, Y_test = self.df_dict["Y_train"], self.df_dict["Y_test"]
        logging.info('Finished unpacking Numpy dict.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        return Y_train, Y_test

    def save_load_model_file(self, model_object=None, model_path=None, algorithm=None, algorithm_variant='none',
                             file_type='.dat', action='save', clean=True):
        if self.save_models_path:
            path = self.save_models_path
        elif model_path:
            path = model_path
        else:
            pass
        full_path = path + '_' + algorithm + '_' + algorithm_variant + '_' + file_type

        if action == 'save':
            self.get_current_timestamp(task='Save blueprint instance.')
            filehandler = open(full_path, 'wb')
            pickle.dump(model_object, filehandler)
            filehandler.close()
            if clean:
                del model_object
                _ = gc.collect()
        elif action == 'load':
            self.get_current_timestamp(task='Load blueprint instance.')
            filehandler = open(full_path, 'rb')
            model_object = pickle.load(filehandler)
            filehandler.close()
            return model_object
        else:
            pass

    def reduce_mem_usage(self, df):
        """
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return df

    def reduce_memory_footprint(self):
        """
        Takes a dataframe and downcasts columns if possible.
        :return: Returns downcasted dataframe.
        """
        self.get_current_timestamp(task='Reduce memory footprint of dataframe')
        logging.info('Started reducing memory footprint.')
        if self.prediction_mode:
            self.dataframe = self.reduce_mem_usage(self.dataframe)
            logging.info('Finished reducing memory footprint.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = self.reduce_mem_usage(X_train)
            X_test = self.reduce_mem_usage(X_test)
            logging.info('Finished reducing memory footprint.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def sort_columns_alphabetically(self):
        """
        Takes a dataframe and sorts its columns alphabetically. This increases pipelines robustness in cases
        where the input data might have been changed in order.
        :return: Updates class instance. Returns dictionary.
        """
        self.get_current_timestamp(task='Sort columns alphabetically')
        if self.prediction_mode:
            logging.info('Started sorting columns alphabetically.')
            self.dataframe = self.dataframe.sort_index(axis=1)
            logging.info('Finished sorting columns alphabetically.')
        else:
            logging.info('Started sorting columns alphabetically.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = X_train.sort_index(axis=1)
            X_test = X_test.sort_index(axis=1)
            logging.info('Finished sorting columns alphabetically.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def label_encoder_decoder(self, target, mode='fit', direction='encode'):
        self.get_current_timestamp(task='Execute label encoding')
        logging.info('Started label encoding.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if direction == 'encode' and mode == 'fit':
            le = preprocessing.LabelEncoder()
            target = le.fit_transform(target)
            self.preprocess_decisions["label_encoder"] = le
            self.labels_encoded = True
        elif direction == 'encode' and mode == 'predict':
            le = self.preprocess_decisions["label_encoder"]
            le.transform(target)
        else:
            le = self.preprocess_decisions["label_encoder"]
            le.inverse_transform(target)
        logging.info('Finished label encoding.')
        del le
        _ = gc.collect()
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        return target

    def data_scaling(self, scaling='minmax'):
        """
        Scales the data using the chosen scaling algorithm.
        :param scaling: Chose 'minmax'.
        :return:
        """
        self.get_current_timestamp(task='Scale data')
        if self.prediction_mode:
            logging.info('Started data scaling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            dataframe_cols = self.dataframe.columns
            if scaling == 'minmax':
                scaler = self.preprocess_decisions["scaling"]
                scaler.fit(self.dataframe)
                scaler.transform(self.dataframe)
            self.dataframe = pd.DataFrame(self.dataframe, columns=dataframe_cols)
            self.data_scaled = True
            logging.info('Finished data scaling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe, self.data_scaled
        else:
            logging.info('Started data scaling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
            logging.info('Finished data scaling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            del scaler
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train,
                                                Y_test), self.data_scaled, self.preprocess_decisions

    def skewness_removal(self):
        self.get_current_timestamp(task='Remove skewness')
        if self.prediction_mode:
            logging.info('Started skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            for col in self.preprocess_decisions["skewed_columns"]:
                log_array = np.log(self.dataframe[col])
                log_array[np.isfinite(log_array) == False] = 0
                self.dataframe[col] = log_array
            logging.info('Finished skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            logging.info('Started skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            skewness = X_train.skew(axis=0, skipna=True)
            left_skewed = skewness[skewness < -0.5].index.to_list()
            print(left_skewed)
            right_skewed = skewness[skewness > 0.5].index.to_list()
            skewed = left_skewed+right_skewed
            for col in X_train[skewed].columns:
                log_array = np.log(X_train[col])
                log_array[np.isfinite(log_array) == False] = 0
                X_train[col] = log_array
                log_array = np.log(X_test[col])
                log_array[np.isfinite(log_array) == False] = 0
                X_test[col] = log_array
            logging.info('Finished skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            print(skewed)
            self.preprocess_decisions["skewed_columns"] = skewed
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train,
                                                Y_test)

    def train_test_split(self, how='cross', split_by_col=None, split_date=None, train_size=0.80):
        """
        This method splits the dataframe either as a simple or as a time split.
        :param how: 'cross' for cross validation, 'time' for time validation.
        :param split_by_col: Chose column to be used for split. For time validation only.
        :param split_date: Chose exact date to split. Test dataframe is equal or greater than provided date.
        :param train_size: Chose how much percentage the train dataframe will have. For cross validation only.
        :return: X_train, X_test, Y_train, Y_test
        """
        self.get_current_timestamp(task='Execute test train split')
        if self.prediction_mode:
            logging.info('Skipped test train split due to prediction mode.')
            pass
        elif how == 'cross':
            logging.info('Started test train split.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
            _ = gc.collect()
            logging.info('Finished test train split.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif how == 'time':
            logging.info('Started test train split.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            if self.source_format == 'numpy array':
                length = self.dataframe.size
                train_length = int(length * train_size)
                test_length = length - train_length
                Y_train, Y_test = self.dataframe[:train_length], self.dataframe[:test_length]
                logging.info('Finished test train split.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.np_array_wrap_test_train_to_dict(Y_train, Y_test)
            elif self.source_format == 'Pandas dataframe':
                length = len(self.dataframe.index)
                train_length = int(length * 0.80)
                test_length = length - train_length
                if not split_by_col:
                    self.dataframe = self.dataframe.sort_index()
                elif split_by_col:
                    self.dataframe = self.dataframe.sort_values(by=[split_by_col])
                else:
                    pass
                if split_date:
                    X_train = self.dataframe[(self.dataframe.split_by_col < split_date)]
                    X_test = self.dataframe[(self.dataframe.split_by_col >= split_date)]
                else:
                    X_train = self.dataframe.head(train_length)
                    X_test = self.dataframe.tail(test_length)
                Y_train = X_train[self.target_variable]
                Y_test = X_test[self.target_variable]
                del X_train[self.target_variable]
                del X_test[self.target_variable]
                logging.info('Finished test train split.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        else:
            logging.warning('No split method provided.')
            raise Exception("Please provide a split method.")

    def data_binning(self, nb_bins=10):
        """
        Takes numerical columns and splits them into desired number of bins. Bins will be attached as
        new columns to the dataframe.
        :param nb_bins: Takes a positive integer.
        :return:
        """
        self.get_current_timestamp(task='Execute numerical binning')

        def binning_on_data(dataframe, cols_to_bin=None):
            num_columns = cols_to_bin.select_dtypes(include=[vartype]).columns
            for col in num_columns:
                dataframe[str(col) + '_binned'] = pd.cut(dataframe[col], bins=nb_bins, labels=False)
                self.new_sin_cos_col_names.append(str(col) + '_binned')
            del num_columns
            _ = gc.collect()
            return dataframe

        logging.info('Start numerical binning.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            for vartype in self.num_dtypes:
                filtered_columns = self.dataframe.loc[:, ~self.dataframe.columns.isin(self.new_sin_cos_col_names)]
            self.dataframe = binning_on_data(self.dataframe, cols_to_bin=filtered_columns)
            logging.info('Finished numerical binning.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            for vartype in self.num_dtypes:
                filtered_columns = X_train.loc[:, ~X_train.columns.isin(self.new_sin_cos_col_names)]

            X_train = binning_on_data(X_train, cols_to_bin=filtered_columns)
            X_test = binning_on_data(X_test, cols_to_bin=filtered_columns)
            logging.info('Finished numerical binning.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def cardinality_remover(self, threshold=100):
        """
        Loops through all columns and delete columns with cardinality higher than defined threshold.
        :param threshold: integer of any size
        :return:Cleaned dataframe.
        """
        self.get_current_timestamp(task='Remove cardinality')

        def remove_high_cardinality(df, threshold=threshold, cols_to_delete=None):
            if not cols_to_delete:
                deleted_columns = []
                cat_columns = df.select_dtypes(include=['object']).columns
                for col in cat_columns:
                    cardinality = df[col].nunique()
                    if cardinality >= threshold:
                        df = df.drop([col], axis=1)
                        deleted_columns.append(col)
                    else:
                        pass
            else:
                deleted_columns = cols_to_delete
                for col in deleted_columns:
                    df = df.drop([col], axis=1)
            return df, deleted_columns

        logging.info('Start cardinality removal.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            threshold = self.preprocess_decisions["cardinality_threshold"]
            self.dataframe, self.preprocess_decisions["cardinality_deleted_columns"] = remove_high_cardinality(self.dataframe, cols_to_delete=self.preprocess_decisions[
                "cardinality_deleted_columns"])
            logging.info('Finished cardinality removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train, self.preprocess_decisions["cardinality_deleted_columns"] = remove_high_cardinality(X_train,
                                                                                                        threshold=threshold)
            X_test, self.preprocess_decisions["cardinality_deleted_columns"] = remove_high_cardinality(df=X_test,
                                             cols_to_delete=self.preprocess_decisions["cardinality_deleted_columns"])
            self.preprocess_decisions["cardinality_threshold"] = threshold
            logging.info('Finished cardinality removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def rare_feature_processor(self, threshold=0.03, mask_as='miscellaneous'):
        """
        Loops through categorical columns and identifies categories, which occur less than the
        given threshold. These features will be grouped together as defined by mask_as parameter.
        :param threshold: Minimum share of categories to be not grouped as misc. Takes a float between 0 and 1.
        :param mask_as: Group name of grouped rare features.
        :return:
        """
        self.get_current_timestamp('Handle rare features')

        def handle_rarity(all_data, threshold=threshold, mask_as=mask_as):
            cat_columns = all_data.select_dtypes(include=['category']).columns
            for col in cat_columns:
                frequencies = all_data[col].value_counts(normalize=True)
                condition = frequencies < threshold
                mask_obs = frequencies[condition].index
                mask_dict = dict.fromkeys(mask_obs, mask_as)
                all_data[col] = all_data[col].replace(mask_dict)  # or you could make a copy not to modify original data
            del cat_columns
            _ = gc.collect()
            return all_data

        logging.info('Start rare feature processing.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            threshold = self.preprocess_decisions["rare_feature_threshold"]
            self.dataframe = handle_rarity(self.dataframe, threshold)
            logging.info('Finished rare feature processing.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = handle_rarity(X_train)
            X_test = handle_rarity(X_test)
            self.preprocess_decisions["rare_feature_threshold"] = threshold
            logging.info('Finished rare feature processing.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
        self.get_current_timestamp('Execute clustering as a feature')

        def add_dbscan_clusters(dataframe, eps=eps, n_jobs=n_jobs, min_samples=min_samples):
            dataframe_red = dataframe.loc[:, dataframe.columns.isin(self.num_columns)].copy()
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(dataframe_red)
            labels = db.labels_
            dataframe['dbscan_cluster'] = labels
            del db
            del labels
            _ = gc.collect()
            return dataframe

        def add_gaussian_mixture_clusters(dataframe, n_components=nb_clusters):
            gaussian = GaussianMixture(n_components=n_components)
            gaussian.fit(dataframe)
            gaussian_clusters = gaussian.predict(dataframe)
            dataframe["gaussian_clusters"] = gaussian_clusters
            del gaussian
            del gaussian_clusters
            _ = gc.collect()
            return dataframe

        def add_kmeans_clusters(dataframe, n_components=nb_clusters):
            kmeans = KMeans(n_clusters=n_components)
            kmeans.fit(dataframe)
            kmeans_clusters = kmeans.predict(dataframe)
            dataframe[f"kmeans_clusters{n_components}"] = kmeans_clusters
            del kmeans
            del kmeans_clusters
            _ = gc.collect()
            return dataframe

        logging.info('Start adding clusters as additional features.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if not self.data_scaled:
            self.data_scaling()
        if algorithm == 'dbscan':
            if self.prediction_mode:
                self.dataframe = add_dbscan_clusters(self.dataframe)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train = add_dbscan_clusters(X_train)
                X_test = add_dbscan_clusters(X_test)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif algorithm == 'gaussian':
            if self.prediction_mode:
                self.dataframe = add_gaussian_mixture_clusters(self.dataframe)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train = add_gaussian_mixture_clusters(X_train)
                X_test = add_gaussian_mixture_clusters(X_test)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
        elif algorithm == 'kmeans':
            if self.prediction_mode:
                self.dataframe = add_kmeans_clusters(self.dataframe)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.dataframe
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train = add_kmeans_clusters(X_train)
                X_test = add_kmeans_clusters(X_test)
                logging.info('Finished adding clusters as additional features.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def delete_high_null_cols(self, threshold=0.5):
        """
        Takes in a dataframe and removes columns, which have more NULLs than the given threshold.
        :param threshold: Maximum percentage of NULLs in a column allowed.
        :return: Updates test and train class attributes.
        """
        self.get_current_timestamp(' Delete columns with high share of NULLs')
        if self.prediction_mode:
            for high_null_col in self.preprocess_decisions["deleted_high_null_cols"]:
                del self.dataframe[high_null_col]
            logging.info('Finished deleting columns with many NULLs.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        else:
            logging.info('Started deleting columns with many NULLs.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            columns_before = X_train.columns.to_list()
            X_train.dropna(axis=1, thresh=int(threshold * len(X_train)))
            columns_after = X_train.columns.to_list()
            deleted_columns = (set(columns_before).difference(columns_after))
            deleted = []
            for key in deleted_columns:
                deleted.append(key)
            self.preprocess_decisions["deleted_high_null_cols"] = deleted
            logging.info(f'Finished deleting columns with many NULLs: {deleted}.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    # TODO: Check if parameters can be used via **kwargs argument
    def fill_nulls(self, how='imputation', selected_cols=None, fill_with=0, **parameters):
        """
        Takes in a dataframe and fills all NULLs with chosen value.
        :param fill_with: Define value to replace NULLs with.
        :param how: Chose 'static' to define static fill values, 'iterative_imputation' for the sklearns iterative
        imputer.
        :param selected_cols: Provide list of columns to define where to replace NULLs
        :return: Returns modified dataframe
        """
        self.get_current_timestamp('Fill nulls')

        def static_filling(dataframe, columns=None):
            dataframe[columns] = dataframe[columns].fillna(fill_with, inplace=False)
            return dataframe

        def iterative_imputation(dataframe, params=None):
            dataframe_cols = dataframe.columns.to_list()
            imp_mean = IterativeImputer(random_state=0)
            if not params:
                pass
            else:
                imp_mean.set_params(**parameters)
            imp_mean.fit(dataframe)
            imp_mean.transform(dataframe)
            dataframe = pd.DataFrame(dataframe, columns=dataframe_cols)
            del imp_mean
            _ = gc.collect()
            return dataframe

        logging.info('Started filling NULLs.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
                self.dataframe[cols] = iterative_imputation(self.dataframe,
                                                            params=self.preprocess_decisions[f"fill_nulls_params"])
            logging.info('Finished filling NULLs.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
            logging.info('Finished filling NULLs.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
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
                outlier_predictions_class = outlier_predictions * -1
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
                outlier_predictions_class_train = outlier_predictions_train * -1
                X_train["isolation_probs"] = outlier_predictions_train
                X_train["isolation_class"] = outlier_predictions_class_train
                outlier_predictions_test = outlier_detector.decision_function(X_test)
                outlier_predictions_class_test = outlier_predictions_test * -1
                X_test["isolation_probs"] = outlier_predictions_test
                X_test["isolation_class"] = outlier_predictions_class_test
                del outlier_predictions_train
                del outlier_predictions_test
                del outlier_predictions_class_train
                del outlier_predictions_class_test
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
                del outlier_predictions_train
            del outlier_detector
            _ = gc.collect()
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
            return self.dataframe
        else:
            whisker_width = threshold
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            dataframe_red = X_train.loc[:, X_train.columns.isin(self.num_columns)].copy()
            dataframe_red[self.target_variable] = Y_train
            for col in dataframe_red.columns:
                # Calculate Q1, Q2 and IQR
                q1 = dataframe_red[col].quantile(0.25)
                q3 = dataframe_red[col].quantile(0.75)
                iqr = q3 - q1
                # Apply filter with respect to IQR, including optional whiskers
                filter = (dataframe_red[col] > q1 - whisker_width * iqr) & (
                            dataframe_red[col] < q3 + whisker_width * iqr)
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
        self.get_current_timestamp('Handle outliers')
        logging.info('Started outlier handling.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if method == 'isolation' and how == 'append':
            logging.info('Finished outlier handling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == 'isolation' and how == 'delete':
            logging.info('Finished outlier handling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.isolation_forest_identifier(how=how, threshold=threshold)
        elif method == 'iqr':
            logging.info('Finished outlier handling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.iqr_remover(threshold=1.5)

    def datetime_converter(self, datetime_handling='all', force_conversion=False):
        """
        Takes in a dataframe and processes date and datetime columns by categorical and/or cyclic transformation.
        Tries to identify datetime columns automatically, if no date columns have been provided during class
        instantiation.
        :param datetime_handling: Chose '
        :return:
        """
        if self.prediction_mode:
            if not self.date_columns:
                logging.info('Started automatic datetime column detection.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                date_columns = []
                # convert date columns from object to datetime type
                for col in self.dataframe.columns:
                    if col not in self.num_columns:
                        try:
                            self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True)
                            date_columns.append(col)
                        except Exception:
                            if force_conversion:
                                self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True,
                                                                     errors='coerce')
                                date_columns.append(col)
                logging.info('Finished automatic datetime column detection.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], infer_datetime_format=True)

        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if not self.date_columns:
                logging.info('Started automatic datetime column detection.')
                logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
                date_columns = []
                # convert date columns from object to datetime type
                for col in X_train.columns:
                    if col not in self.num_columns:
                        try:
                            X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True)
                            X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True)
                            date_columns.append(col)
                        except Exception:
                            if force_conversion:
                                X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True, errors='coerce')
                                X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True, errors='coerce')
                                date_columns.append(col)
            else:
                date_columns = self.date_columns
                for col in date_columns:
                    X_train[col] = pd.to_datetime(X_train[col], infer_datetime_format=True, errors='coerce')
                    X_test[col] = pd.to_datetime(X_test[col], infer_datetime_format=True, errors='coerce')
            logging.info('Finished automatic datetime column detection.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')

        self.date_columns_created = {}
        self.new_sin_cos_col_names = []  # used to filter out these columns from binning

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

        self.get_current_timestamp(task='Apply datetime transformation')
        logging.info('Started datetime column handling.')
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
            logging.info('Finished datetime column handling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            # drop initial date columns
            for dates in date_columns:
                if dates in X_train.columns:
                    # safe_copy = all_data[dates].copy()
                    X_train.drop(dates, axis=1, inplace=True)
                    X_test.drop(dates, axis=1, inplace=True)
            self.preprocess_decisions["datetime_handling"] = datetime_handling
            logging.info('Finished datetime column handling.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.date_columns_created

    def onehot_pca(self):
        self.get_current_timestamp(task='Onehot + PCA categorical features')
        if self.prediction_mode:
            if len(self.cat_columns_encoded) > 0:
                df_branch = self.dataframe[self.cat_columns_encoded].copy()
                enc = self.preprocess_decisions[f"onehot_pca"]["onehot_encoder"]
                df_branch = enc.transform(df_branch[self.cat_columns_encoded])
                df_branch.fillna(0, inplace=True)
                onehot_cols = df_branch.columns
                pca = PCA(n_components=2)
                pred_comps = pca.fit_transform(df_branch[onehot_cols])
                df_branch = pd.DataFrame(pred_comps, columns=['PC-1', 'PC-2'])
                for col in df_branch.columns:
                    self.dataframe[f"{col}_pca"] = df_branch[col]
            else:
                pass
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions[f"onehot_pca"] = {}
            if self.cat_columns_encoded:
                cat_columns = self.cat_columns_encoded
            else:
                cat_columns = X_train.select_dtypes(include=['object']).columns.to_list()
                self.cat_columns_encoded = cat_columns

            if len(self.cat_columns_encoded) > 0:
                enc = OneHotEncoder(handle_unknown='ignore')
                X_train_branch = X_train[cat_columns].copy()
                X_test_branch = X_test[cat_columns].copy()
                X_train_branch = enc.fit_transform(X_train_branch[cat_columns], Y_train)
                X_test_branch = enc.transform(X_test_branch[cat_columns])
                onehot_cols = X_train_branch.columns
                X_train_branch.fillna(0, inplace=True)
                X_test_branch.fillna(0, inplace=True)
                pca = PCA(n_components=2)
                train_comps = pca.fit_transform(X_train_branch[onehot_cols])
                X_train_branch = pd.DataFrame(train_comps, columns=['PC-1', 'PC-2'])
                test_comps = pca.transform(X_test_branch[onehot_cols])
                X_test_branch = pd.DataFrame(test_comps, columns=['PC-1', 'PC-2'])
                pca_cols = []
                for col in X_train_branch.columns:
                    X_train[f"{col}_pca"] = X_train_branch[col]
                    X_test[f"{col}_pca"] = X_test_branch[col]
                    pca_cols.append(f"{col}_pca")
                self.preprocess_decisions[f"onehot_pca"]["pca_cols"] = pca_cols
                self.preprocess_decisions[f"onehot_pca"]["onehot_encoder"] = enc
                del X_train_branch
                del X_test_branch
                del pca
                _ = gc.collect()
            else:
                pass
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def category_encoding(self, algorithm='target'):
        """
        Takes in a dataframe and applies the chosen category encoding algorithm to categorical columns.
        :param algorithm:
        :return: Returns modified dataframe.
        """
        self.get_current_timestamp('Execute categorical encoding')
        logging.info('Started category encoding.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            cat_columns = self.cat_columns_encoded
            enc = self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"]
            self.dataframe[cat_columns] = enc.transform(self.dataframe[cat_columns])
            logging.info('Finished category encoding.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            cat_columns = X_train.select_dtypes(include=['object']).columns.to_list()
            self.cat_columns_encoded = cat_columns
            self.preprocess_decisions[f"category_encoders"] = {}
            if algorithm == 'target':
                enc = TargetEncoder(cols=cat_columns)
                X_train[cat_columns] = enc.fit_transform(X_train[cat_columns], Y_train)
                X_test[cat_columns] = enc.transform(X_test[cat_columns])
                self.preprocess_decisions[f"category_encoders"][f"{algorithm}_all_cols"] = enc
            elif algorithm == 'onehot':
                enc = OneHotEncoder(handle_unknown='ignore')
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
            logging.info('Finished category encoding.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            X_train.drop(cat_columns, axis=1)
            X_test.drop(cat_columns, axis=1)
            del enc
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def remove_collinearity(self, threshold=0.8):
        """
        Loops through all columns and checks, if features are highly positively correlated.
        If correlation is above given threshold, then only one column is kept.
        :param threshold: Maximum allowed correlation. Expects a float from -1 to +1.
        :return: Returns modified dataframe.
        """
        self.get_current_timestamp('Remove collinearity')

        def correlation(dataset, threshold=threshold):
            col_corr = set()  # Set of all the names of deleted columns
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i]  # getting the name of column
                        col_corr.add(colname)
                        del_corr.append(colname)
                        if colname in dataset.columns:
                            del dataset[colname]  # deleting the column from the dataset
            del corr_matrix
            _ = gc.collect()
            return dataset

        logging.info('Started removing collinearity.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            threshold = self.preprocess_decisions[f"remove_collinearity_threshold"]
            self.dataframe = self.dataframe.drop(self.excluded, axis=1)
            logging.info('Finished removing collinearity.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            del_corr = []
            X_train = correlation(X_train, 0.8)
            X_test = X_test.drop(del_corr, axis=1)
            self.excluded = del_corr
            self.preprocess_decisions[f"remove_collinearity_threshold"] = threshold
            logging.info('Finished removing collinearity.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.excluded

    def smote_data(self):
        """
        Applies vanilla form of Synthetical Minority Over-Sampling Technique.
        :return: Returns modifie dataframe.
        """
        self.get_current_timestamp('Smote data')
        if self.prediction_mode:
            logging.info('Skipped SMOTE due to prediction mode.')
            pass
        else:
            logging.info('Started SMOTE.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            oversample = SMOTE(n_jobs=-1)
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train_cols = X_train.columns
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)
            X_train = pd.DataFrame(X_train, columns=X_train_cols)
            logging.info('Finished SMOTE.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            del oversample
            _ = gc.collect()
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def automated_feature_selection(self, metric='logloss'):
        """
        Uses boostaroota algorithm to automatically chose best features. boostaroota choses XGboost under
        the hood.
        :param metric: Metric to evaluate strength of features.
        :return: Returns reduced dataframe.
        """
        self.get_current_timestamp('Select best features')
        if self.prediction_mode:
            logging.info('Start filtering for preselected columns.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.dataframe = self.dataframe[self.selected_feats]
            logging.info('Finished filtering preselected columns.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            logging.info('Start automated feature selection.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            for col in X_train.columns:
                print(col)
            br = BoostARoota(metric=metric)
            br.fit(X_train, Y_train)
            selected = br.keep_vars_
            X_train = X_train[selected]
            X_test = X_test[selected]
            self.selected_feats = selected
            for i in selected:
                print(f" Selected features are... {i}.")
            logging.info('Finished automated feature selection.')
            del br
            _ = gc.collect()
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test), self.selected_feats
