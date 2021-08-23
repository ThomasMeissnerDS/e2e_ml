from pandas.core.common import SettingWithCopyWarning
from sklearn import model_selection
from sklearn.linear_model import BayesianRidge
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
import lightgbm
import xgboost as xgb
import torch
import gc
import warnings
import logging
import pickle
import os
import psutil
import time
import random

pd.options.display.max_colwidth = 1000
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class PreProcessing:
    """
    This class stores all pipeline relevant information. The attribute "df_dict" always holds train and test as well as
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
    :param nlp_columns: NLP columns expect a string declaring one text column.
    :param unique_identifier: A unique identifier (i.e. an ID column) can be passed as well to preserve this information
     for later processing.
    :param ml_task: Can be 'binary', 'multiclass' or 'regression'. On default will be determined automatically.
    :param preferred_training_mode: Must be 'cpu', if e2eml has been installed into an environment without LGBM and Xgboost on GPU.
    Can be set to 'gpu', if LGBM and Xgboost have been installed with GPU support. The default 'auto' will detect GPU support
    and optimize accordingly. (Default: 'auto')
    :param tune_mode: 'Accurate' will lead use K-fold cross validation per hyperparameter set durig optimization. 'Simple'
    will make use of use 1-fold validation only, which leads to much faster training times.
    :param logging_file_path: Preferred location to save the log file. Will otherwise stored in the current folder.
    :param low_memory_mode: Adds a preprocessing feature to reduce dataframe memory footprint. Will lead to a loss in
    model performance. Will be extended by further memory savings features in future releases.
    However we highly recommend GPU usage to heavily decrease model training times.
    """

    def __init__(self, datasource, target_variable, date_columns=None, categorical_columns=None, num_columns=None, rarity_cols=None,
                 unique_identifier=None, selected_feats=None, cat_encoded=None, cat_encoder_model=None, nlp_columns=None,
                 nlp_transformer_columns=None, transformer_chosen='bert-base-uncased', transformer_model_load_from_path=None,
                 transformer_model_save_states_path=None, transformer_epochs=25, prediction_mode=False, preferred_training_mode='auto',
                 preprocess_decisions=None, tune_mode='accurate', trained_model=None, ml_task=None,
                 logging_file_path=None, low_memory_mode=False, save_models_path=None, train_split_type='cross'):

        self.dataframe = datasource
        self.kfolds_column = None
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
                if datasource[target_variable].nunique() > 10:
                    self.class_problem = 'regression'
                elif datasource[target_variable].nunique() > 2:
                    self.class_problem = 'multiclass'
                    self.num_classes = datasource[target_variable].nunique()
                elif datasource[target_variable].nunique() == 2:
                    self.class_problem = 'binary'
                    self.num_classes = 2
                else:
                    self.class_problem = 'regression'
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
            if ml_task == 'multiclass':
                self.num_classes = datasource[target_variable].nunique()
            elif ml_task == 'binary':
                self.num_classes = 2
        print(f"Ml task is {self.class_problem}")

        if preferred_training_mode == 'cpu':
            message = """
            CPU mode has been chosen. Installing e2eml into an environment where LGBM and Xgboost have been installed with GPU acceleration
            is recommended to be able to use preferred_training_mode='gpu'. This will speed up model training and feature importance
            via SHAP. 
            """
            logging.warning(f'{message}')
            print(f'{message}')
            self.preferred_training_mode = preferred_training_mode
        elif preferred_training_mode == 'gpu':
            print('GPU acceleration chosen.')
            self.preferred_training_mode = preferred_training_mode
        elif preferred_training_mode == 'auto':
            print("Preferred training mode auto has been chosen. e2eml will automatically detect, if LGBM and Xgboost can "
                  "use GPU acceleration and optimize the workflow accordingly.")
            self.preferred_training_mode = preferred_training_mode
        else:
            self.preferred_training_mode = 'cpu'
            print('No preferred_training_mode chosen. Fallback to CPU.')
        self.tune_mode = tune_mode
        self.train_split_type = train_split_type
        self.date_columns = date_columns
        self.date_columns_created = None
        self.categorical_columns = categorical_columns
        self.rarity_cols = rarity_cols
        if isinstance(nlp_columns, list):
            print("Please provide nlp_columns parameter with a string.")
            self.nlp_columns = nlp_columns
        else:
            self.nlp_columns = nlp_columns
        self.nlp_transformer_columns = nlp_transformer_columns
        self.nlp_transformers = {}
        self.transformer_chosen = transformer_chosen
        self.transformer_epochs = transformer_epochs
        self.cat_columns_encoded = None
        self.num_columns_encoded = None
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
        self.transformer_model_load_from_path = transformer_model_load_from_path
        self.transformer_model_save_states_path = transformer_model_save_states_path
        self.transformer_settings = {f"train_batch_size": 16,
                                     "test_batch_size": 16,
                                     "pred_batch_size": 16,
                                     "num_workers": 4,
                                     "epochs": self.transformer_epochs, # TODO: Change to 20 again
                                     "transformer_model_path": self.transformer_model_load_from_path,
                                     "model_save_states_path": {self.transformer_model_save_states_path},
                                     "keep_best_model_only": False}

        # automatically determine batch sizes for Tabnet
        if not prediction_mode:
            rec_batch_size = (len(self.dataframe.index)*0.8)/20
            if int(rec_batch_size) % 2 == 0:
                rec_batch_size = int(rec_batch_size)
            else:
                rec_batch_size = int(rec_batch_size)+1

            if rec_batch_size > 8192:
                rec_batch_size = 8192
                virtual_batch_size = 1024
            else:
                virtual_batch_size = int(rec_batch_size/4)
        else:
            rec_batch_size = 8192
            virtual_batch_size = 1024

        self.tabnet_settings = {f"batch_size": rec_batch_size,
                                "virtual_batch_size": virtual_batch_size,
                                # pred batch size?
                                "num_workers": 0,
                                "max_epochs": 1000,
                                'optimization_rounds': 25}
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
        """
        Prints and return the current timestamp (not timezone aware)
        :param task: Expects a string. Can be used to inject the printed message with context.
        :return: Returns timestamp as string.
        """
        t = time.localtime()
        if task:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"Started {task} at {current_time}.")
        else:
            current_time = time.strftime("%H:%M:%S", t)
            print(f"{current_time}")
        return current_time

    def runtime_warnings(self, warn_about='shap_cpu'):
        """
        This function returns custom warnings for a better user experience.
        :return: warning message
        """
        if warn_about == 'shap_cpu':
            warning_message = """Calculating SHAP values for feature importance on CPU might run a long time. To disable
            the calculation set the parameter 'feat_importance' to False. Alternatively the LGBM and Xgboost
            blueprints can be used as well. These run on GPU by default and usually yield better
            classification results as well."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == 'long runtime':
            warning_message = """This blueprint has long runtimes. GPU acceleration is only possible for LGBM and Xgboost
            as of now. Also Ngboost is relatively fast even though it can only run on CPU."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == 'wrong null algorithm':
            warning_message = """The chosen option does not exist. Currently only "iterative_imputation" and "static" 
            exist. Any other declared option will result in not-handling of NULLs and are likely to fail later in the
             pipeline."""
            return warnings.warn(warning_message, RuntimeWarning)
        elif warn_about == 'future_architecture_change':
            warning_message = """The organization of blueprints will change in a future version to better separate NLP
            and non-NLP related preprocessing(!) blueprints. This change is likely to be live with e2eml version 2.0.0
            """
            return warnings.warn(warning_message, DeprecationWarning)
        elif warn_about == 'no_nlp_transformer':
            warning_message = """No nlp_transformer_columns have been provided during class instantiation. Some 
            NLP related functions only run with this information.."""
            return warnings.warn(warning_message, UserWarning)
        elif warn_about == 'not_enough_target_class_members':
            warning_message = """Some target classes have less members than allowed. You can ignore this message, if you
            are running a blueprint without NLP transformers.
            
            In order to create a strong model e2eml splits the data into several folds. Please provide data with at least
             6 class members for each target class. Otherwise the model is likely to fail to a CUDA error on runtime. 
             You can use the following function on your dataframe before passing it to e2eml:
            
            def handle_rarity(all_data, threshold=6, mask_as='miscellaneous', rarity_cols=None, normalize=False):
                if isinstance(rarity_cols, list):
                    for col in rarity_cols:
                        frequencies = all_data[col].value_counts(normalize=normalize)
                        condition = frequencies < threshold
                        mask_obs = frequencies[condition].index
                        mask_dict = dict.fromkeys(mask_obs, mask_as)
                        all_data[col] = all_data[col].replace(mask_dict)
                    del rarity_cols
                else:
                    pass
                return all_data
                
            Example usage:
            train_df = handle_rarity(train_df, rarity_cols=["your_target_column_name"])
            
            Important:
            This function modifies the original data. It is recommended to create a copy of your data first.
            """
            return warnings.warn(warning_message, UserWarning)
        else:
            pass

    def check_gpu_support(self, algorithm=None):
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        try:
            if not self.preprocess_decisions[f"gpu_support"]:
                self.preprocess_decisions[f"gpu_support"] = {}
        except KeyError:
            self.preprocess_decisions[f"gpu_support"] = {}
        else:
            pass
        if algorithm == 'lgbm':
            self.get_current_timestamp(task='Check LGBM for GPU acceleration.')
            train_data = lightgbm.Dataset(data, label=label)
            params = {'num_iterations': 1, 'device': 'gpu'}
            try:
                gbm = lightgbm.train(params, train_set=train_data)
                self.preprocess_decisions[f"gpu_support"][f"{algorithm}"] = 'gpu'
                print('LGBM uses GPU.')
            except Exception:
                self.preprocess_decisions[f"gpu_support"][f"{algorithm}"] = 'cpu'
                print('LGBM uses CPU.')
        elif algorithm == 'xgboost':
            self.get_current_timestamp(task='Check Xgboost for GPU acceleration.')
            D_train = xgb.DMatrix(data, label=label)
            params = {'tree_method': 'gpu_hist', 'steps': 2}
            try:
                model = xgb.train(params, D_train)
                self.preprocess_decisions[f"gpu_support"][f"{algorithm}"] = 'gpu_hist'
                print('Xgboost uses GPU.')
            except Exception:
                self.preprocess_decisions[f"gpu_support"][f"{algorithm}"] = 'exact'
                print('Xgboost uses CPU.')
        else:
            print("No algorithm has been checked for GPU acceleration.")

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
        """
        Function to save and load class instances. This function shall be used to save whole blueprints and to
        reload them.
        :param model_object: The blueprint class instance to be saved.
        :param model_path: Expects a string. The path to save the model to or load from.
        :param algorithm: Expects a string. Used to name the final stored file.
        :param algorithm_variant: Expects a string. Used to name the final stored file.
        :param file_type: File type to be saved as. Default ".dat"
        :param action: Chose 'save' or 'load'.
        :param clean: Expects a boolean. If True, deletes the provided class instance instantly after saving.
        :return: When action is 'load', returns the loaded blueprint class instance.
        """
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
        Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        :param df: Expects a Pandas dataframe.
        :return: Returns downcasted dataframe.
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

    def calc_scale_pos_weight(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        try:
            pos_labels = Y_train.sum()
        except Exception:
            pos_labels = np.sum(Y_train)

        try:
            neg_labels = len(X_train.index)-pos_labels
        except Exception:
            neg_labels = len(Y_train)-np.sum(Y_train)

        scale_pos_weight = neg_labels/pos_labels
        return scale_pos_weight

    def label_encoder_decoder(self, target, mode='fit'):
        """
        Takes a Pandas series and encodes string-based labels to numeric values. Flags previously unseen
        values with -1.
        :param target: Expects Pandas Series.
        :param mode: 'Chose' fit to create label encoding dictionary and 'transform' the labels. Chose 'transform'
        to encode labels based on already created dictionary.
        :return: Returns Pandas Series.
        """
        self.get_current_timestamp(task='Execute label encoding')
        logging.info('Started label encoding.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')

        def label_encoder_fit(pandas_series):
            pandas_series = pandas_series.astype('category')
            col = pandas_series.name
            try:
                pandas_series = pandas_series.to_frame()
            except Exception:
                pass
            values = pandas_series[col].unique()
            cat_mapping = {}
            for label, cat in enumerate(values):
                cat_mapping[cat] = label
            return cat_mapping

        def label_encoder_transform(pandas_series, mapping):
            pandas_series = pandas_series.astype('category')
            col = pandas_series.name
            try:
                pandas_series = pandas_series.to_frame()
            except Exception:
                pass
            mapping = self.preprocess_decisions["label_encoder_mapping"]
            pandas_series[col] = pandas_series[col].apply(lambda x: mapping.get(x, 999))
            #pandas_series = pandas_series[col]
            return pandas_series

        if self.prediction_mode:
            target = label_encoder_transform(target, self.preprocess_decisions["label_encoder_mapping"])
        else:
            if mode == 'fit':
                cat_mapping = label_encoder_fit(target)
                self.preprocess_decisions["label_encoder_mapping"] = cat_mapping
            else:
                pass
            target = label_encoder_transform(target, self.preprocess_decisions["label_encoder_mapping"])
        self.labels_encoded = True
        if self.class_problem == 'binary' or self.class_problem == 'multiclass':
            target = target[self.target_variable].astype(int)
        elif self.class_problem == 'regression':
            target = target[self.target_variable].astype(float)
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        return target

    def check_max_sentence_length(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            text_columns = self.nlp_transformer_columns
            sentence_length = X_train[text_columns].apply(lambda x: np.max([len(w) for w in x.split()]))
            if "nlp_transformers" in self.preprocess_decisions:
                pass
            else:
                self.preprocess_decisions[f"nlp_transformers"] = {}
            self.preprocess_decisions[f"nlp_transformers"][f"max_sentence_len"] = sentence_length.max()

    def data_scaling(self, scaling='minmax'):
        """
        Scales the data using the chosen scaling algorithm.
        :param scaling: Chose 'minmax'.
        :return: Returns scaled dataframes
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
        """
        Loops through the in-class stored dataframe columns and checks the skewness. If skewness exceeds a certain threshold,
        executes log transformation.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Remove skewness')
        if self.prediction_mode:
            logging.info('Started skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            for col in self.preprocess_decisions["skewed_columns"]:
                log_array = np.log1p(self.dataframe[col])
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
            left_skewed = skewness[skewness < -0.75].index.to_list()
            right_skewed = skewness[skewness > 0.75].index.to_list()
            skewed = left_skewed+right_skewed
            for col in X_train[skewed].columns:
                log_array = np.log1p(X_train[col])
                log_array[np.isfinite(log_array) == False] = 0
                X_train[col] = log_array
                log_array = np.log1p(X_test[col])
                log_array[np.isfinite(log_array) == False] = 0
                X_test[col] = log_array
            logging.info('Finished skewness removal.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.preprocess_decisions["skewed_columns"] = skewed
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train,
                                                Y_test)

    def create_folds(self, data, target, num_splits=5, mode='advanced'):
        if self.prediction_mode:
            pass
        else:
            if mode == 'simple':
                data["kfold"] = data.index % num_splits
            else:
                # we create a new column called kfold and fill it with -1
                data["kfold"] = -1

                # the next step is to randomize the rows of the data
                data = data.sample(frac=1).reset_index(drop=True)
                print(data.info())

                # calculate number of bins by Sturge's rule
                # I take the floor of the value, you can also
                # just round it
                num_bins = int(np.floor(1 + np.log2(len(data))))
                # bin targets
                data.loc[:, "bins"] = pd.cut(
                    data[target], bins=num_bins, labels=False
                )
                print(data.info())
                # initiate the kfold class from model_selection module
                kf = model_selection.StratifiedKFold(n_splits=num_splits)
                # fill the new kfold column
                # note that, instead of targets, we use bins!
                for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
                    data.loc[v_, 'kfold'] = f
                # drop the bins column
                data = data.drop("bins", axis=1)
                # return dataframe with folds
            return data

    def reset_test_train_index(self, drop_target=False):
        if self.prediction_mode:
            self.dataframe = self.dataframe.reset_index(drop=True)
        else:
            # index shuffling
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train[self.target_variable] = Y_train
            X_test[self.target_variable] = Y_test
            all_data = pd.concat([X_train, X_test])
            all_data = self.create_folds(all_data, self.target_variable)
            X_train = all_data[all_data["kfold"] != 0].reset_index(drop=True)
            X_test = all_data[all_data["kfold"] == 0].reset_index(drop=True)
            Y_train = X_train[self.target_variable]
            Y_test = X_test[self.target_variable]
            X_train.drop("kfold", axis=1)
            X_test.drop("kfold", axis=1)
            if drop_target:
                X_train.drop(self.target_variable, axis=1)
                X_test.drop(self.target_variable, axis=1)
            else:
                pass
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

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
        elif how == 'cross':
            logging.info('Started test train split.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.check_target_class_distribution()
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(self.dataframe,
                                                                                    self.dataframe[self.target_variable],
                                                                                    train_size=train_size,
                                                                                    random_state=42)
            try:
                Y_train = Y_train.astype(float)
                Y_test = Y_test.astype(float)
            except Exception:
                Y_train = self.label_encoder_decoder(Y_train, mode='fit')
                Y_test = self.label_encoder_decoder(Y_test, mode='transform')
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

    def check_target_class_distribution(self):
        if self.prediction_mode:
            pass
        else:
            min_target_train = self.dataframe[self.target_variable].value_counts().min()
            if min_target_train < 7:
                self.runtime_warnings(warn_about='not_enough_target_class_members')
            else:
                pass

    def set_random_seed(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def apply_k_folds(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = self.create_folds(X_train, num_splits=4)
            X_test["kfold"] = 0
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def data_binning(self, nb_bins=10):
        """
        Takes numerical columns and splits them into desired number of bins. Bins will be attached as
        new columns to the dataframe.
        :param nb_bins: Takes a positive integer.
        :return: Updates class instance.
        """
        self.get_current_timestamp(task='Execute numerical binning')

        def binning_on_data(dataframe, cols_to_bin=None):
            num_columns = cols_to_bin.select_dtypes(include=[vartype]).columns
            for col in num_columns:
                dataframe[str(col) + '_binned'] = pd.cut(dataframe[col].replace(np.inf, np.nan).dropna(), bins=nb_bins, labels=False)
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

    def rare_feature_processor(self, threshold=0.005, mask_as='miscellaneous', rarity_cols=None, normalize=True):
        """
        Loops through categorical columns and identifies categories, which occur less than the
        given threshold. These features will be grouped together as defined by mask_as parameter.
        :param threshold: Minimum share of categories to be not grouped as misc. Takes a float between 0 and 1.
        :param mask_as: Group name of grouped rare features.
        :return: Updates class attributes
        """
        self.get_current_timestamp('Handle rare features')

        def handle_rarity(all_data, threshold=threshold, mask_as=mask_as, rarity_cols=rarity_cols, normalize=normalize):
            if isinstance(rarity_cols, list):
                for col in rarity_cols:
                    frequencies = all_data[col].value_counts(normalize=normalize)
                    condition = frequencies < threshold
                    mask_obs = frequencies[condition].index
                    mask_dict = dict.fromkeys(mask_obs, mask_as)
                    all_data[col] = all_data[col].replace(mask_dict)  # or you could make a copy not to modify original data
                del rarity_cols
                _ = gc.collect()
            return all_data

        logging.info('Start rare feature processing.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            threshold = self.preprocess_decisions["rare_feature_threshold"]
            self.dataframe = handle_rarity(self.dataframe, threshold, mask_as, rarity_cols, normalize)
            logging.info('Finished rare feature processing.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = handle_rarity(X_train, threshold, mask_as, rarity_cols, normalize)
            X_test = handle_rarity(X_test, threshold, mask_as, rarity_cols, normalize)
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
            dataframe[f'dbscan_cluster_{eps}'] = labels
            del db
            del labels
            _ = gc.collect()
            return dataframe

        def add_gaussian_mixture_clusters(dataframe, n_components=nb_clusters):
            gaussian = GaussianMixture(n_components=n_components)
            gaussian.fit(dataframe)
            gaussian_clusters = gaussian.predict(dataframe)
            dataframe[f"gaussian_clusters_{n_components}"] = gaussian_clusters
            del gaussian
            del gaussian_clusters
            _ = gc.collect()
            return dataframe

        def add_kmeans_clusters(dataframe, n_components=nb_clusters):
            kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=20, max_iter=500)
            kmeans.fit(dataframe)
            kmeans_clusters = kmeans.predict(dataframe)
            dataframe[f"kmeans_clusters_{n_components}"] = kmeans_clusters
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

    def iterative_imputation(self, dataframe, imputer=None):
        dataframe_cols = dataframe.columns#[dataframe.isna().any()].tolist()
        imp_mean = IterativeImputer(random_state=0, estimator=BayesianRidge(), imputation_order='ascending')
        if not imputer:
            imp_mean.fit(dataframe)
        else:
            imp_mean = imputer
        dataframe = imp_mean.transform(dataframe)
        dataframe_final = pd.DataFrame(dataframe, columns=dataframe_cols)
        self.preprocess_decisions[f"fill_nulls_imputer"] = imp_mean
        del imp_mean
        _ = gc.collect()
        return dataframe_final

    def static_filling(self, dataframe, fill_with=0, fill_cat_col_with='None'):
        """
        Loop through dataframe and fill categorical and numeric columns seperately with predefined values.
        :param dataframe: Pandas Dataframe
        :param fill_with: Numeric value to fill with
        :param fill_cat_col_with: String to fill categorical NULLs.
        :return:
        """
        cat_columns = dataframe.select_dtypes(include=['object']).columns.to_list()
        for col in cat_columns:
            dataframe[col] = dataframe[col].fillna(fill_cat_col_with, inplace=False)

        for vartype in self.num_dtypes:
            try:
                filtered_columns = dataframe.select_dtypes(include=[vartype]).columns.to_list()
                for col in filtered_columns:
                    dataframe[col] = dataframe[col].fillna(fill_with, inplace=False)
            except ValueError:
                pass
        return dataframe

    # TODO: Check if parameters can be used via **kwargs argument
    def fill_nulls(self, how='iterative_imputation', fill_with=0, fill_cat_col_with='None'):
        """
        Takes in a dataframe and fills all NULLs with chosen value.
        :param fill_with: Define value to replace NULLs with.
        :param how: Chose 'static' to define static fill values, 'iterative_imputation' for the sklearns iterative
        imputer.
        :return: Returns modified dataframe
        """
        self.get_current_timestamp('Fill nulls')

        logging.info('Started filling NULLs.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        algorithms = ["iterative_imputation", "static"]
        if how not in algorithms:
            self.runtime_warnings(warn_about='wrong null algorithm')
        else:
            pass

        if self.prediction_mode:
            if not how:
                how = self.preprocess_decisions[f"fill_nulls_how"]
            else:
                pass

            if how == 'static':
                self.dataframe = self.static_filling(self.dataframe, fill_with=fill_with, fill_cat_col_with=fill_cat_col_with)
            elif how == 'iterative_imputation':
                self.dataframe = self.iterative_imputation(self.dataframe, imputer=self.preprocess_decisions[f"fill_nulls_imputer"])
            logging.info('Finished filling NULLs.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if how == 'static':
                X_train = self.static_filling(X_train, fill_with=fill_with, fill_cat_col_with=fill_cat_col_with)
                X_test = self.static_filling(X_test, fill_with=fill_with, fill_cat_col_with=fill_cat_col_with)
                self.preprocess_decisions[f"fill_nulls_how"] = how
            elif how == 'iterative_imputation':
                # TODO: Test, if it woks + revert LGBM + test model ensemble
                X_train = self.iterative_imputation(X_train)
                X_test = self.iterative_imputation(X_test, imputer=self.preprocess_decisions[f"fill_nulls_imputer"])
                self.preprocess_decisions[f"fill_nulls_how"] = how
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

    def iqr_remover(self, threshold=1.5):
        """
        Remove outliers from a dataframe by column, including optional whiskers, removing rows for which the column value
         are less than Q1-1.5IQR or greater than Q3+1.5IQR.
        :param threshold: whisker_width (float): Optional, loosen the IQR filter by a factor of `whisker_width` * IQR.
        Default is 1.5.
        :return: Updates class attributed.
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
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = cos_sin_transformation(X_train)
            X_test = cos_sin_transformation(X_test)
        elif datetime_handling == 'categorical':
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = date_converter(X_train)
            X_test = date_converter(X_test)
        elif datetime_handling == 'all':
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = date_converter(X_train)
            X_train = cos_sin_transformation(X_train)
            X_test = date_converter(X_test)
            X_test = cos_sin_transformation(X_test)
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

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
        """
        Takes categorical columns, executes onehot encoding on them and reduces dimensionality with PCA.
        :return: Updates class attributes.
        """
        self.get_current_timestamp(task='Onehot + PCA categorical features')
        logging.info('Started Onehot + PCA categorical features.')
        if self.prediction_mode:
            if len(self.cat_columns_encoded) > 0:
                df_branch = self.dataframe[self.cat_columns_encoded].copy()
                enc = self.preprocess_decisions[f"onehot_pca"]["onehot_encoder"]
                df_branch = enc.transform(df_branch[self.cat_columns_encoded])
                df_branch.fillna(0, inplace=True)
                onehot_cols = df_branch.columns
                # pca = self.preprocess_decisions[f"onehot_pca"]["pca_encoder"]
                pca = PCA(n_components=2)
                pred_comps = pca.fit_transform(df_branch[onehot_cols])
                df_branch = pd.DataFrame(pred_comps, columns=['PC-1', 'PC-2'])
                for col in df_branch.columns:
                    self.dataframe[f"{col}_pca"] = df_branch[col]
                del df_branch
                del pca
                del pred_comps
                del enc
                _ = gc.collect()
            else:
                pass
            logging.info('Finished Onehot + PCA categorical features.')
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
                #pac = pacmap.PaCMAP(n_dims=2)
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
                self.preprocess_decisions[f"onehot_pca"]["pca_encoder"] = pca
                del X_train_branch
                del X_test_branch
                del pca
                del train_comps
                del test_comps
                _ = gc.collect()
            else:
                pass
            logging.info('Finished Onehot + PCA categorical features.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def numeric_binarizer_pca(self):
        self.get_current_timestamp(task='Binarize numeric columns + PCA binarized features')
        logging.info('Started to binarize numeric columns + PCA binarized features.')
        if self.prediction_mode:
            if len(self.num_columns_encoded) > 0:
                num_cols_binarized_created = []
                for num_col in self.num_columns_encoded:
                    self.dataframe[num_col+"_binarized"] = self.dataframe[num_col].apply(lambda x: 1 if x > 0 else 0)
                    num_cols_binarized_created.append(num_col+"_binarized")
                pca = PCA(n_components=2)
                df_branch = self.dataframe.copy()
                pred_comps = pca.fit_transform(df_branch[num_cols_binarized_created])
                df_branch = pd.DataFrame(pred_comps, columns=['Num_PC-1', 'Num_PC-2'])
                for col in df_branch.columns:
                    self.dataframe[f"{col}_num_pca"] = df_branch[col]
                del df_branch
                del pred_comps
                del pca
                _ = gc.collect()
            else:
                pass
            logging.info('Finished to binarize numeric columns + PCA binarized features.')
            return self.dataframe
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions[f"numeric_binarizer_pca"] = {}

            encoded_num_cols = []
            for vartype in self.num_dtypes:
                try:
                    filtered_columns = X_train.select_dtypes(include=[vartype]).columns.to_list()
                    for pcas in filtered_columns:
                        try:
                            filtered_columns.remove("Num_PC-1_num_pca")
                            filtered_columns.remove("Num_PC-2_num_pca")
                        except Exception:
                            pass
                    for i in filtered_columns:
                        try:
                            encoded_num_cols.remove(i)
                        except Exception:
                            pass

                    if len(filtered_columns) > 0:
                        num_cols_binarized_created = []
                        for num_col in filtered_columns:
                            X_train[num_col+"_binarized"] = X_train[num_col].apply(lambda x: 1 if x > 0 else 0)
                            X_test[num_col+"_binarized"] = X_test[num_col].apply(lambda x: 1 if x > 0 else 0)
                            num_cols_binarized_created.append(num_col+"_binarized")
                            encoded_num_cols.append(num_col)
                        pca = PCA(n_components=2)
                        X_train_branch = X_train.copy()
                        X_test_branch = X_test.copy()
                        train_comps = pca.fit_transform(X_train_branch[num_cols_binarized_created])
                        test_comps = pca.fit_transform(X_test_branch[num_cols_binarized_created])
                        X_train_branch = pd.DataFrame(train_comps, columns=['Num_PC-1', 'Num_PC-2'])
                        X_test_branch = pd.DataFrame(test_comps, columns=['Num_PC-1', 'Num_PC-2'])
                        pca_cols = []
                        for col in X_train_branch.columns:
                            X_train[f"{col}_num_pca"] = X_train_branch[col]
                            X_test[f"{col}_num_pca"] = X_test_branch[col]
                            pca_cols.append(f"{col}_num_pca")
                        self.preprocess_decisions[f"numeric_binarizer_pca"][f"pca_cols_{vartype}"] = pca_cols
                        del X_train_branch
                        del X_test_branch
                        del train_comps
                        del test_comps
                        del pca
                        _ = gc.collect()
                    else:
                        pass
                except ValueError:
                    pass
            self.num_columns_encoded = encoded_num_cols
            logging.info('Finished to binarize numeric columns + PCA binarized features.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def category_encoding(self, algorithm='target'):
        """
        Takes in a dataframe and applies the chosen category encoding algorithm to categorical columns.
        :param algorithm: Chose type of encoding as 'target' (default), 'onehot', 'woee', 'ordinal', 'leaveoneout' and 'GLMM'.
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
        :return: Returns modified dataframe.
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

    def naive_undersampling(self, df, target_name):
        """
        Takes a dataframe and the name of the target variable to undersample all classes other than the minority class.
        This is a naive undersampling technique and should only be done on the training dataset.
        :param df: Expects a Pandas dataframe.
        :param target_name: Expects a string with the name of the target column.
        :return: Returns the modified Pandas dataframe.
        """
        classes = df[target_name].value_counts().to_dict()
        least_class_amount = min(classes.values())
        classes_list = []
        for key in classes:
            classes_list.append(df[df[target_name] == key])
        classes_sample = []
        for i in range(0, len(classes_list)-1):
            classes_sample.append(classes_list[i].sample(least_class_amount, random_state=50))
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([df_maybe, classes_list[-1]], axis=0)
        final_df = final_df.reset_index(drop=True)
        return final_df

    def naive_oversampling(self, df, target_name):
        classes = df[target_name].value_counts().to_dict()
        most = max(classes.values())
        classes_list = []
        for key in classes:
            classes_list.append(df[df[target_name] == key])
        classes_sample = []
        for i in range(1, len(classes_list)):
            classes_sample.append(classes_list[i].sample(most, replace=True, random_state=50))
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
        final_df = final_df.reset_index(drop=True)
        return final_df

    def undersample_train_data(self):
        if self.prediction_mode:
            pass
        else:
            if self.class_problem == 'regression':
                pass
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train[self.target_variable] = Y_train
                X_train = self.naive_undersampling(X_train, self.target_variable)
                Y_train = X_train[self.target_variable]
                X_train.drop(self.target_variable, axis=1)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def oversample_train_data(self):
        if self.prediction_mode:
            pass
        else:
            if self.class_problem == 'regression':
                pass
            else:
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                X_train[self.target_variable] = Y_train
                X_train = self.naive_oversampling(X_train, self.target_variable)
                #X_train = X_train.sample(frac=0.50)
                Y_train = X_train[self.target_variable]
                X_train.drop(self.target_variable, axis=1)
                return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def automated_feature_selection(self, metric=None):
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
                print(f"Features before selection are...{col}")
            if metric:
                metric = metric
            elif self.class_problem == 'binary':
                metric = 'logloss'
            elif self.class_problem == 'multiclass':
                metric = 'mlogloss'
            elif self.class_problem == 'regression':
                metric = 'mae'
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



