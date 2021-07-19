from e2eml.classification.classification_models import ClassificationModels
from e2eml.full_processing.cpu_processing_nlp import NlpPreprocessing
import logging


class PreprocessingBluePrint(ClassificationModels, NlpPreprocessing):
    def pp_bp01_preprocessing(self, df=None, preprocessing_type='nlp'):
        """
        Our recommended blueprint for model testing.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all')
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca()
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(how='static')
        self.data_binning(nb_bins=10)
        #self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='logloss')
        self.sort_columns_alphabetically()

    def pp_bp02_preprocessing(self, df=None, preprocessing_type='full'):
        """
        This preprocessing blueprint contains alternative decision compare to pp_bp01.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all')
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca()
        self.rare_feature_processor(threshold=0.02, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='GLMM')
        self.delete_high_null_cols(threshold=0.4)
        self.fill_nulls(how='iterative_imputation', fill_with=0)
        self.data_binning(nb_bins=20)
        #self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.85)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='GLMM', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='logloss')
        self.sort_columns_alphabetically()

    def pp_bp03_preprocessing(self, df=None, preprocessing_type='full'):
        """
        This blueprint adds skewness removal by log transformation, data scaling and SMOTE.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all')
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca()
        self.rare_feature_processor(threshold=0.02, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='GLMM')
        self.delete_high_null_cols(threshold=0.4)
        self.fill_nulls(how='static', fill_with=-99)
        self.data_binning(nb_bins=5)
        self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.80)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='GLMM', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='logloss')
        self.sort_columns_alphabetically()
        self.data_scaling()
        self.smote_data()