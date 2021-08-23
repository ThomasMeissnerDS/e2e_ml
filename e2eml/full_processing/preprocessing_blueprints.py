from e2eml.full_processing.postprocessing import FullPipeline
from e2eml.full_processing.cpu_processing_nlp import NlpPreprocessing
import logging


class PreprocessingBluePrint(FullPipeline, NlpPreprocessing):
    def pp_bp01_std_preprocessing(self, df=None, preprocessing_type='full'):
        """
        Our recommended blueprint for Tabnet testing.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
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
        self.pos_tagging_pca(pca_pos_tags=True)
        if preprocessing_type == 'nlp':
            self.append_text_sentiment_score()
            self.tfidf_vectorizer_to_pca(pca_pos_tags=True)
            self.tfidf_naive_bayes_proba(analyzer="char_wb", ngram_range=(1, 2))
            self.tfidf_naive_bayes_proba(analyzer="word", ngram_range=(1, 1))
        self.cardinality_remover(threshold=100)
        self.delete_high_null_cols(threshold=0.5)
        self.onehot_pca()
        self.numeric_binarizer_pca()
        self.category_encoding(algorithm='target')
        self.fill_nulls(how='static') # can only be here when "static"
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.skewness_removal()
        try:
            self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        except ValueError:
            print("Clustering as a feature skipped due to ValueError.")
        for nb_cluster in [3, 5, 7, 9]:
            try:
                self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        for nb_cluster in [2, 4, 6, 8, 10]:
            try:
                self.clustering_as_a_feature(algorithm='gaussian', nb_clusters=nb_cluster)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection()
        self.sort_columns_alphabetically()

    def pp_bp02_std_preprocessing(self, df=None, preprocessing_type='full'):
        """
        This preprocessing blueprint contains alternative decision compare to pp_bp01.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
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
        self.pos_tagging_pca(pca_pos_tags=True)
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca(pca_pos_tags=False)
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='GLMM')
        self.delete_high_null_cols(threshold=0.4)
        self.fill_nulls(how='iterative_imputation', fill_with=0)
        self.data_binning(nb_bins=20)
        #self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.85)
        try:
            self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        except ValueError:
            print("Clustering as a feature skipped due to ValueError.")
        for nb_cluster in range(2, 10):
            try:
                self.clustering_as_a_feature(algorithm='GLMM', nb_clusters=nb_cluster)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection()
        self.sort_columns_alphabetically()

    def pp_bp03_std_preprocessing(self, df=None, preprocessing_type='full'):
        """
        This blueprint adds skewness removal by log transformation, data scaling and SMOTE.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
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
        self.pos_tagging_pca(pca_pos_tags=True)
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca(pca_pos_tags=False)
            #self.tfidf_vectorizer_to_pca(pca_pos_tags=True)
            self.tfidf_naive_bayes_proba()
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='GLMM')
        self.delete_high_null_cols(threshold=0.4)
        self.fill_nulls(how='static', fill_with=-99)
        self.data_binning(nb_bins=5)
        self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.80)
        try:
            self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        except ValueError:
            print("Clustering as a feature skipped due to ValueError.")
        for nb_cluster in range(2, 10):
            try:
                self.clustering_as_a_feature(algorithm='GLMM', nb_clusters=nb_cluster)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection()
        self.sort_columns_alphabetically()
        self.data_scaling()
        if self.class_problem == 'binary' or self.class_problem == 'multiclass':
            self.smote_data()
        else:
            pass

    def pp_bp04_std_preprocessing(self, df=None, preprocessing_type='full'):
        """
        Our recommended blueprint for model testing.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
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
        self.pos_tagging_pca(pca_pos_tags=True)
        if preprocessing_type == 'nlp':
            self.append_text_sentiment_score()
            self.pos_tagging_pca(pca_pos_tags=False)
            self.tfidf_vectorizer_to_pca(pca_pos_tags=True)
            self.tfidf_naive_bayes_proba(analyzer="char_wb", ngram_range=(1, 2))
            self.tfidf_naive_bayes_proba(analyzer="word", ngram_range=(1, 1))
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(how='static')
        self.data_binning(nb_bins=10)
        #self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        try:
            self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        except ValueError:
            print("Clustering as a feature skipped due to ValueError.")
        for nb_cluster in range(2, 10):
            try:
                self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection()
        self.sort_columns_alphabetically()


    def pp_bp10_nlp_preprocessing(self, df):
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.sort_columns_alphabetically()
        self.check_max_sentence_length()
        self.import_transformer_model_tokenizer(transformer_chosen=self.transformer_chosen)

    def pp_bp11_nlp_preprocessing(self, df):
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.regex_clean_text_data()
        self.sort_columns_alphabetically()
        self.check_max_sentence_length()
        self.import_transformer_model_tokenizer(transformer_chosen=self.transformer_chosen)

    def pp_bp12_nlp_preprocessing(self, df):
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.oversample_train_data()
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.sort_columns_alphabetically()
        self.check_max_sentence_length()
        self.import_transformer_model_tokenizer(transformer_chosen=self.transformer_chosen)

    def pp_bp13_nlp_preprocessing(self, df):
        logging.info('Start blueprint.')
        self.runtime_warnings(warn_about="future_architecture_change")
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False
        self.train_test_split(how=self.train_split_type)
        self.replace_synonyms_to_df_copy(words_to_replace=3, mode='auto')
        #self.oversample_train_data()
        self.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=self.rarity_cols)
        self.sort_columns_alphabetically()
        self.check_max_sentence_length()
        self.import_transformer_model_tokenizer(transformer_chosen=self.transformer_chosen)
