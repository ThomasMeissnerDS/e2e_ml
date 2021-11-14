import pandas as pd
import numpy as np
import warnings
import logging
import dill as pickle
import gc
from e2eml.classification.classification_blueprints import ClassificationBluePrint
from e2eml.regression.regression_blueprints import RegressionBluePrint
from e2eml.full_processing.postprocessing import save_to_production


class TimeTravelBaseClass:
    def __init__(self):
        self.checkpoints = {
            "automatic_type_detection_casting": False,
            "early_numeric_only_feature_selection": False,
            "remove_duplicate_column_names": False,
            "reset_dataframe_index": False,
            "regex_clean_text_data": False,
            "handle_target_skewness": False,
            "holistic_null_filling": True, # slow
            "iterative_null_imputation": True,
            "fill_infinite_values": True,
            "datetime_converter": True,
            "pos_tagging_pca": True, # slow with many categories
            "append_text_sentiment_score": True,
            "tfidf_vectorizer_to_pca": True, # slow with many categories
            "tfidf_vectorizer": True,
            "rare_feature_processing": True,
            "cardinality_remover": True,
            "delete_high_null_cols": True,
            "numeric_binarizer_pca": True,
            "onehot_pca": True,
            "category_encoding": True,
            "fill_nulls_static": True,
            "data_binning": True,
            "outlier_care": True,
            "remove_collinearity": True,
            "skewness_removal": True,
            "autotuned_clustering": True,
            "clustering_as_a_feature_dbscan": True,
            "clustering_as_a_feature_kmeans_loop": True,
            "clustering_as_a_feature_gaussian_mixture_loop": True,
            "pca_clustering_results": True,
            "reduce_memory_footprint": True,
            "automated_feature_selection": True,
            "bruteforce_random_feature_selection": True, # slow
            "sort_columns_alphabetically": True,
            "synthetic_data_augmentation": True,
            "delete_unpredictable_training_rows": True,
            "scale_data": True,
            "smote": True,
            "autoencoder_based_oversampling": True,
            "final_pca_dimensionality_reduction": True
        }
        self.checkpoint_reached = {}
        for key, value in self.checkpoints.items():
            self.checkpoint_reached[key] = False


class TimeTravel(TimeTravelBaseClass):

    def time_travel_pipeline(self, class_instance, df=None):
        logging.info('Start blueprint.')
        class_instance.runtime_warnings(warn_about="future_architecture_change")
        class_instance.check_prediction_mode(df)

        class_instance.train_test_split(how=class_instance.train_split_type)
        class_instance.binary_imbalance()
        if class_instance.blueprint_step_selection_non_nlp["automatic_type_detection_casting"]:
            class_instance.automatic_type_detection_casting()
            if class_instance.checkpoints["automatic_type_detection_casting"]:
                save_to_production(class_instance=class_instance, file_name="automatic_type_detection_casting", clean=False)
                class_instance.checkpoint_reached["automatic_type_detection_casting"] = True

        if class_instance.blueprint_step_selection_non_nlp["remove_duplicate_column_names"]:
            class_instance.remove_duplicate_column_names()
            if class_instance.checkpoints["remove_duplicate_column_names"]:
                save_to_production(class_instance=class_instance, file_name="remove_duplicate_column_names", clean=False)
                class_instance.checkpoint_reached["remove_duplicate_column_names"] = True

        if class_instance.blueprint_step_selection_non_nlp["reset_dataframe_index"]:
            class_instance.reset_dataframe_index()
            if class_instance.checkpoints["reset_dataframe_index"]:
                save_to_production(class_instance=class_instance, file_name="reset_dataframe_index", clean=False)
                class_instance.checkpoint_reached["reset_dataframe_index"] = True

        if class_instance.blueprint_step_selection_non_nlp["fill_infinite_values"]:
            class_instance.fill_infinite_values()
            if class_instance.checkpoints["fill_infinite_values"]:
                save_to_production(class_instance=class_instance, file_name="fill_infinite_values", clean=False)
                class_instance.checkpoint_reached["fill_infinite_values"] = True

        if class_instance.blueprint_step_selection_non_nlp["early_numeric_only_feature_selection"]:
            class_instance.automated_feature_selection(numeric_only=True)
            if class_instance.checkpoints["early_numeric_only_feature_selection"]:
                save_to_production(class_instance=class_instance, file_name="early_numeric_only_feature_selection", clean=False)
                class_instance.checkpoint_reached["early_numeric_only_feature_selection"] = True

        if class_instance.blueprint_step_selection_non_nlp["delete_high_null_cols"]:
            class_instance.delete_high_null_cols(threshold=0.05)
            if class_instance.checkpoints["delete_high_null_cols"]:
                save_to_production(class_instance=class_instance, file_name="delete_high_null_cols", clean=False)
                class_instance.checkpoint_reached["delete_high_null_cols"] = True

        if class_instance.blueprint_step_selection_non_nlp["data_binning"]:
            class_instance.data_binning(nb_bins=10)
            if class_instance.checkpoints["data_binning"]:
                save_to_production(class_instance=class_instance, file_name="data_binning", clean=False)
                class_instance.checkpoint_reached["data_binning"] = True

        if class_instance.blueprint_step_selection_non_nlp["regex_clean_text_data"] and len(class_instance.nlp_transformer_columns)>0:
            class_instance.regex_clean_text_data()
            if class_instance.checkpoints["regex_clean_text_data"]:
                save_to_production(class_instance=class_instance, file_name="regex_clean_text_data", clean=False)
                class_instance.checkpoint_reached["regex_clean_text_data"] = True

        class_instance.target_skewness_handling(mode='fit')
        #self.fill_nulls(how='static') # can only be here when "static"
        if class_instance.blueprint_step_selection_non_nlp["datetime_converter"]:
            class_instance.datetime_converter(datetime_handling='all')
            if class_instance.checkpoints["datetime_converter"]:
                save_to_production(class_instance=class_instance, file_name="datetime_converter", clean=False)
                class_instance.checkpoint_reached["datetime_converter"] = True

        if class_instance.blueprint_step_selection_non_nlp["pos_tagging_pca"]:
            class_instance.pos_tagging_pca(pca_pos_tags=True)
            if class_instance.checkpoints["pos_tagging_pca"]:
                save_to_production(class_instance=class_instance, file_name="pos_tagging_pca", clean=False)
                class_instance.checkpoint_reached["pos_tagging_pca"] = True

        if class_instance.blueprint_step_selection_non_nlp["append_text_sentiment_score"] and len(class_instance.nlp_transformer_columns)>0:
            class_instance.append_text_sentiment_score()
            if class_instance.checkpoints["append_text_sentiment_score"]:
                save_to_production(class_instance=class_instance, file_name="append_text_sentiment_score", clean=False)
                class_instance.checkpoint_reached["append_text_sentiment_score"] = True

        if class_instance.blueprint_step_selection_non_nlp["tfidf_vectorizer_to_pca"]:
            class_instance.tfidf_vectorizer_to_pca(pca_pos_tags=True)
            if class_instance.checkpoints["tfidf_vectorizer_to_pca"]:
                save_to_production(class_instance=class_instance, file_name="tfidf_vectorizer_to_pca", clean=False)
                class_instance.checkpoint_reached["tfidf_vectorizer_to_pca"] = True

        if class_instance.blueprint_step_selection_non_nlp["tfidf_vectorizer"] and len(class_instance.nlp_transformer_columns)>0:
            class_instance.tfidf_vectorizer_to_pca(pca_pos_tags=False)
            if class_instance.checkpoints["tfidf_vectorizer"]:
                save_to_production(class_instance=class_instance, file_name="tfidf_vectorizer", clean=False)
                class_instance.checkpoint_reached["tfidf_vectorizer"] = True

        if class_instance.blueprint_step_selection_non_nlp["rare_feature_processing"]:
            class_instance.rare_feature_processor(threshold=0.005, mask_as='miscellaneous', rarity_cols=class_instance.rarity_cols)
            if class_instance.checkpoints["rare_feature_processing"]:
                save_to_production(class_instance=class_instance, file_name="rare_feature_processing", clean=False)
                class_instance.checkpoint_reached["rare_feature_processing"] = True

        if class_instance.blueprint_step_selection_non_nlp["cardinality_remover"]:
            class_instance.cardinality_remover(threshold=100)
            if class_instance.checkpoints["cardinality_remover"]:
                save_to_production(class_instance=class_instance, file_name="cardinality_remover", clean=False)
                class_instance.checkpoint_reached["cardinality_remover"] = True

        if class_instance.blueprint_step_selection_non_nlp["holistic_null_filling"]:
            class_instance.holistic_null_filling(iterative=class_instance.blueprint_step_selection_non_nlp["iterative_null_imputation"])
            if class_instance.checkpoints["holistic_null_filling"]:
                save_to_production(class_instance=class_instance, file_name="holistic_null_filling", clean=False)
                class_instance.checkpoint_reached["holistic_null_filling"] = True

        if class_instance.blueprint_step_selection_non_nlp["numeric_binarizer_pca"]:
            class_instance.numeric_binarizer_pca()
            if class_instance.checkpoints["numeric_binarizer_pca"]:
                save_to_production(class_instance=class_instance, file_name="numeric_binarizer_pca", clean=False)
                class_instance.checkpoint_reached["numeric_binarizer_pca"] = True

        if class_instance.blueprint_step_selection_non_nlp["onehot_pca"]:
            class_instance.onehot_pca()
            if class_instance.checkpoints["onehot_pca"]:
                save_to_production(class_instance=class_instance, file_name="onehot_pca", clean=False)
                class_instance.checkpoint_reached["onehot_pca"] = True

        if class_instance.blueprint_step_selection_non_nlp["category_encoding"]:
            class_instance.category_encoding(algorithm='target')
            if class_instance.checkpoints["category_encoding"]:
                save_to_production(class_instance=class_instance, file_name="category_encoding", clean=False)
                class_instance.checkpoint_reached["category_encoding"] = True

        if class_instance.blueprint_step_selection_non_nlp["fill_nulls_static"]:
            class_instance.fill_nulls(how='static') # can only be here when "static"
            if class_instance.checkpoints["fill_nulls_static"]:
                save_to_production(class_instance=class_instance, file_name="fill_nulls_static", clean=False)
                class_instance.checkpoint_reached["fill_nulls_static"] = True

        if class_instance.blueprint_step_selection_non_nlp["outlier_care"]:
            class_instance.outlier_care(method='isolation', how='append')
            if class_instance.checkpoints["outlier_care"]:
                save_to_production(class_instance=class_instance, file_name="outlier_care", clean=False)
                class_instance.checkpoint_reached["outlier_care"] = True

        if class_instance.blueprint_step_selection_non_nlp["remove_collinearity"]:
            class_instance.remove_collinearity(threshold=0.8)
            if class_instance.checkpoints["remove_collinearity"]:
                save_to_production(class_instance=class_instance, file_name="remove_collinearity", clean=False)
                class_instance.checkpoint_reached["remove_collinearity"] = True

        if class_instance.blueprint_step_selection_non_nlp["skewness_removal"]:
            class_instance.skewness_removal(overwrite_orig_col=False)
            if class_instance.checkpoints["skewness_removal"]:
                save_to_production(class_instance=class_instance, file_name="skewness_removal", clean=False)
                class_instance.checkpoint_reached["skewness_removal"] = True

        if class_instance.blueprint_step_selection_non_nlp["clustering_as_a_feature_dbscan"]:
            try:
                class_instance.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
            if class_instance.checkpoints["clustering_as_a_feature_dbscan"]:
                save_to_production(class_instance=class_instance, file_name="clustering_as_a_feature_dbscan", clean=False)
                class_instance.checkpoint_reached["clustering_as_a_feature_dbscan"] = True

        if class_instance.blueprint_step_selection_non_nlp["clustering_as_a_feature_kmeans_loop"]:
            for nb_cluster in [3, 5, 7, 9]:
                try:
                    class_instance.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")
            if class_instance.checkpoints["clustering_as_a_feature_kmeans_loop"]:
                save_to_production(class_instance=class_instance, file_name="clustering_as_a_feature_kmeans_loop", clean=False)
                class_instance.checkpoint_reached["clustering_as_a_feature_kmeans_loop"] = True

        if class_instance.blueprint_step_selection_non_nlp["clustering_as_a_feature_gaussian_mixture_loop"]:
            for nb_cluster in [2, 4, 6, 8, 10]:
                try:
                    class_instance.clustering_as_a_feature(algorithm='gaussian', nb_clusters=nb_cluster)
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")
            if class_instance.checkpoints["clustering_as_a_feature_gaussian_mixture_loop"]:
                save_to_production(class_instance=class_instance, file_name="clustering_as_a_feature_gaussian_mixture_loop", clean=False)
                class_instance.checkpoint_reached["clustering_as_a_feature_gaussian_mixture_loop"] = True

        if class_instance.blueprint_step_selection_non_nlp["pca_clustering_results"]:
            class_instance.pca_clustering_results()
            if class_instance.checkpoints["pca_clustering_results"]:
                save_to_production(class_instance=class_instance, file_name="pca_clustering_results", clean=False)
                class_instance.checkpoint_reached["pca_clustering_results"] = True

        if class_instance.blueprint_step_selection_non_nlp["autotuned_clustering"]:
            class_instance.auto_tuned_clustering()
            if class_instance.checkpoints["autotuned_clustering"]:
                save_to_production(class_instance=class_instance, file_name="autotuned_clustering", clean=False)
                class_instance.checkpoint_reached["autotuned_clustering"] = True

        if class_instance.blueprint_step_selection_non_nlp["reduce_memory_footprint"]:
            if class_instance.low_memory_mode:
                class_instance.reduce_memory_footprint()
                if class_instance.checkpoints["reduce_memory_footprint"]:
                    save_to_production(class_instance=class_instance, file_name="reduce_memory_footprint", clean=False)
                    class_instance.checkpoint_reached["reduce_memory_footprint"] = True

        if class_instance.blueprint_step_selection_non_nlp["scale_data"]:
            class_instance.data_scaling()
            if class_instance.checkpoints["scale_data"]:
                save_to_production(class_instance=class_instance, file_name="scale_data", clean=False)
                class_instance.checkpoint_reached["scale_data"] = True

        if class_instance.blueprint_step_selection_non_nlp["smote"]:
            if class_instance.class_problem == 'binary' or class_instance.class_problem == 'multiclass':
                class_instance.smote_data()
            else:
                pass
            if class_instance.checkpoints["smote"]:
                save_to_production(class_instance=class_instance, file_name="smote", clean=False)
                class_instance.checkpoint_reached["smote"] = True

        if class_instance.blueprint_step_selection_non_nlp["automated_feature_selection"]:
            class_instance.automated_feature_selection(numeric_only=False)
            if class_instance.checkpoints["automated_feature_selection"]:
                save_to_production(class_instance=class_instance, file_name="automated_feature_selection", clean=False)
                class_instance.checkpoint_reached["automated_feature_selection"] = True

        if class_instance.blueprint_step_selection_non_nlp["bruteforce_random_feature_selection"]:
            class_instance.bruteforce_random_feature_selection()
            if class_instance.checkpoints["bruteforce_random_feature_selection"]:
                save_to_production(class_instance=class_instance, file_name="bruteforce_random_feature_selection", clean=False)
                class_instance.checkpoint_reached["bruteforce_random_feature_selection"] = True

        if class_instance.blueprint_step_selection_non_nlp["delete_unpredictable_training_rows"]:
            class_instance.delete_unpredictable_training_rows()
            if class_instance.checkpoints["delete_unpredictable_training_rows"]:
                save_to_production(class_instance=class_instance, file_name="delete_unpredictable_training_rows", clean=False)
                class_instance.checkpoint_reached["delete_unpredictable_training_rows"] = True

        if class_instance.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"]:
            class_instance.autoencoder_based_oversampling()
            if class_instance.checkpoints["autoencoder_based_oversampling"]:
                save_to_production(class_instance=class_instance, file_name="autoencoder_based_oversampling", clean=False)
                class_instance.checkpoint_reached["autoencoder_based_oversampling"] = True

        #self.autoencoder_based_dimensionality_reduction()
        if class_instance.blueprint_step_selection_non_nlp["synthetic_data_augmentation"]:
            class_instance.synthetic_data_augmentation()
            if class_instance.checkpoints["synthetic_data_augmentation"]:
                save_to_production(class_instance=class_instance, file_name="synthetic_data_augmentation", clean=False)
                class_instance.checkpoint_reached["synthetic_data_augmentation"] = True

        if class_instance.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"]:
            class_instance.final_pca_dimensionality_reduction()
            if class_instance.checkpoints["final_pca_dimensionality_reduction"]:
                save_to_production(class_instance=class_instance, file_name="final_pca_dimensionality_reduction", clean=False)
                class_instance.checkpoint_reached["final_pca_dimensionality_reduction"] = True

        if class_instance.blueprint_step_selection_non_nlp["sort_columns_alphabetically"]:
            class_instance.sort_columns_alphabetically()
            if class_instance.checkpoints["sort_columns_alphabetically"]:
                save_to_production(class_instance=class_instance, file_name="sort_columns_alphabetically", clean=False)
                class_instance.checkpoint_reached["sort_columns_alphabetically"] = True
