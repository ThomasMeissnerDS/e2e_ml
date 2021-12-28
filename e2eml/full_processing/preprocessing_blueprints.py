import logging

from e2eml.full_processing.cpu_processing_nlp import NlpPreprocessing
from e2eml.full_processing.postprocessing import FullPipeline


class PreprocessingBluePrint(FullPipeline, NlpPreprocessing):
    def check_prediction_mode(self, df):
        """
        Takes in the dataframe that has been passed to the blueprint pipeline. If no dataframe has been passed,
        sets prediction mode to False. Otherwise sets prediction mode to True. This is a fallback function. In case
        someone has built a custom pipeline it makes sure to fall back into prediction mode once used for that.
        :param df: Pandas Dataframe
        :return: Updates class attributes
        """
        try:
            if df.empty:
                self.prediction_mode = False
            else:
                self.dataframe = df
                self.prediction_mode = True
        except AttributeError:
            self.prediction_mode = False

    def dbscan_clustering(self):
        if self.blueprint_step_selection_non_nlp["clustering_as_a_feature_dbscan"]:
            try:
                self.clustering_as_a_feature(
                    algorithm="dbscan", eps=0.3, n_jobs=-1, min_samples=10
                )
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")

    def kmeans_clustering_loop(self):
        if self.blueprint_step_selection_non_nlp["clustering_as_a_feature_kmeans_loop"]:
            for nb_cluster in [3, 5, 7, 9]:
                try:
                    self.clustering_as_a_feature(
                        algorithm="kmeans",
                        eps=0.3,
                        n_jobs=-1,
                        nb_clusters=nb_cluster,
                        min_samples=10,
                    )
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")

    def gaussian_mixture_clustering_loop(self):
        if self.blueprint_step_selection_non_nlp[
            "clustering_as_a_feature_gaussian_mixture_loop"
        ]:
            for nb_cluster in [2, 4, 6, 8, 10]:
                try:
                    self.clustering_as_a_feature(
                        algorithm="gaussian",
                        eps=0.3,
                        n_jobs=-1,
                        nb_clusters=nb_cluster,
                        min_samples=10,
                    )
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")

    def svm_outlier_detection_loop(self):
        if self.blueprint_step_selection_non_nlp["svm_outlier_detection_loop"]:
            for nu in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
                try:
                    self.svm_outlier_detection(nu=nu)
                except Exception:
                    print("SVM outlier detection skipped due to error..")

    def smote_binary_multiclass(self):
        if self.class_problem == "binary" or self.class_problem == "multiclass":
            self.smote_data()
        else:
            pass

    def std_preprocessing_pipeline(self, df=None):  # noqa: C901
        """
        Our recommended blueprint for Tabnet testing.
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info("Start blueprint.")
        self.runtime_warnings(warn_about="future_architecture_change")
        self.check_prediction_mode(df)

        self.train_test_split(how=self.train_split_type)
        self.binary_imbalance()

        if self.blueprint_step_selection_non_nlp["automatic_type_detection_casting"]:
            self.automatic_type_detection_casting()
        if self.blueprint_step_selection_non_nlp["remove_duplicate_column_names"]:
            self.remove_duplicate_column_names()
        if self.blueprint_step_selection_non_nlp["reset_dataframe_index"]:
            self.reset_dataframe_index()
        if self.blueprint_step_selection_non_nlp["fill_infinite_values"]:
            self.fill_infinite_values()
        if self.blueprint_step_selection_non_nlp[
            "early_numeric_only_feature_selection"
        ]:
            self.automated_feature_selection(numeric_only=True)
        if self.blueprint_step_selection_non_nlp["delete_high_null_cols"]:
            self.delete_high_null_cols(threshold=0.05)
        if self.blueprint_step_selection_non_nlp["data_binning"]:
            self.data_binning(nb_bins=10)
        if (
            self.blueprint_step_selection_non_nlp["regex_clean_text_data"]
            and len(self.nlp_transformer_columns) > 0
        ):
            self.regex_clean_text_data()
        if self.blueprint_step_selection_non_nlp["handle_target_skewness"]:
            self.target_skewness_handling(mode="fit")
        # self.fill_nulls(how='static') # can only be here when "static"
        if self.blueprint_step_selection_non_nlp["datetime_converter"]:
            self.datetime_converter(datetime_handling="all")
        if self.blueprint_step_selection_non_nlp["pos_tagging_pca"]:
            self.pos_tagging_pca(pca_pos_tags=True)
        if (
            self.blueprint_step_selection_non_nlp["append_text_sentiment_score"]
            and len(self.nlp_transformer_columns) > 0
        ):
            self.append_text_sentiment_score()
        if self.blueprint_step_selection_non_nlp["tfidf_vectorizer_to_pca"]:
            self.tfidf_vectorizer_to_pca(pca_pos_tags=True)
        if (
            self.blueprint_step_selection_non_nlp["tfidf_vectorizer"]
            and len(self.nlp_transformer_columns) > 0
        ):
            self.tfidf_vectorizer_to_pca(pca_pos_tags=False)
        if self.blueprint_step_selection_non_nlp["rare_feature_processing"]:
            self.rare_feature_processor(
                threshold=0.005, mask_as="miscellaneous", rarity_cols=self.rarity_cols
            )
        if self.blueprint_step_selection_non_nlp["cardinality_remover"]:
            self.cardinality_remover(threshold=100)
        if self.blueprint_step_selection_non_nlp["categorical_column_embeddings"]:
            self.categorical_column_embeddings()
        if self.blueprint_step_selection_non_nlp["holistic_null_filling"]:
            self.holistic_null_filling(iterative=False)
        if self.blueprint_step_selection_non_nlp["numeric_binarizer_pca"]:
            self.numeric_binarizer_pca()
        if self.blueprint_step_selection_non_nlp["onehot_pca"]:
            self.onehot_pca()
        if self.blueprint_step_selection_non_nlp["category_encoding"]:
            self.category_encoding(algorithm="target")
        if self.blueprint_step_selection_non_nlp["fill_nulls_static"]:
            self.fill_nulls(how="static")  # can only be here when "static"
        if self.blueprint_step_selection_non_nlp["outlier_care"]:
            self.outlier_care(method="isolation", how="append")
        if self.blueprint_step_selection_non_nlp["delete_outliers"]:
            self.outlier_care(method="isolation", how="delete", threshold=-0.5)
        if self.blueprint_step_selection_non_nlp["remove_collinearity"]:
            self.remove_collinearity(threshold=0.8)
        if self.blueprint_step_selection_non_nlp["skewness_removal"]:
            self.skewness_removal(overwrite_orig_col=False)
        if self.blueprint_step_selection_non_nlp["automated_feature_transformation"]:
            self.automated_feature_transformation()
        if self.blueprint_step_selection_non_nlp["clustering_as_a_feature_dbscan"]:
            try:
                self.clustering_as_a_feature(
                    algorithm="dbscan", eps=0.3, n_jobs=-1, min_samples=10
                )
            except ValueError:
                print("Clustering as a feature skipped due to ValueError.")
        if self.blueprint_step_selection_non_nlp["clustering_as_a_feature_kmeans_loop"]:
            for nb_cluster in [3, 5, 7, 9]:
                try:
                    self.clustering_as_a_feature(
                        algorithm="kmeans",
                        nb_clusters=nb_cluster,
                        eps=None,
                        n_jobs=-1,
                        min_samples=50,
                    )
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")
        if self.blueprint_step_selection_non_nlp[
            "clustering_as_a_feature_gaussian_mixture_loop"
        ]:
            for nb_cluster in [2, 4, 6, 8, 10]:
                try:
                    self.clustering_as_a_feature(
                        algorithm="gaussian", nb_clusters=nb_cluster
                    )
                except ValueError:
                    print("Clustering as a feature skipped due to ValueError.")
        if self.blueprint_step_selection_non_nlp["pca_clustering_results"]:
            self.pca_clustering_results()
        if self.blueprint_step_selection_non_nlp["autotuned_clustering"]:
            self.auto_tuned_clustering()
        if self.blueprint_step_selection_non_nlp["svm_outlier_detection_loop"]:
            for nu in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
                try:
                    self.svm_outlier_detection(nu=nu)
                except Exception:
                    print("SVM outlier detection skipped due to error.")
        if self.blueprint_step_selection_non_nlp["reduce_memory_footprint"]:
            if self.low_memory_mode:
                self.reduce_memory_footprint()
        if self.blueprint_step_selection_non_nlp["scale_data"]:
            self.data_scaling()
        if self.blueprint_step_selection_non_nlp["smote"]:
            if self.class_problem == "binary" or self.class_problem == "multiclass":
                self.smote_data()
            else:
                pass
        if self.blueprint_step_selection_non_nlp["automated_feature_selection"]:
            self.automated_feature_selection(numeric_only=False)
        if self.blueprint_step_selection_non_nlp["bruteforce_random_feature_selection"]:
            self.bruteforce_random_feature_selection()
        if self.blueprint_step_selection_non_nlp[
            "final_kernel_pca_dimensionality_reduction"
        ]:
            self.final_kernel_pca_dimensionality_reduction()
        if self.blueprint_step_selection_non_nlp["final_pca_dimensionality_reduction"]:
            self.final_pca_dimensionality_reduction()
        if self.blueprint_step_selection_non_nlp["delete_low_variance_features"]:
            self.delete_low_variance_features()
        if self.blueprint_step_selection_non_nlp["shap_based_feature_selection"]:
            self.shap_based_feature_selection()
        if self.blueprint_step_selection_non_nlp["autoencoder_based_oversampling"]:
            self.autoencoder_based_oversampling()
        if self.blueprint_step_selection_non_nlp["synthetic_data_augmentation"]:
            self.synthetic_data_augmentation()
        if self.blueprint_step_selection_non_nlp["delete_unpredictable_training_rows"]:
            self.delete_unpredictable_training_rows()
        if self.blueprint_step_selection_non_nlp["random_trees_embedding"]:
            self.random_trees_embedding()
        if self.blueprint_step_selection_non_nlp["sort_columns_alphabetically"]:
            self.sort_columns_alphabetically()

    def nlp_transformer_preprocessing_pipeline(self, df):
        logging.info("Start blueprint.")
        self.runtime_warnings(warn_about="future_architecture_change")
        self.check_prediction_mode(df)

        if self.blueprint_step_selection_nlp_transformers["train_test_split"]:
            self.train_test_split(how=self.train_split_type)
        if (
            self.blueprint_step_selection_nlp_transformers["regex_clean_text_data"]
            and len(self.nlp_transformer_columns) > 0
        ):
            self.regex_clean_text_data()
        if self.blueprint_step_selection_nlp_transformers["random_synonym_replacement"]:
            self.replace_synonyms_to_df_copy(words_to_replace=3, mode="auto")
        if self.blueprint_step_selection_nlp_transformers["oversampling"]:
            self.oversample_train_data()
        if self.blueprint_step_selection_nlp_transformers["rare_feature_processing"]:
            self.rare_feature_processor(
                threshold=0.005, mask_as="miscellaneous", rarity_cols=self.rarity_cols
            )
        if self.blueprint_step_selection_nlp_transformers[
            "sort_columns_alphabetically"
        ]:
            self.sort_columns_alphabetically()
        self.check_max_sentence_length()
        self.import_transformer_model_tokenizer(
            transformer_chosen=self.transformer_chosen
        )
