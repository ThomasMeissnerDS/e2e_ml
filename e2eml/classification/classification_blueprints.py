from e2eml.classification.classification_models import ClassificationModels
from e2eml.full_processing.cpu_processing_nlp import NlpPreprocessing
import logging


class ClassificationBluePrint(ClassificationModels, NlpPreprocessing):
    def full_classification_preprocessing(self, df=None, preprocessing_type='full'):
        """
        Runs a preprocessing blueprint only. This is useful for building custom pipelines.
        :param df: Accepts a dataframe to run ml preprocessing on it.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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

    def train_pred_selected_model(self, algorithm=None, skip_train=False, tune_mode='simple'):
        logging.info(f'Start ML training {algorithm}')
        if algorithm == 'xgboost':
            # train Xgboost
            if skip_train:
                pass
            else:
                self.xg_boost_train(autotune=True, tune_mode=tune_mode)
            self.xgboost_predict(feat_importance=True)
            self.classification_eval(algorithm=algorithm)
        elif algorithm == 'ngboost':
            # train Ngboost
            if skip_train:
                pass
            else:
                self.ngboost_train(tune_mode=tune_mode)
            self.ngboost_predict(feat_importance=True, importance_alg='SHAP')
            self.classification_eval(algorithm=algorithm)
        elif algorithm == 'lgbm':
            # train LGBM
            if skip_train:
                pass
            else:
                try:
                    self.lgbm_train(tune_mode=tune_mode)
                except Exception:
                    self.lgbm_train(tune_mode=tune_mode)
            self.lgbm_predict(feat_importance=True)
            self.classification_eval(algorithm=algorithm)
        elif algorithm == 'sklearn_ensemble':
            # train sklearn ensemble
            if skip_train:
                pass
            else:
                self.sklearn_ensemble_train()
            self.sklearn_ensemble_predict(feat_importance=True, importance_alg='permutation')
            algorithm = 'sklearn_ensemble'
            self.classification_eval(algorithm=algorithm, pred_probs=self.predicted_probs[algorithm][:, 1])

    def ml_bp00_train_test_binary_full_processing_log_reg_prob(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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
        self.skewness_removal()
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='logloss')
        self.sort_columns_alphabetically()
        if skip_train:
            pass
        else:
            self.logistic_regression_train()
        self.data_scaling()
        self.smote_data()
        algorithm = 'logistic_regression'
        self.logistic_regression_predict(feat_importance=True, importance_alg='permutation')
        self.classification_eval(algorithm=algorithm, pred_probs=self.predicted_probs[algorithm][:, 1])
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp01_multiclass_full_processing_xgb_prob(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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
        if skip_train:
            pass
        else:
            self.xg_boost_train(autotune=True, tune_mode='accurate')
        self.xgboost_predict(feat_importance=True)
        self.classification_eval('xgboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp02_multiclass_full_processing_lgbm_prob(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all', force_conversion=False)
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
        if skip_train:
            pass
        else:
            self.lgbm_train(tune_mode='accurate')
        self.lgbm_predict(feat_importance=True)
        self.classification_eval('lgbm')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp03_multiclass_full_processing_sklearn_stacking_ensemble(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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
        self.smote_data()
        if skip_train:
            pass
        else:
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'sklearn_ensemble'
        self.classification_eval(algorithm=algorithm, pred_probs=self.predicted_probs[algorithm][:, 1])
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp04_multiclass_full_processing_ngboost(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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
        if skip_train:
            pass
        else:
            self.ngboost_train(tune_mode='accurate')
        self.ngboost_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'ngboost'
        self.classification_eval(algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_binary_full_processing_boosting_blender(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
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
        if skip_train:
            pass
        else:
            self.ngboost_train(tune_mode='accurate')
            self.lgbm_train(tune_mode='accurate')
            self.xg_boost_train(autotune=True, tune_mode='accurate')
        self.ngboost_predict(feat_importance=False, importance_alg='SHAP')
        self.classification_eval('ngboost')
        self.lgbm_predict(feat_importance=False)
        self.classification_eval('lgbm')
        self.xgboost_predict(feat_importance=True)
        self.classification_eval('xgboost')
        algorithm = 'blended_preds'
        if self.prediction_mode:
            self.dataframe["lgbm_preds"] = self.predicted_probs[f"lgbm"]
            self.dataframe["ngboost_preds"] = self.predicted_probs[f"ngboost"]
            self.dataframe["xgboost_preds"] = self.predicted_probs[f"xgboost"]
            self.dataframe["blended_preds"] = (self.dataframe["lgbm_preds"] + self.dataframe["ngboost_preds"] + self.dataframe["xgboost_preds"])/3
            self.predicted_probs[f"blended_preds"] = self.dataframe["blended_preds"]
            predicted_classes = self.predicted_probs[f"blended_preds"] > self.preprocess_decisions[f"probability_threshold"]
            self.predicted_classes[f"blended_preds"] = predicted_classes
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_test["lgbm_preds"] = self.predicted_probs[f"lgbm"]
            X_test["ngboost_preds"] = self.predicted_probs[f"ngboost"]
            X_test["xgboost_preds"] = self.predicted_probs[f"xgboost"]
            X_test["blended_preds"] = (X_test["lgbm_preds"] + X_test["ngboost_preds"] + X_test["xgboost_preds"])/3
            self.predicted_probs[f"blended_preds"] = X_test["blended_preds"]
            self.threshold_refiner(self.predicted_probs[f"blended_preds"], Y_test)
            predicted_classes = self.predicted_probs[f"blended_preds"] > self.preprocess_decisions[f"probability_threshold"]
            self.predicted_classes[f"blended_preds"] = predicted_classes
        self.classification_eval('blended_preds')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_multiclass_auto_model_exploration(self, df=None, preprocessing_type='full'):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :param preprocessing_type: Select the type of preprocessing pipeline. "Minimum" executes the least possible steps,
        "full" the whole standard preprocessing and "nlp" adds functionality especially for NLP tasks.
        :return: Updates class attributes by its predictions.
        """
        logging.info('Start blueprint.')
        try:
            if df.empty:
                skip_train = False
            else:
                self.dataframe = df
                skip_train = True
        except AttributeError:
            skip_train = False
        self.train_test_split(how=self.train_split_type)
        self.datetime_converter(datetime_handling='all')
        if preprocessing_type == 'nlp':
            self.pos_tagging_pca()
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(how='imputation')
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
        if not self.prediction_mode:
            self.train_pred_selected_model(algorithm='lgbm')
            self.train_pred_selected_model(algorithm='xgboost')
            self.train_pred_selected_model(algorithm='ngboost')
            self.train_pred_selected_model(algorithm='sklearn_ensemble')

            # select best model
            max_matthews = 0
            self.best_model = 'xgboost'
            for k, v in self.evaluation_scores.items():
                if max_matthews < (v['matthews']):
                    max_matthews = (v['matthews'])
                    self.best_model = k
            self.train_pred_selected_model(algorithm=self.best_model)
            self.prediction_mode = True
        else:
            self.train_pred_selected_model(algorithm=self.best_model, skip_train=skip_train)
        logging.info('Finished blueprint.')
