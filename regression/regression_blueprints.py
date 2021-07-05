from regression.regression_models import RegressionModels
import numpy as np
import logging


class RegressionBluePrint(RegressionModels):
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
                    self.lgbm_train(tune_mode=tune_mode, run_on='gpu', gpu_use_dp=True)
                except Exception:
                    self.lgbm_train(tune_mode=tune_mode, run_on='cpu', gpu_use_dp=False)
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

    def ml_bp10_train_test_regression_full_processing_linear_reg(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='mae')
        self.sort_columns_alphabetically()
        if skip_train:
            pass
        else:
            self.linear_regression_train()
        self.data_scaling()
        algorithm = 'linear_regression'
        self.linear_regression_predict(feat_importance=True, importance_alg='permutation')
        self.regression_eval(algorithm=algorithm)
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp11_regression_full_processing_xgboost(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='mae')
        self.sort_columns_alphabetically()
        if skip_train:
            pass
        else:
            self.xg_boost_train(autotune=True)
        self.xgboost_predict(feat_importance=True)
        self.regression_eval('xgboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp12_regressions_full_processing_lgbm(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='mae')
        self.sort_columns_alphabetically()
        if skip_train:
            pass
        else:
            try:
                self.lgbm_train(tune_mode='simple', run_on='gpu', gpu_use_dp=True)
            except Exception:
                self.lgbm_train(tune_mode='simple', run_on='cpu', gpu_use_dp=False)
        self.lgbm_predict(feat_importance=True)
        self.regression_eval('lgbm')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp13_regression_full_processing_sklearn_stacking_ensemble(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='mae')
        self.sort_columns_alphabetically()
        self.data_scaling()
        if skip_train:
            pass
        else:
            self.sklearn_ensemble_train()
        self.sklearn_ensemble_predict(feat_importance=True, importance_alg='permutation')
        algorithm = 'sklearn_ensemble'
        self.regression_eval('sklearn_ensemble')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_bp14_regressions_full_processing_ngboost(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.train_test_split()
        self.datetime_converter(datetime_handling='all')
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=100)
        self.onehot_pca()
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=10)
        for nb_cluster in range(2, 10):
            self.clustering_as_a_feature(algorithm='kmeans', nb_clusters=nb_cluster)
        if self.low_memory_mode:
            self.reduce_memory_footprint()
        self.automated_feature_selection(metric='mae')
        self.sort_columns_alphabetically()
        if skip_train:
            pass
        else:
            self.ngboost_train()
        self.ngboost_predict(feat_importance=True, importance_alg='permutation')
        self.regression_eval('ngboost')
        self.prediction_mode = True
        logging.info('Finished blueprint.')

    def ml_special_regression_auto_model_exploration(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
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
        self.rare_feature_processor(threshold=0.03, mask_as='miscellaneous')
        self.cardinality_remover(threshold=1000)
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
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