from regression.regression_models import RegressionModels
import numpy as np


class BluePrint(RegressionModels):
    def ml_bp11_regression_full_processing_xgb_prob(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :return: Updates class attributes by its predictions.
        """
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
        self.cardinality_remover(threshold=1000)
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=50)
        self.automated_feature_selection(metric='mae')
        if skip_train:
            pass
        else:
            self.xg_boost_train(autotune=True)
        self.xgboost_predict(feat_importance=True)
        self.classification_eval('xgboost')
        self.prediction_mode = True

    def ml_bp12_regressions_full_processing_lgbm_prob(self, df=None):
        """
        Runs a blue print from preprocessing to model training. Can be used as a pipeline to predict on new data,
        if the predict_mode attribute is True.
        :param df: Accepts a dataframe to make predictions on new data.
        :return: Updates class attributes by its predictions.
        """
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
        self.cardinality_remover(threshold=1000)
        self.category_encoding(algorithm='target')
        self.delete_high_null_cols(threshold=0.5)
        self.fill_nulls(inplace=False, how='static')
        self.data_binning(nb_bins=10)
        self.outlier_care(method='isolation', how='append')
        self.remove_collinearity(threshold=0.8)
        self.clustering_as_a_feature(algorithm='dbscan', eps=0.3, n_jobs=-1, min_samples=50)
        self.automated_feature_selection(metric='mae')
        if skip_train:
            pass
        else:
            try:
                self.lgbm_train(tune_mode='simple', run_on='gpu', gpu_use_dp=True)
            except Exception:
                self.lgbm_train(tune_mode='simple', run_on='cpu', gpu_use_dp=False)
        self.lgbm_predict(feat_importance=True)
        self.prediction_mode = True