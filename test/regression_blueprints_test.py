import regression.regression_blueprints as rb
import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_housingprices_data():
    """
    Load & preprocess Housing prices dataset. T
    :return: Several dataframes and series to be processed by blueprint.
    """
    data = pd.read_csv("housingprices_train.csv")
    print('Do dataframe splits.')
    test_df = data.head(1000).copy()
    val_df = data.tail(460).copy()
    val_df_target = val_df["SalePrice"].copy()
    del val_df["SalePrice"]
    test_target = "SalePrice"
    test_categorical_cols = ["MSZoning", "Street", "Alley", "LotShape", "LotFrontage", "Street", "LandContour",
                             "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                             "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                             "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
                             "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "Electrical", "KitchenQual",
                             "Functional", "FireplaceQU", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual",
                             "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
    return test_df, test_target, val_df, val_df_target, test_categorical_cols


def blueprint_regression_test_housingprices(blueprint='lgbm'):
    test_df, test_target, val_df, val_df_target, test_categorical_cols = load_housingprices_data()
    titanic_auto_ml = rb.BluePrint(dataframe=test_df,
                                   target_variable=test_target,
                                   categorical_columns=test_categorical_cols)
    if blueprint == 'lgbm':
        titanic_auto_ml.ml_bp12_regressions_full_processing_lgbm_prob()
        print("Start prediction on holdout dataset")
        titanic_auto_ml.ml_bp12_regressions_full_processing_lgbm_prob(val_df)
        val_y_hat = titanic_auto_ml.predicted_values['lgbm']

        mae = mean_absolute_error(val_df_target, val_y_hat)
        print(mae)
    elif blueprint == 'xgboost':
        titanic_auto_ml.ml_bp11_regression_full_processing_xgb_prob()
        print("Start prediction on holdout dataset")
        titanic_auto_ml.ml_bp11_regression_full_processing_xgb_prob(val_df)
        val_y_hat = titanic_auto_ml.predicted_values['xgboost']

        mae = mean_absolute_error(val_df_target, val_y_hat)
        print(mae)
    else:
        pass


blueprint_regression_test_housingprices(blueprint='xgboost')