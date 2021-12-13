import re

import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.utils import shuffle

from e2eml.classification import classification_blueprints as cb
from e2eml.timetravel import timetravel

# import pytest


def load_titanic_data():
    """
    Load & preprocess Titanic dataset. The feature engineering simulates the business knowledge part.
    The code has been taken from:
    https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
    :return: Several dataframes and series to be processed by blueprint.
    """
    data = pd.read_csv("titanic_train.csv")
    print("Create additional features and modify existing ones.")
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "": 6, "G": 7, "U": 8}
    data["Cabin"] = data["Cabin"].fillna("U0")
    data["Deck"] = data["Cabin"].map(
        lambda x: re.compile("([a-zA-Z]+)").search(x).group()
    )
    data["Deck"] = data["Deck"].map(deck)
    data["Deck"] = data["Deck"].fillna(0)
    data["Deck"] = data["Deck"].astype(int)

    data.loc[data["Fare"] <= 7.91, "Fare"] = 0
    data.loc[(data["Fare"] > 7.91) & (data["Fare"] <= 14.454), "Fare"] = 1
    data.loc[(data["Fare"] > 14.454) & (data["Fare"] <= 31), "Fare"] = 2
    data.loc[(data["Fare"] > 31) & (data["Fare"] <= 99), "Fare"] = 3
    data.loc[(data["Fare"] > 99) & (data["Fare"] <= 250), "Fare"] = 4
    data.loc[data["Fare"] > 250, "Fare"] = 5

    data["Age"].fillna(0, inplace=True)
    data["Age"] = data["Age"].astype(int)
    data.loc[data["Age"] <= 11, "Age"] = 0
    data.loc[(data["Age"] > 11) & (data["Age"] <= 18), "Age"] = 1
    data.loc[(data["Age"] > 18) & (data["Age"] <= 22), "Age"] = 2
    data.loc[(data["Age"] > 22) & (data["Age"] <= 27), "Age"] = 3
    data.loc[(data["Age"] > 27) & (data["Age"] <= 33), "Age"] = 4
    data.loc[(data["Age"] > 33) & (data["Age"] <= 40), "Age"] = 5
    data.loc[(data["Age"] > 40) & (data["Age"] <= 66), "Age"] = 6
    data.loc[data["Age"] > 66, "Age"] = 6

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    # extract titles
    data["Title"] = data.Name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    # replace titles with a more common title or as Rare
    data["Title"] = data["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    data["Title"] = data["Title"].replace("Mlle", "Miss")
    data["Title"] = data["Title"].replace("Ms", "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")
    # convert titles into numbers
    data["Title"] = data["Title"].map(titles)
    # filling NaN with 0, to get safe
    data["Title"] = data["Title"].fillna(0)

    data["relatives"] = data["SibSp"] + data["Parch"]
    data.loc[data["relatives"] > 0, "not_alone"] = 0
    data.loc[data["relatives"] == 0, "not_alone"] = 1
    data["not_alone"] = data["not_alone"].astype(int)

    data["Fare"] = data["Fare"].astype(int)
    data["Age_Class"] = data["Age"] * data["Pclass"]
    data["Fare_Per_Person"] = data["Fare"] / (data["relatives"] + 1)
    data["Fare_Per_Person"] = data["Fare_Per_Person"].astype(int)
    print("Do dataframe splits.")
    test_df = data.head(800).copy()
    val_df = data.tail(91).copy()
    val_df_target = val_df["Survived"].copy()
    del val_df["Survived"]
    test_target = "Survived"
    test_categorical_cols = ["Pclass", "Name", "Sex", "PassengerId"]
    return test_df, test_target, val_df, val_df_target, test_categorical_cols


def steel_fault_multiclass_data():
    data = pd.read_csv("faults.csv")
    data = shuffle(data)
    # data["leakage"] = data["target"]
    print(data.info())
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # X = data.drop('target', axis=1)
    # Y = le.fit_transform(data['target'])

    # X['target'] = Y
    test_df = data.head(1500).copy()
    val_df = data.tail(441).copy()
    val_df_target = val_df["target"].copy()
    del val_df["target"]
    test_target = "target"
    test_categorical_cols = None
    return test_df, test_target, val_df, val_df_target, test_categorical_cols


def nlp_multiclass_data():
    data = pd.read_csv("Corona_NLP_train.csv", encoding="latin-1")
    test_df = data.head(100).copy()
    print(test_df[["OriginalTweet"]])
    val_df = data.tail(499).copy()
    val_df_target = val_df["Sentiment"].copy()
    del val_df["Sentiment"]
    test_target = "Sentiment"
    test_categorical_cols = ["Location", "OriginalTweet"]
    return test_df, test_target, val_df, val_df_target, test_categorical_cols


def test_timetravel(dataset="titanic"):
    if dataset == "titanic":
        (
            test_df,
            test_target,
            val_df,
            val_df_target,
            test_categorical_cols,
        ) = load_titanic_data()
        titanic_auto_ml = cb.ClassificationBluePrint(
            datasource=test_df,
            target_variable=test_target,
            categorical_columns=test_categorical_cols,
            preferred_training_mode="auto",
            tune_mode="accurate",
            rapids_acceleration=True,
        )

    elif dataset == "synthetic_multiclass":
        (
            test_df,
            test_target,
            val_df,
            val_df_target,
            test_categorical_cols,
        ) = steel_fault_multiclass_data()
        titanic_auto_ml = cb.ClassificationBluePrint(
            datasource=test_df,
            target_variable=test_target,
            categorical_columns=test_categorical_cols,
        )
    elif dataset == "corona_tweet":
        (
            test_df,
            test_target,
            val_df,
            val_df_target,
            test_categorical_cols,
        ) = nlp_multiclass_data()
        titanic_auto_ml = cb.ClassificationBluePrint(
            datasource=test_df,
            target_variable=test_target,
            categorical_columns=test_categorical_cols,
            preferred_training_mode="auto",
            tune_mode="accurate",
            nlp_transformer_columns="OriginalTweet",
        )

    titanic_auto_ml.blueprint_step_selection_non_nlp[
        "autoencoder_based_oversampling"
    ] = True
    titanic_auto_ml.blueprint_step_selection_non_nlp[
        "final_pca_dimensionality_reduction"
    ] = False
    titanic_auto_ml.blueprint_step_selection_non_nlp["scale_data"] = False

    # creating checkpoints and training the model
    automl_travel = timetravel.TimeTravel()
    automl_travel.create_time_travel_checkpoints(titanic_auto_ml)
    automl_travel.timetravel_model_training(titanic_auto_ml, "xgboost")

    # loading checkpoint and adjusting the preprocessing
    titanic_auto_ml = automl_travel.load_checkpoint(
        checkpoint_to_load="automated_feature_selection"
    )
    titanic_auto_ml.blueprint_step_selection_non_nlp[
        "final_pca_dimensionality_reduction"
    ] = True
    automl_travel.create_time_travel_checkpoints(titanic_auto_ml, reload_instance=True)
    automl_travel.timetravel_model_training(titanic_auto_ml, "xgboost")

    # predicting on new data
    automl_travel.create_time_travel_checkpoints(titanic_auto_ml, df=val_df)
    automl_travel.timetravel_model_training(titanic_auto_ml, "xgboost")

    def get_matthews(algorithm):
        # Assess prediction quality on holdout data
        print(
            classification_report(
                pd.Series(val_df_target).astype(bool),
                titanic_auto_ml.predicted_classes[algorithm],
            )
        )
        try:
            matthews = matthews_corrcoef(
                pd.Series(val_df_target).astype(bool),
                titanic_auto_ml.predicted_classes[algorithm],
            )
        except Exception:
            print("Matthew failed.")
            matthews = 0
        print(matthews)

    for i in ["lgbm"]:
        print(f"---------Start evaluating {i}----------")
        get_matthews(i)
    finished = True
    assert finished is True


if __name__ == "__main__":
    test_timetravel("titanic")
