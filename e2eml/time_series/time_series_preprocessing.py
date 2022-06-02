import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from e2eml.full_processing import cpu_preprocessing

plt.style.use("fivethirtyeight")


class TimeSeriesPreprocessing(cpu_preprocessing.PreProcessing):
    def time_series_unpacking(self):
        """
        Takes the datasource dictionary within the class and unpacks it, depending on the provided source format.
        :return: Returns unpacked data objects.
        """
        if self.source_format == "numpy array":
            Y_train, Y_test = self.np_array_unpack_test_train_dict()
            return Y_train, Y_test
        elif self.source_format == "Pandas dataframe":
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            return X_train, X_test, Y_train, Y_test

    def keep_target_only(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train = X_train[[self.target_variable]]
            X_test = X_test[[self.target_variable]]
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def reattach_targets(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train[self.target_variable] = Y_train
            X_test[self.target_variable] = Y_test
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def sliding_windows(self, data, seq_length=5):
        x = []
        y = []

        for i in range(len(data) - seq_length - 1):
            _x = data[i : (i + seq_length)]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def create_sliding_windows(self):
        x, y = self.sliding_windows(self.dataframe.to_numpy())
        train_size = int(len(y) * 0.30)

        X_train = np.array(x[0:train_size])
        Y_train = np.array(y[0:train_size])

        X_test = np.array(x[train_size : len(x)])
        Y_test = np.array(y[train_size : len(y)])
        self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def time_series_wrap_test_train_to_dict(
        self, X_train=None, X_test=None, Y_train=None, Y_test=None
    ):
        """
        Takes either four Pandas dataframes and series or two numpy arrays and packs them into the class
        dictionary.
        :param X_train: Pandas dataframe (optional)
        :param X_test: Pandas dataframe (optional)
        :param Y_train: Pandas series or numpy array
        :param Y_test: Pandas series or numpy array
        :return: Updates class and return dictionary.
        """
        if self.source_format == "numpy array":
            return self.np_array_wrap_test_train_to_dict(Y_train, Y_test)
        elif self.source_format == "Pandas dataframe":
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def augmented_dickey_fuller_test(self, window):
        """
        The Dickey Fuller test is one of the most popular statistical tests. It can be used to determine the
        presence of unit root in the series, and hence help us understand if the series is stationary or not.
        The null and alternate hypothesis of this test are:
          - Null Hypothesis: The series has a unit root (value of a =1)
          - Alternate Hypothesis: The series has no unit root.
        If we fail to reject the null hypothesis, we can say that the series is non-stationary.
        This means that the series can be linear or difference stationary.
        :return:
        """
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        if self.auto_arima_settings["plot_adf_kpss"]:
            # Determing rolling statistics
            rolmean = X_train.rolling(window=window).mean()
            rolstd = X_train.rolling(window=window).std()
            # Plot rolling statistics:
            plt.plot(X_train, color="blue", label="Original")
            plt.plot(rolmean, color="red", label="Rolling Mean")
            plt.plot(rolstd, color="black", label="Rolling Std")
            plt.legend(loc="best")
            plt.title("Rolling Mean & Standard Deviation")
            plt.show()
            # Perform Dickey-Fuller test:
            print("Results of Dickey-Fuller Test:")
        dftest = adfuller(X_train, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        if self.auto_arima_settings["plot_adf_kpss"]:
            print(dfoutput)
        if dfoutput["p-value"] >= 0.5:
            self.preprocess_decisions["adf_test_is_stationary"] = False
        else:
            self.preprocess_decisions["adf_test_is_stationary"] = True

    def kwiatkowski_phillips_schmidt_shin_test(self):
        """
        KPSS is another test for checking the stationarity of a time series (slightly less popular than the Dickey
        Fuller test). The null and alternate hypothesis for the KPSS test are opposite that of the ADF test,
        which often creates confusion.
        The authors of the KPSS test have defined the null hypothesis as the process is trend stationary,
        to an alternate hypothesis of a unit root series.
        - Null Hypothesis: The process is trend stationary.
        - Alternate Hypothesis: The series has a unit root (series is not stationary).
        Test for stationarity: If the test statistic is greater than the critical value,
        we reject the null hypothesis (series is not stationary). If the test statistic is less than the critical value,
         if fail to reject the null hypothesis (series is stationary).
        So in summary, the ADF test has an alternate hypothesis of linear or difference stationary,
        while the KPSS test identifies trend-stationarity in a series.
        :return:
        """
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        try:
            kpsstest = kpss(X_train.values, regression="c")
            kpss_output = pd.Series(
                kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
            )
            for key, value in kpsstest[3].items():
                kpss_output["Critical Value (%s)" % key] = value

            if self.auto_arima_settings["plot_adf_kpss"]:
                print("Results of KPSS Test:")
                print(kpss_output)
            if kpss_output["p-value"] < 0.5:
                self.preprocess_decisions["kpss_test_is_stationary"] = False
            else:
                self.preprocess_decisions["kpss_test_is_stationary"] = True
        except ValueError:
            kpss_output = {}
            kpss_output["p-value"] = "nan"
            if self.auto_arima_settings["plot_adf_kpss"]:
                print("KPSS test failed. Use ADF results only.")
            self.preprocess_decisions[
                "kpss_test_is_stationary"
            ] = self.preprocess_decisions["adf_test_is_stationary"]

    def make_stationary(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions["arima_nb_differentiations"] = 0
            self.preprocess_decisions["arima_transformation_used"] = "None"
            X_train_copy = X_train.copy()

            self.augmented_dickey_fuller_test(window=6)
            self.kwiatkowski_phillips_schmidt_shin_test()

            while (
                not self.preprocess_decisions["adf_test_is_stationary"]
                and not self.preprocess_decisions["kpss_test_is_stationary"]
            ):
                self.preprocess_decisions["arima_nb_differentiations"] += 1
                X_train = X_train.diff(
                    self.preprocess_decisions["arima_nb_differentiations"]
                ).dropna()
                self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

                if (
                    self.preprocess_decisions["arima_nb_differentiations"]
                    > self.auto_arima_settings["max_d"]
                ):
                    X_train = X_train_copy
                    X_train = np.log1p(X_train)
                    X_train[np.isfinite(X_train) is False] = 0
                    Y_train = X_train
                    self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
                    self.preprocess_decisions["arima_transformation_used"] = "log"

                self.augmented_dickey_fuller_test(window=6)
                self.kwiatkowski_phillips_schmidt_shin_test()

            Y_train = X_train

            if self.preprocess_decisions["arima_transformation_used"] == "log":
                Y_test = np.log1p(Y_test)
                Y_test[np.isfinite(Y_test) is False] = 0
            elif self.preprocess_decisions["arima_nb_differentiations"] > 0:
                length = len(Y_test.index)
                full_test = pd.concat([Y_train, Y_test])
                full_test = full_test.diff(
                    self.preprocess_decisions["arima_nb_differentiations"]
                )  # .dropna()
                full_test = full_test.ffill()
                Y_test = full_test.tail(length)
                Y_test = pd.Series(Y_test[self.target_variable].values)
            else:
                pass

            Y_train = pd.Series(Y_train[self.target_variable].values)
            self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def stationarity_explainer(self):
        explanation = """
        What are the different types of stationarity?
        - Strict Stationary: A strict stationary series satisfies the mathematical definition of a stationary process.
          For a strict stationary series, the mean, variance and covariance are not the function of time.
          The aim is to convert a non-stationary series into a strict stationary series for making predictions.
        - Trend Stationary: A series that has no unit root but exhibits a trend is referred to as a trend stationary
          series. Once the trend is removed, the resulting series will be strict stationary. The KPSS test classifies a
          series as stationary on the absence of unit root. This means that the series can be strict stationary or
          trend stationary.
        - Difference Stationary: A time series that can be made strict stationary by differencing falls under difference
          stationary. ADF test is also known as a difference stationarity test.

        It’s always better to apply both the tests, so that we are sure that the series is truly stationary.
        Let us look at the possible outcomes of applying these stationary tests.

        - Case 1: Both tests conclude that the series is not stationary -> series is not stationary
        - Case 2: Both tests conclude that the series is stationary -> series is stationary
        - Case 3: KPSS = stationary and ADF = not stationary -> trend stationary, remove the trend to make series
          strict stationary
        - Case 4: KPSS = not stationary and ADF = stationary -> difference stationary, use differencing to make series
          stationary
        """
        print(explanation)
