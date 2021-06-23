from full_processing import cpu_preprocessing
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
import warnings


class FullPipeline(cpu_preprocessing.PreProcessing):
    def shap_explanations(self, model, test_df, cols, explainer='tree', algorithm=None):
        """
        See explanations under:
        https://medium.com/rapids-ai/gpu-accelerated-shap-values-with-xgboost-1-3-and-rapids-587fad6822
        :param model: Trained ML model
        :param test_df: Test data to predict on.
        :param explainer: Set "tree" for TreeExplainer. Otherwise uses KernelExplainer.
        :param algorithm: Define name of the chosen ml algorithm as a string.
        :return: Returns plot of feature importance and interactions.
        """
        # print the JS visualization code to the notebook
        shap.initjs()
        if self.prediction_mode:
            to_pred = test_df
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            to_pred = X_test
        if explainer == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(to_pred)
            shap.summary_plot(shap_values, to_pred, plot_type="bar", show=False)
            plt.savefig(f'{algorithm}Shap_feature_importance.png')
            plt.show()
        else:
            model_shap_explainer = shap.KernelExplainer(model.predict, to_pred)
            model_shap_values = model_shap_explainer.shap_values(to_pred)
            shap.summary_plot(model_shap_values, to_pred, show=False)
            plt.savefig(f'{algorithm}Shap_feature_importance.png')
            plt.show()

    def runtime_warnings(self, warn_about='shap_cpu'):
        """
        This function returns custom warnings for a better user experience.
        :return: warning message
        """
        if warn_about == 'shap_cpu':
            warning_message = """Calculating SHAP values for feature importance on CPU might run a long time. To disable
            the calculation set the parameter 'feat_importance' to False. Alternatively the LGBM and Xgboost
            blueprints can be used as well. These run on GPU by default and usually yield better
            classification results as well."""
            return warnings.warn(warning_message, RuntimeWarning)
        else:
            pass

    def classification_eval(self, algorithm, pred_probs=None, pred_class=None):
        """
        Takes in the algorithm name. This is needed to grab saved predictions and to store cvlassification scores
        of different evaluation functions within the class. Returns the evaluation dictionary.
        :param algorithm: Name of the used algorithm
        :param pred_probs: Probabilities of predictions
        :param pred_class: Predicted classes
        :return: Returns the evaluation dictionary.
        """
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            """
            We need a fallback logic as we might receive different types of data.
            If pred_class will not work with numpy arrays and needs the .any() function.
            """
            try:
                if pred_class:
                    y_hat = pred_class
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]
            except Exception:
                if pred_class.any():
                    y_hat = pred_class
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]

            try:
                if pred_probs:
                    y_hat_probs = pred_probs
                else:
                    y_hat_probs = self.predicted_probs[f"{algorithm}"]
            except Exception:
                if pred_probs.any():
                    y_hat_probs = pred_probs
                else:
                    y_hat_probs = self.predicted_probs[f"{algorithm}"]

            """
            Calculating Matthews, ROC_AUC score and different F1 scores.
            """
            try:
                matthews = matthews_corrcoef(Y_test, y_hat)
            except Exception:
                matthews = 0
            print(f"The Matthew correlation is {matthews}")

            roc_auc = roc_auc_score(Y_test, y_hat_probs)
            print(f"The ROC_AUC score is {roc_auc}")
            f1_score_macro = f1_score(Y_test, y_hat, average='macro')
            print(f"The macro F1 score is {f1_score_macro}")
            f1_score_micro = f1_score(Y_test, y_hat, average='micro')
            print(f"The micro F1 score is {f1_score_micro}")
            f1_score_weighted = f1_score(Y_test, y_hat, average='weighted')
            print(f"The weighted F1 score is {f1_score_weighted}")

            full_classification_report = classification_report(Y_test, y_hat)
            print(full_classification_report)
            self.evaluation_scores[f"{algorithm}"] = {
                'matthews': matthews,
                'roc_auc': roc_auc,
                'f1_score_macro': f1_score_macro,
                'f1_score_micro': f1_score_micro,
                'f1_score_weighted': f1_score_weighted,
                'classfication_report': full_classification_report
            }
            return self.evaluation_scores

    def regression_eval(self, algorithm, pred_probs=None, pred_reg=None):
        """
        Takes in the algorithm name. This is needed to grab saved predictions and to store regression scores
        of different evaluation functions within the class. Returns the evaluation dictionary.
        :param algorithm: Name of the used algorithm
        :param pred_probs: Probabilities of predictions
        :param pred_class: Predicted classes
        :return: Returns the evaluation dictionary.
        """
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            """
            We need a fallback logic as we might receive different types of data.
            If pred_class will not work with numpy arrays and needs the .any() function.
            """
            try:
                if pred_reg:
                    y_hat = pred_reg
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]
            except Exception:
                if pred_reg.any():
                    y_hat = pred_reg
                else:
                    y_hat = self.predicted_classes[f"{algorithm}"]

            """
            Calculating Matthews, ROC_AUC score and different F1 scores.
            """
            r2 = r2_score(Y_test, y_hat)
            print(f"The R2 score is {r2}")
            mean_absolute_error_score = mean_absolute_error(Y_test, y_hat)
            print(f"The MAE score is {mean_absolute_error_score}")
            median_absolute_error_score = median_absolute_error(Y_test, y_hat)
            print(f"The Median absolute error score is {median_absolute_error_score}")
            mean_squared_error_score = mean_squared_error(Y_test, y_hat, squared=False)
            print(f"The MSE score is {mean_squared_error_score}")
            root_mean_squared_error_score = mean_squared_error(Y_test, y_hat, squared=True)
            print(f"The RMSE score is {root_mean_squared_error_score}")

            self.evaluation_scores[f"{algorithm}"] = {
                'mae': mean_absolute_error_score,
                'r2_score': r2,
                'MSE': mean_squared_error_score,
                'RMSE': root_mean_squared_error_score,
                'median_absolute_error': median_absolute_error_score
            }
            return self.evaluation_scores