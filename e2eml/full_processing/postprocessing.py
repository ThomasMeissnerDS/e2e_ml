from e2eml.full_processing import cpu_preprocessing
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging


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
        logging.info('Started creating SHAP values.')
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
        logging.info('Finished creating SHAP values.')

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
            logging.info('Skipped classification evaluation due to prediction mode.')
            pass
        else:
            logging.info('Started classification evaluation.')
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
            logging.info(f'The Matthew correlation of {algorithm} is {matthews}')

            if self.class_problem == 'binary':
                def get_preds(threshold, probabilities):
                    return [1 if prob > threshold else 0 for prob in probabilities]
                roc_values = []
                for thresh in np.linspace(0, 1, 100):
                    preds = get_preds(thresh, y_hat_probs)
                    tn, fp, fn, tp = confusion_matrix(Y_test, preds).ravel()
                    tpr = tp/(tp+fn)
                    fpr = fp/(fp+tn)
                    roc_values.append([tpr, fpr])
                tpr_values, fpr_values = zip(*roc_values)
                fig, ax = plt.subplots(figsize=(10,7))
                ax.plot(fpr_values, tpr_values)
                ax.plot(np.linspace(0, 1, 100),
                        np.linspace(0, 1, 100),
                        label='baseline',
                        linestyle='--')
                plt.title('Receiver Operating Characteristic Curve', fontsize=18)
                plt.ylabel('TPR', fontsize=16)
                plt.xlabel('FPR', fontsize=16)
                plt.legend(fontsize=12)
                plt.show()
                roc_auc = roc_auc_score(Y_test, y_hat_probs)
                print(f"The ROC_AUC score is {roc_auc}")
            else:
                pass
            f1_score_macro = f1_score(Y_test, y_hat, average='macro')
            print(f"The macro F1 score is {f1_score_macro}")
            logging.info(f'The macro F1 score of {algorithm} is {f1_score_macro}')
            f1_score_micro = f1_score(Y_test, y_hat, average='micro')
            print(f"The micro F1 score is {f1_score_micro}")
            logging.info(f'The micro F1 score of {algorithm} is {f1_score_micro}')
            f1_score_weighted = f1_score(Y_test, y_hat, average='weighted')
            print(f"The weighted F1 score is {f1_score_weighted}")
            logging.info(f'The weighted F1 score of {algorithm} is {f1_score_weighted}')

            full_classification_report = classification_report(Y_test, y_hat)
            print(full_classification_report)
            logging.info(f'The classification report of {algorithm} is {full_classification_report}')
            self.evaluation_scores[f"{algorithm}"] = {
                'matthews': matthews,
                'roc_auc': roc_auc,
                'f1_score_macro': f1_score_macro,
                'f1_score_micro': f1_score_micro,
                'f1_score_weighted': f1_score_weighted,
                'classfication_report': full_classification_report
            }
            logging.info('Finished classification evaluation.')
            return self.evaluation_scores

    def regression_eval(self, algorithm, pred_reg=None):
        """
        Takes in the algorithm name. This is needed to grab saved predictions and to store regression scores
        of different evaluation functions within the class. Returns the evaluation dictionary.
        :param algorithm: Name of the used algorithm
        :param pred_probs: Probabilities of predictions
        :param pred_class: Predicted classes
        :return: Returns the evaluation dictionary.
        """
        if self.prediction_mode:
            logging.info('Skipped regression evaluation due to prediction mode.')
            pass
        else:
            logging.info('Started regression evaluation.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            """
            We need a fallback logic as we might receive different types of data.
            If pred_class will not work with numpy arrays and needs the .any() function.
            """
            try:
                if pred_reg:
                    y_hat = pred_reg
                else:
                    y_hat = self.predicted_values[f"{algorithm}"]
            except Exception:
                if pred_reg.any():
                    y_hat = pred_reg
                else:
                    y_hat = self.predicted_values[f"{algorithm}"]

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
            logging.info('Finished regression evaluation.')
            return self.evaluation_scores