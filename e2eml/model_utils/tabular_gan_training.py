from e2eml.model_utils.tabular_gan_for_classification import (
    TabularGeneratorClassification,
)
from e2eml.model_utils.tabular_gan_for_regression import TabularGeneratorRegression


class TabularGan(TabularGeneratorClassification, TabularGeneratorRegression):
    def train_tabular_gans(self):
        if self.prediction_mode:
            pass
        else:
            self.set_random_seed()
            if self.class_problem in ("binary", "multiclass"):
                self.train_class_generators()
                self.create_synthetic_data()
            elif self.class_problem == "regression":
                self.train_regression_generators()
                self.create_synthetic_data_regression()
            else:
                pass
