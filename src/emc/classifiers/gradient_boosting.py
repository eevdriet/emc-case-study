import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from emc.classifiers import Classifier
from emc.model import Label


class GradientBoosting(Classifier):
    def _preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        return super()._preprocess(features)

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.gbc = GradientBoostingClassifier(random_state=self.SEED)
        print(f"Fitting with {X_train.shape[0]} simulations...")
        self.gbc.fit(X_train, y_train)

    def _test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {X_test.shape[0]} simulations...")
        return self.gbc.predict(X_test)
