import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from emc.classifiers import Classifier
from emc.model import Label


class GradientBoostingClassifier(Classifier):
    def _preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        return super()._preprocess(features)

    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.gbc = GradientBoostingClassifier(random_state=self.SEED)
        print(f"Fitting with {len(X_train)} simulations...")
        self.gbc.fit(X_train, y_train)

    def _test(self, X_test: np.ndarray, y_test: np.ndarray):
        print(f"Predicting with {X_test.shape[0]} simulations...")
        return self.gbc.predict(X_test)
