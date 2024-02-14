import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from emc.classifiers import Classifier
from emc.model import Label


class GradientBoosting(Classifier):
    def _preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        return super()._preprocess(features)

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBClassifier(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {X_train[0]} simulations...")
        self.xgb.fit(X_train, y_train)

    def _test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {X_test[0]} simulations...")
        return self.xgb.predict(X_test)
