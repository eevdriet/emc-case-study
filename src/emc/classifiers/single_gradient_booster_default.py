import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from emc.classifiers import Classifier
from math import isnan


class SingleGradientBoosterDefault(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        groups = data.groupby(['scenario', 'simulation'])

        features = {}
        targets = {}

        for key, df in groups:
            df = df.drop(columns=['simulation', 'scenario', 'time', 'ERR'])

            df = df.reset_index(drop=True)
            target = df['target'].iloc[-1]
            if isnan(target):
                continue

            del df['target']
            targets[key] = target

            row = df.to_numpy().T.flatten()
            features[key] = row

        return features, targets

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBRegressor(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {len(X_train)} simulations...")
        self.xgb.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series = np.array([])):
        print(f"Predicting with {len(X_test)} simulations...")
        predictions = self.xgb.predict(X_test)
        return predictions