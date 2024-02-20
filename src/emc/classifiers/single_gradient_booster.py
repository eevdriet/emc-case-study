import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from emc.classifiers import Classifier
from math import isnan


class SingleGradientBooster(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        groups = data.groupby(['scenario', 'simulation'])

        features = []
        targets = []

        for _, df in groups:
            target = df.reset_index(drop=True).loc[-1, 'target']
            if isnan(target):
                continue
            del df['target']
            targets.append(target)

            row = df.to_numpy().T.flatten()
            features.append(row)

        # Transposing and flattening the DataFrame to create the features (X)
        # Extracting the target (y) - the final value of the last column
        features = np.vstack(features)
        targets = np.array(targets)

        return features, targets

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBRegressor(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {len(X_train)} simulations...")
        self.xgb.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {len(X_test)} simulations...")
        return self.xgb.predict(X_test)
