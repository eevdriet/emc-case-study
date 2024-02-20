import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from emc.classifiers import Classifier


class SingleGradientBooster(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        groups = data.groupby(['scenario', 'simulation'])

        features = []
        targets = []

        for _, df in groups:
            target = df.reset_index(drop=True).loc[0, 'target']
            del df['target']
            targets.append(target)

            row = df.to_numpy().T.flatten()
            features.append(row)

        # Transposing and flattening the DataFrame to create the features (X)
        # Extracting the target (y) - the final value of the last column
        features = np.vstack(features)
        target = np.array(target)

        return features, targets

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBClassifier(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {X_train[0]} simulations...")
        self.xgb.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame):
        print(f"Predicting with {X_test[0]} simulations...")
        return self.xgb.predict(X_test)
