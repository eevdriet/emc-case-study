import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from emc.classifiers import Classifier


class SingleGradientBooster(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> np.ndarray:
        groups = data.groupby(data.index)

        empty_array = np.empty((0, num_columns))

        n_cols = data.index.value_counts()[0]
        X_data = np.empty((0, n_cols))

        for _, df in groups:
            row = np.empty((0, n_cols))

            for name, series in df.items():
                cols = series.to_numpy().T.flatten()

                if name == 'target':
                    y_data = row
                else:
                    row = np.append(row, cols)

            X_data = np.vstack((X_data, row))

        # Transposing and flattening the DataFrame to create the features (X)
        # Extracting the target (y) - the final value of the last column

        return X_data, y_data

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBClassifier(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {X_train[0]} simulations...")
        self.xgb.fit(X_train, y_train)

    def _test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {X_test[0]} simulations...")
        return self.xgb.predict(X_test)
