import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from emc.classifiers import Classifier
from emc.model import Label


class SingleGradientBooster(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        
        
        # Transposing and flattening the DataFrame to create the features (X)
        X_data = data.to_numpy().T.flatten()
        # Extracting the target (y) - the final value of the last column
        y_data = np.array([data.iloc[-1, -1]])

        return X_data, y_data

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBClassifier(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {X_train[0]} simulations...")
        self.xgb.fit(X_train, y_train)

    def _test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {X_test[0]} simulations...")
        return self.xgb.predict(X_test)