import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from emc.classifiers import Classifier
from math import isnan


class SingleGradientBoosterRandomCV(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        groups = data.groupby(['scenario', 'simulation'])

        features = {}
        targets = {}

        for key, df in groups:
            df = df.drop(columns=['simulation', 'scenario', 'time', 'ERR'])

            df = df.reset_index(drop=True)
            target = df['target'].iloc[-1]

            # TODO: Consider using isnan or using the last working target value
            if isnan(target):
                continue

            del df['target']
            targets[key] = target

            row = df.to_numpy().T.flatten()
            features[key] = row

        return features, targets

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("Initializing XGBoost regressor with default parameters...")
        xgb = XGBRegressor(random_state=self.SEED, missing=np.NaN)

        # Define the parameter grid to search
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2, 3]
        }

        print("Setting up Randomized Search CV for hyperparameter optimization...")
        random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, n_iter=100,
                                        scoring='neg_mean_squared_error', n_jobs=-1, cv=5,
                                        random_state=self.SEED, verbose=1)

        print(f"Starting fitting process with {len(X_train)} samples...")
        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # After fitting
        print("Fitting complete.")
        print("Best hyperparameters found:")
        print(random_search.best_params_)

        # Best estimator
        self.xgb = random_search.best_estimator_

        self.parameters = random_search.best_params_
        print("Best estimator and parameters set for the model.")

    def test(self, X_test: pd.DataFrame, y_test: pd.Series = np.array([])):
        print(f"Predicting with {len(X_test)} simulations...")
        predictions = self.xgb.predict(X_test)
        return predictions