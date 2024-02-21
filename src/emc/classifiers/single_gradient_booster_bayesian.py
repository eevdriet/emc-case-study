import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from emc.classifiers import Classifier
from math import isnan

optuna.logging.set_verbosity(optuna.logging.WARNING)

class SingleGradientBoosterBayesian(Classifier):
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

        def objective(trial):
            # Hyperparameters to be tuned by Optuna using log scale where appropriate
            hyperparams = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                # Example usage of suggest_float with step for parameters where linear scale search is appropriate
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
            }

            # Model initialization and training
            model = XGBRegressor(**hyperparams, random_state=self.SEED, missing=np.nan)
            model.fit(X_train, y_train)

            # Predictions and evaluation
            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)

            return mse

        # Create a study object and perform the optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, timeout=600)  # Adjust n_trials or timeout as needed

        # Best hyperparameters
        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")

        # Retraining the model on the best hyperparameters
        self.xgb = XGBRegressor(**best_hyperparams, random_state=self.SEED, missing=np.nan)
        self.xgb.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series = np.array([])):
        print(f"Predicting with {len(X_test)} simulations...")
        predictions = self.xgb.predict(X_test)
        return predictions
