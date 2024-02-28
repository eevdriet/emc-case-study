import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from emc.regressors import Regressor
from emc.data.constants import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GradientBoosterOptuna(Regressor):
    def _train(self, X_train: np.ndarray, y_train: np.array) -> None:
        def objective(trial):
            hyperparams = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
            }

            print(f"Trial {trial.number}: Testing with params {hyperparams}")

            model = XGBRegressor(**hyperparams, random_state=SEED, missing=np.nan)
            model.fit(X_train, y_train)

            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)

            print(f"Trial {trial.number} MSE: {mse}")

            return mse

        study = optuna.create_study(direction='minimize')
        print("Optimization process started...")
        study.optimize(objective, n_trials=100, timeout=600)
        print("Optimization process completed.")

        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")

        self.parameters = best_hyperparams

        self.regression_model = XGBRegressor(**best_hyperparams, random_state=SEED, missing=np.nan)
        self.regression_model.fit(X_train, y_train)
        print("Final model trained with best hyperparameters.")

    def test(self, X_test: np.ndarray, y_test: np.array) -> np.array:
        print(f"Predicting with {len(X_test)} simulations...")
        predictions = self.regression_model.predict(X_test)
        return predictions
