import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from emc.classifiers import Classifier
from emc.data.constants import SEED
from emc.log import setup_logger
from math import isnan

logger = setup_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SingleGradientBoosterBayesian(Classifier):
    def _preprocess(self, data: pd.DataFrame):
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

            logger.debug(f"Trial {trial.number}: Testing with params {hyperparams}")

            model = XGBRegressor(**hyperparams, random_state=SEED, missing=np.nan)
            model.fit(X_train, y_train)

            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)

            logger.debug(f"Trial {trial.number} MSE: {mse}")

            return mse

        study = optuna.create_study(direction='minimize')
        logger.debug("Optimization process started...")
        study.optimize(objective, n_trials=100, timeout=600)
        logger.debug("Optimization process completed.")

        best_hyperparams = study.best_trial.params
        logger.debug(f"Best hyperparameters: {best_hyperparams}")

        self.parameters = best_hyperparams

        self.xgb = XGBRegressor(**best_hyperparams, random_state=SEED, missing=np.nan)
        self.xgb.fit(X_train, y_train)
        logger.debug("Final model trained with best hyperparameters.")

    def test(self, X_test: np.ndarray, y_test: np.array) -> np.array:
        logger.debug(f"Predicting with {len(X_test)} simulations...")
        predictions = self.xgb.predict(X_test)
        return predictions
