import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score
from xgboost import XGBRegressor

from emc.regressors import Regressor
from emc.data.constants import SEED
from emc.log import setup_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = setup_logger(__name__)


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

            logger.debug(f"Trial {trial.number}: Testing with params {hyperparams}")

            model = XGBRegressor(**hyperparams, random_state=SEED, missing=np.nan)
            model.fit(X_train, y_train)

            X_test = np.vstack(tuple(self.features_test.values()))
            y_test = np.array(tuple(self.targets_test.values()))

            predictions = model.predict(X_test)
            y_test = (y_test < 0.85).astype(int)
            predictions = (predictions < 0.85).astype(int)

            f1 = f1_score(y_test, predictions, average='weighted')

            logger.debug(f"Trial {trial.number} F1: {f1}")

            return f1

        study = optuna.create_study(direction='maximize')
        logger.debug("Optimization process started...")
        study.optimize(objective, n_trials=5, timeout=600)
        logger.debug("Optimization process completed.")

        best_hyperparams = study.best_trial.params
        logger.debug(f"Best hyperparameters: {best_hyperparams}")

        self.parameters = best_hyperparams

        self.regression_model = XGBRegressor(**best_hyperparams, random_state=SEED, missing=np.nan)
        self.regression_model.fit(X_train, y_train)
        logger.debug("Final model trained with best hyperparameters.")
