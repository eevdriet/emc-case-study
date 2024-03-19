import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from emc.regressors import Regressor
from emc.data.constants import SEED
from emc.log import setup_logger

logger = setup_logger(__name__)


class GradientBoosterRandomCV(Regressor):
    def _train(self, X_train: np.ndarray, y_train: np.array) -> None:
        logger.debug("Initializing XGBoost regressor with default parameters...")
        regression_model = XGBRegressor(random_state=SEED, missing=np.NaN)

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

        logger.debug("Setting up Randomized Search CV for hyperparameter optimization...")
        random_search = RandomizedSearchCV(estimator=regression_model, param_distributions=param_grid, n_iter=100,
                                           scoring='neg_mean_squared_error', n_jobs=-1, cv=5,
                                           random_state=SEED, verbose=1)

        logger.debug(f"Starting fitting process with {len(X_train)} samples...")
        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # After fitting
        logger.debug("Fitting complete.")
        logger.debug("Best hyperparameters found:")
        logger.debug(random_search.best_params_)

        # Best estimator
        self.regression_model = random_search.best_estimator_

        self.parameters = random_search.best_params_
        logger.debug("Best estimator and parameters set for the model.")
