import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from emc.regressors import Regressor
from emc.data.constants import SEED
from emc.log import setup_logger

logger = setup_logger(__name__)


class GradientBoosterDefault(Regressor):
    def _train(self, X_train: np.ndarray, y_train: np.array) -> None:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
        }

        self.parameters = params

        self.regression_model = XGBRegressor(**params, random_state=SEED, missing=np.nan)
        logger.debug(f"Fitting with {len(X_train)} simulations...")
        self.regression_model.fit(X_train, y_train)
