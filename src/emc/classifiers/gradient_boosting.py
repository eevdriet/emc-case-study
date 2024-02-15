import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from emc.classifiers import Classifier
from emc.model import Label
from emc.model.scenario import Scenario


class GradientBoosting(Classifier):
    def _preprocess(self, data: list[Scenario]):
        features = []
        target = []

        for scenario in data:
            if scenario.mda_freq == 2 and scenario.mda_strategy == 'community':
                for simulation in scenario:
                    simulation.monitor_age = simulation.monitor_age[simulation.monitor_age['age_cat'] == 5]
                    simulation.monitor_age = simulation.monitor_age.drop(columns=['age_cat'])
                    features.append(simulation.monitor_age['n_host_eggpos'].tolist() + simulation.monitor_age['a_epg_obs'].tolist() + simulation.monitor_age['inf_level'].tolist())
                    target.append(simulation.label.value)

        return features, target

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.xgb = XGBClassifier(random_state=self.SEED, missing=np.NaN)
        print(f"Fitting with {X_train[0]} simulations...")
        self.xgb.fit(X_train, y_train)

    def _test(self, X_test: pd.DataFrame, y_test: pd.Series):
        print(f"Predicting with {X_test[0]} simulations...")
        return self.xgb.predict(X_test)