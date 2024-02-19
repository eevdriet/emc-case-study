from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd

from emc.model import Label
from emc.model.scenario import Scenario


class Classifier(ABC):
    TRAIN_TEST_SPLIT_SIZE: float = 0.2
    TRAIN_VAL_SPLIT_SIZE: float = 0.25
    SEED: int = 76

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def run(self, data: pd.DataFrame) -> float:
        """
        Run the classifier to find the labels of the given data
        :return: Results from the classifier
        """
        features, target = self._preprocess(data)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.TRAIN_TEST_SPLIT_SIZE,
                                                            random_state=self.SEED)

        self._train(X_train, y_train)
        predictions = self._test(X_test, y_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    @abstractmethod
    def _preprocess(self, data: list[Scenario]):
        """
        Preprocess the given data to
        - standardize data
        - remove columns that cannot be observed from epidemiological surveys
        - ...
        """

        features = []
        target = []

        for scenario in data:
            if scenario.mda_freq == 2 and scenario.mda_strategy == 'community':
                for simulation in scenario:
                    simulation.monitor_age = simulation.monitor_age[simulation.monitor_age['age_cat'] == 5]
                    simulation.monitor_age = simulation.monitor_age.drop(columns=['age_cat'])
                    features.append(simulation.monitor_age['n_host_eggpos'].tolist() + simulation.monitor_age[
                        'a_epg_obs'].tolist() + simulation.monitor_age['inf_level'].tolist())
                    target.append(simulation.label.value)

        return features, target

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: int):
        """
        Train the classifier on the training data
        """
        ...

    @abstractmethod
    def _test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Test the classifier by finding the label fitting its data
        :return: Multi-Criteria Decision Analysis composite score
        """
        ...
