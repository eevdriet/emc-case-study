from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd

from emc.model.scenario import Scenario
from emc.model.policy import Policy


class Classifier(ABC):
    SEED: int = 76
    X = dict[tuple[int, int], np.ndarray]
    y = dict[tuple[int, int], np.array]

    def __init__(self, policy: Policy, train: pd.DataFrame, test: pd.DataFrame):
        # Raw data, including the policy to regress on
        self.policy = policy
        self.data = train
        self.test_data = test
        self.predictions: dict[tuple[int, int], bool] = {}

        # Preprocessed data
        self.features_data: Optional["X"] = None
        self.features_test: Optional["X"] = None
        self.targets_data: Optional["y"] = None
        self.targets_test: Optional["y"] = None

    def run(self) -> float:
        """
        Run the classifier to find the labels of the given data
        :return: Results from the classifier
        """
        # Preprocess to split data into features and targets per simulation
        self.features_data, self.targets_data = self._preprocess(self.data)
        self.features_test, self.targets_test = self._preprocess(self.test_data)

        # Merge the simulation data into usable arrays for the regressor and start training
        X_data = np.vstack(tuple(self.features_data.values()))
        y_data = np.array(tuple(self.targets_data.values()))
        self._train(X_data, y_data)

        # Train
        X_test = np.vstack(tuple(self.features_test.values()))
        y_test = np.array(tuple(self.targets_test.values()))
        predictions = self.test(X_test, y_test)

        # threshold
        y_test = (y_test < 0.85).astype(int)
        predictions = (predictions < 0.85).astype(int)

        # check scores
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        print(f"F1 score for {self.policy}: {f1}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame) -> tuple[X, y]:
        """
        Preprocess the training data
            Standardise all features
            Create X_train, y_train, X_test, y_test
        """
        ...

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.array):
        """
        Train the classifier on the training data
        """
        ...

    @abstractmethod
    def test(self, X_test: np.ndarray, y_test: np.array) -> float:
        """
        Test the classifier by finding the label fitting its data
        :return: Multi-Criteria Decision Analysis composite score
        """
        ...
