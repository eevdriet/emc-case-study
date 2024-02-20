from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd

from emc.model.scenario import Scenario
from emc.model.policy import Policy


class Classifier(ABC):
    SEED: int = 76

    def __init__(self, policy: Policy, train: pd.DataFrame, test: pd.DataFrame):
        self.policy = policy
        self.data = train
        self.test = test

    def run(self) -> float:
        """
        Run the classifier to find the labels of the given data
        :return: Results from the classifier
        """
        X_data, y_data = self._preprocess(self.data)
        X_test, y_test = self._preprocess(self.test)

        self._train(X_data, y_data)
        predictions = self._test(X_test, y_test)

        # threshold
        y_test = (y_test < 0.85).astype(int)
        predictions = (predictions < 0.85).astype(int)

        # check scores
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
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
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
    def test(self, X_test: pd.DataFrame) -> float:
        """
        Test the classifier by finding the label fitting its data
        :return: Multi-Criteria Decision Analysis composite score
        """
        ...
