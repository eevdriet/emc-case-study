from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd

from emc.model.scenario import Scenario


class Classifier(ABC):
    SEED: int = 76

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self, data: pd.DataFrame) -> float:
        """
        Run the classifier to find the labels of the given data
        :return: Results from the classifier
        """
        X_data, y_data = self._preprocess(data)

        self._train(X_data, y_data)
        # predictions = self._test(X_test, y_test)

        # accuracy = accuracy_score(y_test, predictions)
        # precision = precision_score(y_test, predictions, average='weighted')
        # recall = recall_score(y_test, predictions, average='weighted')
        # f1 = f1_score(y_test, predictions, average='weighted')

        # return {
        #     'accuracy': accuracy,
        #     'precision': precision,
        #     'recall': recall,
        #     'f1_score': f1
        # }
    
    def _preprocess(self, train: pd.DataFrame, val: pd.DataFrame):
        """
        Preprocess the training data
            Standardise all features
            Create X_train, y_train, X_test, y_test
        """
        ...

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
