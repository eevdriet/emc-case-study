from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd
from xgboost import XGBRegressor

from emc.model.simulation import Simulation
from emc.model.policy import Policy
from emc.data.constants import SEED

_X = dict[tuple[int, int], np.ndarray]
_Y = dict[tuple[int, int], float]


class Classifier(ABC):
    SEED: int = 76

    def __init__(self, policy: Policy, train: pd.DataFrame, test: pd.DataFrame):
        # Raw data, including the policy to regress on
        self.policy = policy
        self.data = train
        self.test_data = test
        self.predictions: dict[tuple[int, int], bool] = {}
        self.parameters: dict[str, float] = False

        # Preprocessed data
        self.features_data: Optional[_X] = None
        self.features_test: Optional[_X] = None
        self.targets_data: Optional[_Y] = None
        self.targets_test: Optional[_Y] = None

        self.xgb = None

    def run(self) -> None:
        """
        Run the classifier to find the labels of the given data
        """
        import time  # Importing the time module

        # Function to calculate and print runtime
        def print_runtime(start_time, end_time, description):
            runtime = end_time - start_time
            print(f"Runtime for {description}: {runtime} seconds")

        start_time = time.time()  # Start timing the execution

        # Preprocess to split data into features and targets per simulation
        preprocess_start_time = time.time()
        self.features_data, self.targets_data = self._preprocess(self.data)
        self.features_test, self.targets_test = self._preprocess(self.test_data)
        preprocess_end_time = time.time()
        print_runtime(preprocess_start_time, preprocess_end_time, "preprocessing")

        # Merge the simulation data into usable arrays for the regressor and start training
        merge_start_time = time.time()
        X_data = np.vstack(tuple(self.features_data.values()))
        y_data = np.array(tuple(self.targets_data.values()))
        merge_end_time = time.time()
        print_runtime(merge_start_time, merge_end_time, "merging data")

        # Either train the classifier with pre-optimized hyperparameter or perform search for optimal hyperparameters
        if self.xgb == None:
            if self.parameters:
                print("Using already stored hyperparameters")
                train_start_time = time.time()
                self._train_basic(X_data, y_data)
                train_end_time = time.time()
                print_runtime(train_start_time, train_end_time, "training with stored hyperparameters")
            else:
                print("Generating new hyperparameters")
                train_start_time = time.time()
                self._train(X_data, y_data)
                train_end_time = time.time()
                print_runtime(train_start_time, train_end_time, "training with new hyperparameters")

        # Train
        train_start_time = time.time()
        X_test = np.vstack(tuple(self.features_test.values()))
        y_test = np.array(tuple(self.targets_test.values()))
        predictions = self.test(X_test, y_test)
        train_end_time = time.time()
        print_runtime(train_start_time, train_end_time, "testing")

        # Threshold
        threshold_start_time = time.time()
        y_test = (y_test < 0.85).astype(int)
        predictions = (predictions < 0.85).astype(int)
        threshold_end_time = time.time()
        print_runtime(threshold_start_time, threshold_end_time, "thresholding")

        # Check scores
        score_start_time = time.time()
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        score_end_time = time.time()
        print_runtime(score_start_time, score_end_time, "calculating scores")

        print(f"F1 score for {self.policy}: {f1}")

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        end_time = time.time()  # Stop timing the execution
        print_runtime(start_time, end_time, "entire process")

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame) -> tuple[_X, _Y]:
        """
        Preprocess the training data
            Standardise all features
            Create X_train, y_train, X_test, y_test
        """
        ...

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.array) -> None:
        """
        :param X_train: Train features
        :param y_train: Train targets
        """
        ...

    def _train_basic(self, X_train: pd.DataFrame, y_train: pd.Series):
        params = self.parameters
        self.xgb = XGBRegressor(**params, random_state=SEED, missing=np.nan)
        print(f"Fitting with {len(X_train)} simulations...")
        self.xgb.fit(X_train, y_train)

    @abstractmethod
    def test(self, X_test: np.ndarray, y_test: np.array) -> np.array:
        """
        :param X_test: Test features
        :param y_test: Test targets
        :return: Prediction for each target based on the features
        """
        ...

    def getParameters(self) -> dict:
        """
        Get the used hyperparameters
        :return: dict containing the hyperparameters
        """
        return self.parameters

    def setParameters(self, params) -> None:
        """
        Set the already found hyperparameters
        """
        self.parameters = params

    def predict(self, simulation: Simulation) -> Optional[float]:
        """
        Predict the signal from a single simulation
        :param simulation: Simulation to test the data for
        :return: Prediction for the simulation if simulation is valid
        """
        # Retrieve the relevant data for the simulation
        key = (simulation.scenario.id, simulation.id)
        if key not in self.features_test or key not in self.targets_test:
            return None

        # Reshape the simulation data into single rowed data
        X_test = self.features_test[key].reshape(1, -1)
        y_test = np.array([self.targets_test[key]])

        # Prediction (only first result needed as only one row tested)
        return self.test(X_test, y_test)[0]
    
    def getModel(self):
        return self.xgb
    
    def setModel(self, model):
        self.xgb = model

    @staticmethod
    def createInstance(constructor, model, policy: Policy, train, test):
        newClassifier = constructor(policy, train, test)
        newClassifier.setModel(model)
        return newClassifier
