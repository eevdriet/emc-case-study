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
        self.parameters: dict[str, float] = False

        # Preprocessed data
        self.features_data: Optional[_X] = None
        self.features_test: Optional[_X] = None
        self.targets_data: Optional[_Y] = None
        self.targets_test: Optional[_Y] = None

        # To be Loaded data
        self.xgb = None
        self.X_test = None
        self.y_test = None
        self.X_data = None
        self.y_data = None

    def initialize_and_train_model(self) -> None:
        """
        This method initializes the dataset and test set if they are not already set up. It preprocesses the input and test data,
        setting up feature and target variables for both. If the XGBoost model is not initialized, the method proceeds to train the
        model. It checks if hyperparameters are provided; if they are, it uses these to train the model. Otherwise, it generates
        new hyperparameters and then trains the model. This method ensures that the data is prepared and the model is trained,
        ready for further processing or evaluation.
        """
        if self.features_data is None or self.targets_data is None or self.features_test is None or self.targets_test is None:
            self.features_data, self.targets_data = self._preprocess(self.data)
            self.features_test, self.targets_test = self._preprocess(self.test_data)

        X_data = np.vstack(tuple(self.features_data.values()))
        y_data = np.array(tuple(self.targets_data.values()))
        X_test = np.vstack(tuple(self.features_test.values()))
        y_test = np.array(tuple(self.targets_test.values()))

        if self.xgb == None:
            if self.parameters:
                print("Using already stored hyperparameters")
                self._train_basic(self.X_data, self.y_data)
            else:
                print("Generating new hyperparameters")
                self._train(self.X_data, self.y_data)

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
        """
        Get the model used in the classifier.
        :return: The model object.
        """
        return self.xgb
        
    def setModel(self, model):
        """
        Set the model for the classifier.
        :param model: The model to be set.
        """
        self.xgb = model
        
    def getPreprocessing(self):
        """
        Retrieve the preprocessing data including features and targets for training and testing.
        :return: A tuple containing features and targets for training and testing.
        """
        return (self.features_data, self.targets_data, self.features_test, self.targets_test)
        
    def setPreprocessing(self, features_data, targets_data, features_test, targets_test) -> None:
        """
        Set the preprocessing data for the classifier.
        :param features_data: Training features data.
        :param targets_data: Training targets data.
        :param features_test: Testing features data.
        :param targets_test: Testing targets data.
        """
        self.features_data = features_data
        self.targets_data = targets_data
        self.features_test = features_test
        self.targets_test = targets_test

    @staticmethod
    def createInstance(constructor, model, policy: Policy, train, test):
        """
        Create a new instance of the classifier.
        :param constructor: The constructor for the classifier.
        :param model: The model to be used in the classifier.
        :param policy: The policy to be used in the classifier.
        :param train: The training data.
        :param test: The testing data.
        :return: A new instance of the classifier.
        """
        newClassifier = constructor(policy, train, test)
        newClassifier.setModel(model)
        return newClassifier
        
    def getStats(self):
        """
        Calculate and return the statistics including accuracy, precision, recall, and F1 score for the model.
        :return: A dictionary containing the calculated statistics.
        """
        X_test_local = self.X_test
        y_test_local = self.y_test

        predictions = self.test(X_test_local, y_test_local)
        y_test_local = (y_test_local < 0.85).astype(int)
        predictions = (predictions < 0.85).astype(int)

        accuracy = accuracy_score(y_test_local, predictions)
        precision = precision_score(y_test_local, predictions, average='weighted')
        recall = recall_score(y_test_local, predictions, average='weighted')
        f1 = f1_score(y_test_local, predictions, average='weighted')

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return results
