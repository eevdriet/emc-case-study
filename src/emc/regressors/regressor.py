from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd

from emc.model.simulation import Simulation
from emc.model.policy import Policy
from emc.data.constants import SEED

_X = dict[tuple[int, int], np.ndarray]
_Y = dict[tuple[int, int], float]


class Regressor(ABC):
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

        # To be Loaded model
        self.regression_model = None

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

        if self.regression_model == None:
            self._train(X_data, y_data)

    def _preprocess(self, data: pd.DataFrame) -> tuple[_X, _Y]:
        """
        Preprocess the training data
            Standardise all features
            Create X_train, y_train, X_test, y_test
        """

        data = data.copy()

        # Calculate percentage changes without looping
        data['inf_level_change'] = data.groupby(['scenario', 'simulation'])['inf_level'].pct_change(fill_method=None)
        data['a_epg_obs_change'] = data.groupby(['scenario', 'simulation'])['a_epg_obs'].pct_change(fill_method=None)

        # Replace inf values with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Prepare the target and features dictionaries
        features = {}
        targets = {}

        # Iterate over the groups to assemble the final features and targets
        for (scenario, simulation), df in data.groupby(['scenario', 'simulation']):
            # Get the last target value and skip if it's NaN
            target = df['target'].iloc[-1]
            if np.isnan(target):
                continue
            targets[(scenario, simulation)] = target

            # Prepare the feature array without the target and other unnecessary columns
            feature_df = df.drop(columns=['target', 'simulation', 'scenario', 'time', 'ERR']).reset_index(drop=True)
            features[(scenario, simulation)] = feature_df.to_numpy().T.flatten()

        return features, targets

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.array) -> None:
        """
        :param X_train: Train features
        :param y_train: Train targets
        """
        ...

    @abstractmethod
    def test(self, X_test: np.ndarray, y_test: np.array) -> np.array:
        """
        :param X_test: Test features
        :param y_test: Test targets
        :return: Prediction for each target based on the features
        """
        ...

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
        return self.regression_model
        
    def setModel(self, model):
        """
        Set the model for the classifier.
        :param model: The model to be set.
        """
        self.regression_model = model
        
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
    
    def getParameters(self):
        if self.regression_model == None:
            return None
        else:
            return self.regression_model.get_params()
        
    def getStats(self):
        """
        Calculate and return the statistics including accuracy, precision, recall, and F1 score for the model.
        :return: A dictionary containing the calculated statistics.
        """

        X_test = np.vstack(tuple(self.features_test.values()))
        y_test = np.array(tuple(self.targets_test.values()))

        predictions = self.test(X_test, y_test)
        y_test = (y_test < 0.85).astype(int)
        predictions = (predictions < 0.85).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return results
