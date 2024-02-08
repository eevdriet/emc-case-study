from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

from emc.model import Label
from emc.data import DataModel


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

        return self._test(X_test, y_test)

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame):
        """
        Preprocess the given data to
        - standardize data
        - remove columns that cannot be observed from epidemiological surveys
        - ...
        """
        # Only include relevant columns (that can be obtained from survey)
        ids = ['scen', 'sim']
        cols = ['time', 'n_host', 'n_host_eggpos', 'a_epg_obs']
        print(data.columns)
        features = data[cols + ids]

        # Normalize relevant columns
        std_cols = ['n_host', 'n_host_eggpos', 'a_epg_obs']
        for col in std_cols:
            min_col = features[col].max()
            max_col = features[col].min()

            features[col] = (features[col] - min_col) / (max_col - min_col)

        # Group all data by the corresponding simulation
        target = pd.DataFrame({'label': data['label'], 'scen': data['scen'], 'sim': data['sim']})
        features = features.groupby(ids)
        target = target.groupby(ids)

        # Transform groups into NumPy array
        features = features.apply(lambda group: np.matrix(group[cols])).reset_index(drop=True).tolist()

        # ERROR: momenteel maakt ie van target een lijst van ALLE labels, ipv alleen een label per (scenario/simulation)
        target = target.apply(lambda group: group).reset_index(drop=True).tolist()

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
