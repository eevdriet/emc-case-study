from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

from emc.model import Label, Scenario
from emc.data import DataModel


class Classifier(ABC):
    TRAIN_TEST_SPLIT_SIZE: float = 0.2
    TRAIN_VAL_SPLIT_SIZE: float = 0.25
    SEED: int = 76

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def run(self, data: list[Scenario]) -> float:
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
    def _preprocess(self, data: list[Scenario]):
        """
        Preprocess the given data to
        - standardize data
        - remove columns that cannot be observed from epidemiological surveys
        - ...
        """

        # features = pd.DataFrame()
        # target = pd.DataFrame()

        # time_data = data[0]._epi_data['time']
        # n_host_data = data[0]._epi_data['n_host']
        # n_host_eggpos_data = data[0]._epi_data['n_host_eggpos']
        # a_epg_obs_data = data[0]._epi_data['a_epg_obs']
        

        # start_indices = []

        # tolerance = 0.001

        # start_indices.append(0)
        # for i in range(1, len(time_data)):
        #     if time_data[i] < tolerance and abs(time_data[i - 1]) > tolerance:
        #         start_indices.append(i)

        # time_result = [[] for _ in range(len(start_indices) * 4)]
        # n_host_result = [[] for _ in range(len(start_indices) * 4)]
        # n_host_eggpos_result = [[] for _ in range(len(start_indices) * 4)]
        # a_epg_obs_result = [[] for _ in range(len(start_indices) * 4)]

        # for i in range(len(start_indices)):
        #     start_index = start_indices[i]
        #     end_index = start_indices[i + 1] if i < len(start_indices) - 1 else len(time_data)
        #     sublist_length = end_index - start_index
            
        #     for j in range(sublist_length):
        #         sublist_index = i * 4 + (j % 4)
        #         time_result[sublist_index].append(time_data[start_index + j])
        #         n_host_result[sublist_index].append(n_host_data[start_index + j])
        #         n_host_eggpos_result[sublist_index].append(n_host_eggpos_data[start_index + j])
        #         a_epg_obs_result[sublist_index].append(a_epg_obs_data[start_index + j])


        # print(time_result[0])

        # target = [[] for _ in range(len(start_indices))]

        


        # Only include relevant columns (that can be obtained from survey)
        ids = ['scen', 'sim']
        cols = ['time', 'n_host', 'n_host_eggpos', 'a_epg_obs']

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
