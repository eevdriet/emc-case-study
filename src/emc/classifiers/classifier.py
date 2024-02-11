from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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
        indices = [i for i in range(len(features)) if len(features[i]) == 84]
        features = np.array([features[i] for i in indices])
        # print(features[0])

        # Trickery
        # Compound the data from all age groups per year
        new_features = []
        for sim in features:
            reshaped_sim = sim.reshape(-1, 4, 4)
            averaged_sim = reshaped_sim.mean(axis=1)
            # Add rate of change of number of infected hosts AND expected eggs level
            rate_of_change_3rd = np.diff(averaged_sim[:, 2], prepend=0)
            rate_of_change_4th = np.diff(averaged_sim[:, 3], prepend=0)
            averaged_sim = np.column_stack((averaged_sim, rate_of_change_3rd, rate_of_change_4th))
            # Remove time as a feature
            averaged_sim = np.delete(averaged_sim, 0, axis=1)
            new_features.append(averaged_sim)
        features = np.array(new_features)
        # Print final boys
        # print(features[0])
        # print(features[-1])

        print(features.shape)

        # ERROR: momenteel maakt ie van target een lijst van ALLE labels, ipv alleen een label per (scenario/simulation)
        target = np.array([0] * 4000 + [1] * 12000)
        target = target[indices]
        print(target.shape)

        # SMOTE
        # smote = SMOTE()
        # features_resampled, target_resampled = smote.fit_resample(features, target)
        # print(features.shape)
        # print(target.shape)

        return features, target

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the classifier on the training data
        """
        ...

    @abstractmethod
    def _test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Test the classifier by finding the label fitting its data
        :return: Multi-Criteria Decision Analysis composite score
        """
        ...
