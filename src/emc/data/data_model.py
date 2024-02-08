from abc import ABC, abstractmethod

import pandas as pd
from attrs import define


@define
class DataModel(ABC):
    # Annual summary of the simulated population right before administration of each PC round
    # NOTE: not filtered on the current simulation/scenario
    _epi_data: pd.DataFrame

    # Summary of the simulated population right before administration of each PC round
    # NOTE: not filtered on the current simulation/scenario
    _drug_data: pd.DataFrame

    @property
    def epidemiological_data(self):
        return self._retrieve_filtered_data(self._epi_data)

    @property
    def drug_efficacy_data(self):
        return self._retrieve_filtered_data(self._drug_data)

    def _retrieve_filtered_data(self, df: pd.DataFrame):
        """
        Retrieve the data of the given data frame when applying the filter
        :param df: Dataframe to filter
        :return: Filtered data frame
        """
        return df[self.filter_cond(df)]

    @abstractmethod
    def filter_cond(self, df: pd.DataFrame):
        """
        Filter condition for the given data model
        :param df: Data frame to filter on
        :return: Condition to filter on
        """
        ...
