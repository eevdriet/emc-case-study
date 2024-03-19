import json
from typing import Optional

import numpy as np
import pandas as pd

from emc.data.constants import *
from emc.log import setup_logger
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.util import Paths

logger = setup_logger(__name__)


class DataLoader:
    """
    Class responsible for loading the following data files
    - Epidemiological survey data
    - Drug efficacy survey data
    - Metadata of the surveys
    """

    def __init__(self, species: str, *, use_merged: bool = True, load_efficacy: bool = True):
        self.species = species
        self.use_merged = use_merged
        self.load_efficacy = load_efficacy

        self.metadata: Optional[dict] = self._load_metadata()
        self.monitor_age: Optional[pd.DataFrame] = self._load_monitor_ages()
        self.drug_efficacy: Optional[pd.DataFrame] = self._load_drug_efficacy() if self.load_efficacy else None

    def load_scenarios(self) -> list[Scenario]:
        """
        Load all scenarios from the monitor age table and metadata
        :return: Scenarios that were loaded from the data sets
        """
        logger.info("Loading scenarios...")

        return [self._load_scenario(scenario_id, scenario) for scenario_id, scenario in
                enumerate(self.metadata, start=1)]

    def _load_scenario(self, scen_id: int, metadata: dict) -> Scenario:
        """
        Load specific scenario from its metadata
        :param scen_id: Identifier of the scenario
        :param metadata: Metadata of the scenario
        :return: Scenario loaded from the metadata
        """
        logger.debug(f"\t Scenario {scen_id}...")

        # Construct scenario from its metadata
        mda_freq = metadata['mda_freq']
        mda_strategy = metadata['mda_strategy']
        res_freq = metadata['p_SNP']
        res_mode = metadata['pheno_SNP']

        scenario = Scenario(id=scen_id, species=self.species, mda_freq=mda_freq,
                            mda_strategy=mda_strategy, res_freq=res_freq, res_mode=res_mode,
                            monitor_age=self.monitor_age)

        # Load in its relevant simulations
        simulations = self._load_simulations(scenario, metadata)
        scenario.simulations = simulations

        return scenario

    def _load_simulations(self, scenario: Scenario, metadata: dict) -> list[Simulation]:
        """
        Load the simulations of a given scenario
        :param scenario: Scenario for which to load the simulations
        :param metadata: Metadata of the scenario
        :return: Simulations from the given scenario
        """
        simulations = metadata['simulations']

        return [self._load_simulation(scenario, simulation_id, simulation)
                for simulation_id, simulation in enumerate(simulations, start=1)]

    def _load_simulation(self, scenario: Scenario, sim_id: int, metadata: dict) -> Simulation:
        """
        Load specific simulation from its metadata
        :param scenario: Scenario for which to load the simulations
        :param sim_id: Identifier of the simulation
        :param metadata: Metadata of the simulation
        :return: Simulation from the metadata
        """
        # Retrieve relevant data
        mda_time = metadata['mda_time']
        mda_age = metadata['mda_age']
        mda_cov = metadata['mda_cov']

        label = Label.NO_SIGNAL if scenario.res_mode == 'none' else Label.SIGNAL

        # Determines rows that belong to the current simulation
        coeff = 1 if self.use_merged else N_AGE_CATEGORIES
        step = coeff * N_YEARS
        start = step * (N_SIMULATIONS * (scenario.id - 1) + sim_id - 1)

        # Load monitor age
        monitor_age = self.monitor_age.iloc[start:start + step]

        # Load drug efficacy (if required)
        drug_efficacy_s = self.drug_efficacy.loc[(scenario.id, sim_id)] if self.load_efficacy else None

        return Simulation(id=sim_id, scenario=scenario, mda_time=mda_time,
                          mda_age=mda_age, mda_cov=mda_cov, monitor_age=monitor_age, drug_efficacy_s=drug_efficacy_s,
                          label=label)

    def _load_metadata(self) -> Optional[dict]:
        """
        Load the metadata for all scenarios
        :return: Metadata if available
        """
        path = Paths.worm_data(self.species, 'metadata', self.use_merged)
        if not path.exists():
            logger.error(f"Path {path} does not exist, cannot load in meta data!")
            return None

        with open(path, 'r') as file:
            return json.load(file)

    def _load_monitor_ages(self) -> Optional[pd.DataFrame]:
        """
        Load the epidemiological survey data for all simulations
        :return: Survey data if available
        """
        path = Paths.worm_data(self.species, 'monitor_age', self.use_merged)
        if not path.exists():
            logger.error(f"Path {path} does not exist, cannot load in epidemiological survey!")
            return None

        # Correctly order the columns and set data types
        df = pd.read_csv(path)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def _load_drug_efficacy(self) -> Optional[pd.DataFrame]:
        """
        Load the drug efficacy survey data for all simulations
        :return: Survey data if available
        """
        path = Paths.worm_data(self.species, 'drug_efficacy', self.use_merged)
        if not path.exists():
            logger.error(f"Path {path} does not exist, cannot load in drug efficacy survey!")
            return None

        # Correctly order the columns and set data types
        df = pd.read_csv(path)

        # Index based on the simulation/scenario
        # NOTE: this is REQUIRED, since not every simulation has the same number of observations
        if not isinstance(df.index, pd.MultiIndex):
            df.set_index(['scenario', 'simulation'], inplace=True)

        return df
