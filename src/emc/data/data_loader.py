import json
import numpy as np

import pandas as pd
import pyreadr

from emc.model.scenario import Scenario, Simulation
from emc.model.label import Label
from emc.util import data_path


class DataLoader:
    def __init__(self, species: str):
        self.species = species
        self.monitor_age = self._load_monitor_ages()
        self.metadata = self._load_metadata()

    def load_scenarios(self, loadEfficacy: bool = False) -> list[Scenario]:
        """
        Load all scenarios from the monitor age table and metadata
        :return: Scenarios that were loaded from the data sets
        """
        if loadEfficacy: 
            self.drug_efficacy = self._load_drug_efficacy()

        return [self._load_scenario(scen_id, scenario, loadEfficacy) for scen_id, scenario in enumerate(self.metadata, start=1)]

    def _load_scenario(self, scen_id: int, metadata: dict, loadEfficacy: bool) -> Scenario:
        """
        Load specific scenario from its metadata
        :param scen_id: Identifier of the scenario
        :param metadata: Metadata of the scenario
        :return: Scenario loaded from the metadata
        """
        print(f"Loading scenario {scen_id}...")

        # Construct scenario from its metadata
        mda_freq = metadata['mda_freq']
        mda_strategy = metadata['mda_strategy']
        res_freq = metadata['p_SNP']
        res_mode = metadata['pheno_SNP']

        scenario = Scenario(id=scen_id, species=self.species, mda_freq=mda_freq,
                            mda_strategy=mda_strategy, res_freq=res_freq, res_mode=res_mode)

        # Load in its relevant simulations
        simulations = self._load_simulations(scenario, metadata, loadEfficacy)
        scenario.simulations = simulations

        return scenario

    def _load_simulations(self, scenario: Scenario, metadata: dict, loadEfficacy: bool) -> list[Simulation]:
        """
        Load the simulations of a given scenario
        :param scenario: Scenario for which to load the simulations
        :param metadata: Metadata of the scenario
        :return: Simulations from the given scenario
        """
        simulations = metadata['simulations']

        return [self._load_simulation(scenario, simulation_id, simulation, loadEfficacy)
                for simulation_id, simulation in enumerate(simulations, start=1)]

    def _load_simulation(self, scenario: Scenario, sim_id: int, metadata: dict, loadEfficacy: bool) -> Simulation:
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

        start = 84 * (1000 * (scenario.id - 1) + (sim_id - 1))
        monitor_age = self.monitor_age.iloc[start:start + 84]
        
        if (loadEfficacy):
            drug_efficacy_s = self.drug_efficacy.loc[(scenario.id, sim_id)]
        else:
            drug_efficacy_s = None

        label = Label.NO_SIGNAL if scenario.res_mode == 'none' else Label.SIGNAL

        return Simulation(id=sim_id, scenario=scenario, mda_time=mda_time,
                          mda_age=mda_age, mda_cov=mda_cov, monitor_age=monitor_age, drug_efficacy_s=drug_efficacy_s, label=label)

    def _load_metadata(self):
        """
        Load the metadata for all scenarios
        :return: Metadata
        """
        path = data_path() / f'{self.species}_metadata.json'

        with open(path, 'r') as file:
            return json.load(file)

    def _load_monitor_ages(self) -> pd.DataFrame:
        """
        Load the simulation data for all simulations
        :return: Simulation data
        """
        path = data_path() / f'{self.species}_monitor_age.csv'
        return pd.read_csv(path)
    
    def _load_drug_efficacy(self) -> pd.DataFrame:
        """
        Load the drug efficacy data for all simulations
        :return: Drug Efficacy data
        """
        path = data_path() / f'drug_efficacy_{self.species}.feather'
        df = pd.read_feather(path)
        if not isinstance(df.index, pd.MultiIndex):
            df.set_index(['scenario', 'simulation'], inplace=True)
        
        return df
