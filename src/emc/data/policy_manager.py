import pandas as pd
import random
from collections import defaultdict

from emc.model.policy import Policy
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.data.constants import *
from emc.util import Writer, Paths

from emc.classifiers import *
from emc.util import normalised, Pair


class PolicyManager:
    # Data is split into pairs of train/test data
    # The data consists of the simulations and their (combined) data as data frame
    SplitData = tuple[Pair[list[Simulation]], Pair[pd.DataFrame]]

    """
    Manages the classification of different policies and their sub-policies
    """

    __TRAIN_TEST_SPLIT_SIZE: float = 0.2
    __TRAIN_VAL_SPLIT_SIZE: float = 0.25
    __NORMALISED_COLS = {'n_host', 'n_host_eggpos', 'a_epg_obs'}

    def __init__(self, scenarios: list[Scenario], strategy: str, frequency: str, worm: str, regression_model: int):
        regressor_constructors = {
            0: SingleGradientBoosterDefault,
            1: SingleGradientBoosterRandomCV,
            2: SingleGradientBoosterBayesian
        }

        self.scenarios: list[Scenario] = scenarios
        self.policy_classifiers = {}
        self.policy_costs = {}

        self.train_simulations = []
        self.test_simulations = []

        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.strategy = str(strategy)
        self.frequency = str(frequency)
        self.worm = str(worm)

        filename = self.worm + "_" + self.strategy + "_" + self.frequency + "_" + regressor_constructors[regression_model].__name__ + ".json"
        self.hp_path = Paths.hyperparameter_opt(filename)
        
        filename = "classifier_stats_" + self.worm + "_" + self.strategy + "_" + self.frequency + "_" + regressor_constructors[regression_model].__name__ + ".json"
        self.plot_path = Paths.hyperparameter_opt(filename)
        self.constructor = regressor_constructors[regression_model]

    def manage(self):
        # Split the data into train/validation data for the classifiers
        simulations, dfs = self.__split_data()
        self.train_simulations, self.test_simulations = simulations
        self.train_df, self.test_df = dfs

        # Go through a policy and its sub-policies
        policy = self.__create_init_policy()

        for sub_policy in policy.sub_policies:
            # Already trained the classifier for the given policy
            if sub_policy in self.policy_classifiers:
                continue

            # Otherwise, filter the train/validation data based on the policy and start classifying
            train = self.__filter_data(self.train_df, sub_policy)
            test = self.__filter_data(self.test_df, sub_policy)

            found = Writer.get_value_from_json(self.hp_path, str(hash(sub_policy)))

            classifier = self.constructor(sub_policy, train, test)
            classifier.setParameters(found)
            classifier_stats = classifier.run()

            Writer.update_json_file(self.plot_path, str(sub_policy.epi_time_points[-1]), classifier_stats)
            
            if not found:
                Writer.update_json_file(self.hp_path, str(hash(sub_policy)), classifier.getParameters())

            # Store the classifier results
            self.policy_classifiers[sub_policy] = classifier

        # # Keep track of the costs of all simulations that terminate in a certain policy
        # policy_simulation_costs: dict[Policy, list] = defaultdict(list)

        # for simulation in self.test_simulations:
        #     for sub_policy in policy.sub_policies:
        #         classifier = self.policy_classifiers[sub_policy]

        #         # Continue with epidemiological surveys as long as resistance does not seem to be a problem yet
        #         epi_signal = classifier.predict(simulation)
        #         if epi_signal is None:
        #             continue
        #         if epi_signal >= 0.85:
        #             continue

        #         # Otherwise, verify whether resistance is a problem by scheduling a drug efficacy the year after
        #         drug_signal = simulation.predict(sub_policy)

        #         # If no drug efficacy data is available, penalize the policy for not finding a signal sooner
        #         if drug_signal is None:
        #             costs = simulation.calculate_cost(sub_policy)
        #             costs += RESISTANCE_NOT_FOUND_COSTS
        #             policy_simulation_costs[sub_policy].append(costs)
        #             print(
        #                 f"Simulation {simulation.scenario.id, simulation.id} -> {sub_policy} with costs {costs} [Epi < 0.85, no drug data]")
        #             break

        #         # If data is available and resistance is indeed a problem, stop the simulation and register its cost
        #         elif drug_signal < 0.85:
        #             drug_policy = sub_policy.with_drug_survey()
        #             costs = simulation.calculate_cost(sub_policy)
        #             print(
        #                 f"Simulation {simulation.scenario.id, simulation.id} -> {sub_policy} with costs {costs} [Epi < 0.85, drug < 0.85]")
        #             policy_simulation_costs[drug_policy].append(costs)
        #             break

        #     # If resistance never becomes a problem under the policy, register its costs without drug efficacy surveys
        #     else:
        #         costs = simulation.calculate_cost(policy)
        #         print(
        #             f"Simulation {simulation.scenario.id, simulation.id} -> {policy} with costs {costs} [Epi>= 0.85, drug >= 0.85]")
        #         policy_simulation_costs[policy].append(costs)

        # # Register the average costs of each of the observed sub-policies
        # for policy, simulation_costs in policy_simulation_costs.items():
        #     if len(simulation_costs) == 0:
        #         continue

        #     self.policy_costs[policy] = sum(simulation_costs) / len(simulation_costs)

        # TODO: neighborhood descent for the next policy

    def __create_init_policy(self) -> Policy:
        """
        Create an initial policy to start the policy improvement from
        :return: Initial policy
        """
        self.scenarios = self.scenarios

        every_n_year = 4
        # tests = (True,) + (False,) * (every_n_year - 1)
        # epi_surveys = tests * (N_YEARS // every_n_year) + tests[:N_YEARS % every_n_year]

        epi_surveys = (True,) * 21

        return Policy(epi_surveys)

    def __split_data(self) -> SplitData:
        """
        Split all simulation data into a train and test set
        :return: Train and test simulations/data frames
        """

        # Combine all simulations from all scenarios
        simulations: list[Simulation] = []
        for scenario in self.scenarios:
            simulations += scenario.simulations

        # Randomly order the simulations and split into train/validation
        random.seed(SEED)
        random.shuffle(simulations)

        # Split the simulations
        split_idx = int(len(simulations) * self.__TRAIN_TEST_SPLIT_SIZE)
        train_sims = simulations[split_idx:]
        test_sims = simulations[:split_idx]

        # Combine all simulation data and normalise relevant columns
        df = pd.concat([simulation.monitor_age for simulation in simulations])
        for col in self.__NORMALISED_COLS:
            df[col] = normalised(df[col])

        # Split the data frames
        split_idx = int(len(df) * self.__TRAIN_TEST_SPLIT_SIZE)
        train_df = df.iloc[split_idx:]
        test_df = df.iloc[:split_idx]

        print("Created train/test")
        return (train_sims, test_sims), (train_df, test_df)

    @classmethod
    def __filter_data(cls, df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
        """
        Filter a data frame given a policy
        :param df: Data frame to filter
        :param policy: Policy to filter by
        :return: Filtered data frame
        """
        # Select only time points that occur in the policy
        time_points = policy.epi_time_points
        return df[df['time'].isin(time_points)]


def main():
    from emc.data.data_loader import DataLoader

    # Get the data
    for worm in Worm:
        worm = worm.value

        loader = DataLoader(worm)
        all_scenarios = loader.load_scenarios()

        for frequency in MDA_FREQUENCIES:
            for strategy in MDA_STRATEGIES:
                scenarios = [
                    s for s in all_scenarios
                    if s.mda_freq == frequency and s.mda_strategy == strategy
                ]

                # Use the policy manager
                print(f"\n\n\n-- {worm}: {strategy} with {frequency} --")
                manager = PolicyManager(scenarios, strategy, frequency, worm, 2)
                manager.manage()


if __name__ == '__main__':
    main()
