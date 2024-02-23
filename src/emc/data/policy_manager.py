import json
from pathlib import Path

import pandas as pd
import random
from collections import defaultdict
from emc.util import Paths
from math import isnan
import logging

#init logging
log_directory = Paths.log()
Path(log_directory).mkdir(parents=True, exist_ok=True)
log_file_path = log_directory / 'policy_manager.log'

logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from emc.model.policy import Policy, create_init_policy
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.data.constants import *
from emc.util import Writer, Paths

from emc.classifiers import *
from emc.data.neighborhood import Neighborhood
from emc.util import normalised, Pair


class PolicyManager:
    # Data is split into pairs of train/test data
    # The data consists of the simulations and their (combined) data as data frame
    SplitData = tuple[Pair[list[Simulation]], Pair[pd.DataFrame]]

    """
    Manages the classification of different policies and their sub-policies
    """

    __N_MAX_ITERS: int = 5
    __TRAIN_TEST_SPLIT_SIZE: float = 0.2
    __TRAIN_VAL_SPLIT_SIZE: float = 0.25
    __NORMALISED_COLS = {'n_host', 'n_host_eggpos', 'a_epg_obs'}

    def __init__(self, scenarios: list[Scenario], strategy: str, frequency: int, worm: str, regression_model: int,
                 neighborhoods: list[Neighborhood]):
        regressor_constructors = {
            0: SingleGradientBoosterDefault,
            1: SingleGradientBoosterRandomCV,
            2: SingleGradientBoosterBayesian
        }

        self.logger = logging.getLogger(__name__)

        # Setup data fields
        self.scenarios: list[Scenario] = scenarios
        self.policy_classifiers = {}
        self.policy_costs = {}

        # Split the data into train/validation data for the classifiers
        simulations, dfs = self.__split_data()
        self.train_simulations, self.test_simulations = simulations
        self.train_df, self.test_df = dfs

        # Setup hyperparameters
        self.strategy = str(strategy)
        self.frequency = str(frequency)
        self.worm = str(worm)

        filename = self.worm + "_" + self.strategy + "_" + self.frequency + "_" + regressor_constructors[
            regression_model].__name__ + ".json"
        self.hp_path = Paths.hyperparameter_opt(filename)
        self.constructor = regressor_constructors[regression_model]

        # Setup local iterative search
        self.neighborhoods = neighborhoods

    def manage(self):
        # TODO: figure out whether to use a better search scheme for new policies
        self.logger.info("Start simulation")
        best_cost = float('inf')
        best_policy = None

        curr_policy = create_init_policy(1)
        iteration = 0
        self.policy_costs = {}

        while iteration < self.__N_MAX_ITERS:
            for neighborhood in self.neighborhoods:
                for neighbor in neighborhood(curr_policy):
                    try:
                        # Get the costs for the current policy and update
                        self.__build_regressors(neighbor)

                        # Determine the costs and make sure no invalid data is present
                        costs = self.__calculate_costs(neighbor)
                        costs = {key: val for key, val in costs.items() if not isnan(val)}

                        # Update all costs
                        self.policy_costs = {**self.policy_costs, **costs}

                    except Exception as err:
                        self.logger.error(f"Policy {neighbor} raises an exception: {err}")

            # Update the best policy if an improvement was found
            curr_policy, curr_cost = min(self.policy_costs.items(), key=lambda pair: pair[1])
            if curr_cost < best_cost:
                self.logger.info(f"Policy {curr_policy} is improving! Cost {curr_cost} < {best_cost}")
                best_cost = curr_cost
                best_policy = curr_policy.copy()
                iteration = 0
            else:
                iteration += 1

        return best_policy, self.policy_costs

    def __build_regressors(self, policy: Policy) -> None:
        """
        Build and train the regressor for a given policy and its sub-policies
        :param policy: Policy to build regressors for
        """
        for sub_policy in policy.sub_policies:
            # Already trained the regressor for the given policy
            if sub_policy in self.policy_classifiers:
                continue

            # Otherwise, filter the train/validation data based on the policy and start regressing
            train = self.__filter_data(self.train_df, sub_policy)
            test = self.__filter_data(self.test_df, sub_policy)

            # Build the regressor with previously found hyperparameters if they exist
            found = Writer.get_value_from_json(self.hp_path, str(hash(sub_policy)))
            regressor = self.constructor(sub_policy, train, test)
            regressor.setParameters(found)
            regressor.run()

            # Update best hyperparameters for the regressor
            if not found:
                Writer.update_json_file(self.hp_path, str(hash(sub_policy)), regressor.getParameters())

            # Store the regressor results
            self.policy_classifiers[sub_policy] = regressor

    def __calculate_costs(self, policy: Policy):
        """
        Calculate the cost of a policy and all its sub-policies based on regression under each simulation
        :param policy: Policy to find costs for
        :return: Costs of the policy and its sub-policies
        """
        # Keep track of the costs of all simulations that terminate in a certain policy
        # TODO: figure out if costs are being calculated properly
        policy_simulation_costs: dict[Policy, list] = defaultdict(list)

        for simulation in self.test_simulations:
            for sub_policy in policy.sub_policies:
                classifier = self.policy_classifiers[sub_policy]

                # Continue with epidemiological surveys as long as resistance does not seem to be a problem yet
                epi_signal = classifier.predict(simulation)

                if epi_signal is None:  # cannot use simulations that have incomplete data
                    break
                if epi_signal >= 0.85:  # skip drug efficacy survey when signal is still fine
                    continue

                # Otherwise, verify whether resistance is a problem by scheduling a drug efficacy the year after
                drug_signal = simulation.predict(sub_policy)

                # If no drug efficacy data is available, penalize the policy for not finding a signal sooner
                if drug_signal is None:
                    costs = simulation.calculate_cost(sub_policy)
                    costs += RESISTANCE_NOT_FOUND_COSTS
                    policy_simulation_costs[sub_policy].append(costs)
                    print(
                        f"Simulation {simulation.scenario.id, simulation.id} -> {sub_policy} with costs {costs} [Epi < 0.85, no drug data]")
                    break

                # If data is available and resistance is indeed a problem, stop the simulation and register its cost
                elif drug_signal < 0.85:
                    drug_policy = sub_policy.with_drug_survey()
                    costs = simulation.calculate_cost(sub_policy)
                    print(
                        f"Simulation {simulation.scenario.id, simulation.id} -> {sub_policy} with costs {costs} [Epi < 0.85, drug < 0.85]")
                    policy_simulation_costs[drug_policy].append(costs)
                    break

            # If resistance never becomes a problem under the policy, register its costs without drug efficacy surveys
            else:
                costs = simulation.calculate_cost(policy)
                print(
                    f"Simulation {simulation.scenario.id, simulation.id} -> {policy} with costs {costs} [Epi>= 0.85, drug >= 0.85]")
                policy_simulation_costs[policy].append(costs)

        # Average simulation costs per policy
        policy_costs = {}
        for policy, simulation_costs in policy_simulation_costs.items():
            # Disregard costs that are nan
            costs = [cost for cost in simulation_costs if not isnan(cost)]

            # Calculate average policy costs
            policy_costs[policy] = float('inf') if len(costs) == 0 else sum(costs) / len(costs)

        return policy_costs

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
    from emc.data.neighborhood import flip_neighbors, swap_neighbors

    # TODO: adjust scenario before running the policy manager
    worm = Worm.HOOKWORM.value
    frequency = 1
    strategy = 'sac'

    loader = DataLoader(worm)
    all_scenarios = loader.load_scenarios()

    scenarios = [
        s for s in all_scenarios
        if s.mda_freq == frequency and s.mda_strategy == strategy
    ]

    # Use the policy manager
    print(f"\n\n\n-- {worm}: {strategy} with {frequency} --")
    neighborhoods = [flip_neighbors]  # also swap_neighbors
    manager = PolicyManager(scenarios, strategy, frequency, worm, 0, neighborhoods)

    # Register best policy and save all costs
    best_policy, policy_costs = manager.manage()
    json_costs = {str(policy.epi_time_points): cost for policy, cost in policy_costs.items()}
    path = Paths.data('policies') / f"{worm}{frequency}{strategy}.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w') as file:
        json.dump(json_costs, file, allow_nan=True, indent=4)

    print(best_policy)


if __name__ == '__main__':
    main()
