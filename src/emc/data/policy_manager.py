import pandas as pd
import random

from emc.model.policy import Policy
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.data.constants import *
from emc.data.neighborhood_descent import NeighborhoodDescent

from emc.classifiers.single_gradient_booster import SingleGradientBooster
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

    def __init__(self, scenarios: list[Scenario]):
        self.scenarios: list[Scenario] = scenarios
        self.policy_classifiers = {}

        self.train_simulations = []
        self.test_simulations = []

        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.neighborhood_descent = NeighborhoodDescent()

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

            classifier = SingleGradientBooster(sub_policy, train, test)
            classifier.run()

            # Store the classifier results
            # self.policy_classifiers[sub_policy] = classifier

            # for simulation, df in zip(self.test_simulations, self.test_df):
            #     res = classifier.test(df)
            #     cost = simulation.calculate_cost(policy)

            # for
            # totalCosts = 0
            # per simulation in test set:
            #   from generated classifiers construct de_survey schedule
            #   for every time t from t = 0 to t = n
            #       if epi_survey[t]:
            #           totalCosts += epi_survey_costs at time t
            #       if de_survey [t]:
            #           totalCosts += de_survey_costs at time t
            #           if de_survey result < 0.85:
            #               continue
            #   if (de_efficacy < 0.85 but not found):
            #       totalCosts += 100000 (costs if not found)
            # averageCosts = totalCosts / len(simulation in test_set)
            # TODO: hoe goed is de policy = averageCosts

    def __create_init_policy(self) -> Policy:
        """
        Create an initial policy to start the policy improvement from
        :return: Initial policy
        """
        self.scenarios = self.scenarios
        epi_surveys = (True, False,) * (N_YEARS // 2) + (True,)

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
        time_points = policy.time_points
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
                manager = PolicyManager(scenarios)
                manager.manage()


if __name__ == '__main__':
    main()
