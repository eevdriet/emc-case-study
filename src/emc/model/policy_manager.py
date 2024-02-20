import pandas as pd
import random

from emc.model.policy import Policy
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.data.constants import *

from emc.classifiers.single_gradient_booster import SingleGradientBooster


class PolicyManager:
    TRAIN_TEST_SPLIT_SIZE: float = 0.2
    TRAIN_VAL_SPLIT_SIZE: float = 0.25

    """
    Manages the classification of different policies and their sub-policies
    """

    def __init__(self, scenarios: list[Scenario]):
        self.scenarios: list[Scenario] = scenarios
        self.policy_classifiers = {}

        self.train_simulations = []
        self.train_df = pd.DataFrame()

        self.test_simulations = []
        self.test_df = pd.DataFrame()

    def manage(self):
        # Split the data into train/validation data for the classifiers
        self.train_simulations, self.train_df, self.test_simulations, self.test_df = self.__split_data()

        #
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
            self.policy_classifiers[sub_policy] = classifier

            for simulation in self.test_simulations:

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
        self.scenarios = self.scenarios
        epi_surveys = (True,) * N_YEARS

        return Policy(epi_surveys)

    def __split_data(self):
        # Combine all simulations from all scenarios
        simulations: list[Simulation] = []
        for scenario in self.scenarios:
            simulations += scenario.simulations

        # Randomly order the simulations and split into train/validation
        random.seed(SEED)
        random.shuffle(simulations)
        split_idx = int(len(simulations) * self.TRAIN_TEST_SPLIT_SIZE)
        #
        print("Splitting...")
        train_sims = simulations[split_idx:]
        test_sims = simulations[:split_idx]

        print("Merging...")
        train_df = pd.concat([sim.monitor_age for sim in train_sims])
        test_df = pd.concat([sim.monitor_age for sim in test_sims])

        print("Created train/test")
        return train_sims, train_df, test_sims, test_df

    def __filter_data(self, df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
        time_points = policy.time_points
        df = df[df['time'].isin(time_points)]

        return df


def main():
    from emc.data.data_loader import DataLoader

    # Get the data
    worm = Worm.ASCARIS
    loader = DataLoader(worm.value)
    scenarios = loader.load_scenarios()

    # Use the policy manager
    manager = PolicyManager(scenarios)
    manager.manage()


if __name__ == '__main__':
    main()
