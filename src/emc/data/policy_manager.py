import json
from pathlib import Path

import pandas as pd
import random
from collections import defaultdict
from emc.util import Paths
from emc.log import setup_logger
from math import isnan
import logging

logger = setup_logger(__name__)

from emc.model.policy import Policy
from emc.model.scenario import Scenario
from emc.model.simulation import Simulation
from emc.data.constants import *
from emc.model.score import Score
from emc.util import Writer, Paths

from emc.regressors import *
from emc.data.neighborhood import Neighborhood
from emc.util import normalised, Pair


class PolicyManager:
    # Data is split into pairs of train/test data
    # The data consists of the simulations and their (combined) data as data frame
    SplitData = tuple[Pair[list[Simulation]], Pair[pd.DataFrame]]

    """
    Manages the classification of different policies and their sub-policies
    """

    __N_MAX_ITERS: int = 10
    __TRAIN_TEST_SPLIT_SIZE: float = 0.2
    __NORMALISED_COLS = {'n_host', 'n_host_eggpos', 'a_epg_obs'}

    def __init__(self, scenarios: list[Scenario], strategy: str, frequency: int, worm: str, regression_model: regressor,
                 neighborhoods: list[Neighborhood], init_policy: Policy):
        self.logger = logging.getLogger(__name__)

        # Setup data fields
        self.scenarios: list[Scenario] = scenarios
        self.policy_classifiers = {}
        self.policy_scores: dict[Policy, float] = {}
        self.sub_policy_simulations: dict[Policy, set[tuple[int, int]]] = defaultdict(set)

        # Split the data into train/validation data for the classifiers
        simulations, dfs = self.__split_data()
        self.train_simulations, self.test_simulations = simulations
        self.train_df, self.test_df = dfs

        # Setup hyperparameters
        self.strategy = strategy
        self.frequency = frequency
        self.worm = worm
        self.constructor = regression_model

        # Setup local iterative search
        self.neighborhoods = neighborhoods

        # Setup first policy
        self.init_policy = init_policy

    def manage(self):
        # TODO: figure out whether to use a better search scheme for new policies

        # Setup iteration variables
        logger.info("Start iterated local search")
        self.policy_scores = {}
        scores = {}

        best_score = Score.create_missing()
        iteration = 0
        total_iteration = 0

        # Setup policy
        curr_policy = self.init_policy
        curr_score = float('inf')

        ils_best_policy = self.init_policy
        ils_best_score = float('inf')

        while iteration < self.__N_MAX_ITERS:
            logger.info(f"ILS Iteration {iteration}/{self.__N_MAX_ITERS} (total: {total_iteration}) \n")
            total_iteration += 1

            while True:
                logger.info(f"Greedy Policy Improvement")

                neighbor_scores = {}

                for neighborhood in self.neighborhoods:
                    neighbors: list[Policy] = list(neighborhood(curr_policy))

                    for it, neighbor in enumerate(neighbors, 1):
                        if neighbor in scores:
                            score = scores[neighbor]
                            logger.debug(f"{score.policy}\n- Using previous score : {float(scores[neighbor])}")
                            neighbor_scores[neighbor] = score
                            continue

                        self.__build_regressors(neighbor)

                        # Determine the score and make sure no invalid data is present
                        score = self.__calculate_score(neighbor)
                        logger.info(score)
                        neighbor_scores[neighbor] = score
                        scores[neighbor] = score

                # Register all policy score
                self.policy_scores = {**self.policy_scores, **neighbor_scores}

                # Update the best policy if an improvement was found
                curr_policy, curr_score = min(neighbor_scores.items(), key=lambda pair: pair[1])
                if curr_score < best_score:
                    logger.debug(f"Greedy optimisation: {curr_policy} is improving! Score {float(curr_score)}\n")
                    best_score = curr_score
                    best_policy = curr_policy.copy()
                else:
                    logger.debug(f"Greedy optimisation: No policy improvement found, stopping greedy")
                    break
            
            if (best_score < ils_best_score):
                ils_best_score = best_score
                ils_best_policy = best_policy

                iteration = 0

                logger.info(f"Improvement found {iteration}/{self.__N_MAX_ITERS} (total: {total_iteration})")
                logger.info(f"Current best policy: {ils_best_policy.epi_time_points}, (cost: {ils_best_score})")
                curr_policy = ils_best_policy.perturbe()
                logger.info(f"New perturbed policy: {curr_policy.epi_time_points}")

            else:
                iteration += 1

                logger.info(f"No improvement found in perturbed policy {iteration}/{self.__N_MAX_ITERS} (total: {total_iteration})")
                logger.info(f"Current best policy: {ils_best_policy.epi_time_points}, (cost: {ils_best_score})")
                curr_policy = ils_best_policy.perturbe()
                logger.info(f"New perturbed policy: {curr_policy.epi_time_points}")

        return best_score, self.policy_scores

    def __build_regressors(self, policy: Policy) -> None:
        """
        Build and train regressors for a given policy and its sub-policies.
        
        This function iterates over sub-policies of a given policy, checks if a regressor has already been trained,
        and if not, it proceeds to train a new one. It handles the creation and training of regressors,
        managing of models and preprocessing data, and updating hyperparameters as needed.

        :param policy: Policy object for which the regressors need to be built and trained.
        """
        for sub_policy in policy.sub_policies:
            # Skip training if the regressor for this sub-policy already exists.
            if sub_policy in self.policy_classifiers:
                continue

            # Filter training and testing data specific to the current sub-policy.
            train = self.__filter_data(self.train_df, sub_policy)
            test = self.__filter_data(self.test_df, sub_policy)

            # Define paths for saving and loading the model and preprocessing data.
            model_path = Paths.models(self.worm, self.frequency, self.strategy, self.constructor.__name__,
                                      str(sub_policy.epi_time_points) + ".pkl")
            prepro_path = Paths.preprocessing(self.worm, self.frequency, self.strategy,
                                              str(sub_policy.epi_time_points))
            model = Writer.loadPickle(model_path)

            if model is None:
                # If no existing model is found, create a new model and preprocessing data.
                logger.debug(f'Creating new model for: Policy({sub_policy.epi_time_points})')

                features_data = Writer.loadPickle(prepro_path / "features_data.pkl")
                targets_data = Writer.loadPickle(prepro_path / "targets_data.pkl")
                features_test = Writer.loadPickle(prepro_path / "features_test.pkl")
                targets_test = Writer.loadPickle(prepro_path / "targets_test.pkl")

                # Check for existing hyperparameters and initialize the regressor.
                regressor = self.constructor(sub_policy, train, test)

                if features_data is not None or targets_data is not None or features_test is not None or targets_test is not None:
                    logger.debug("Using already calculated preprocessing")
                    regressor.setPreprocessing(features_data, targets_data, features_test, targets_test)

                regressor.initialize_and_train_model()

                # Save preprocessing data.
                (features_data, targets_data, features_test, targets_test) = regressor.getPreprocessing()
                Writer.savePickle(prepro_path / "features_data.pkl", features_data)
                Writer.savePickle(prepro_path / "targets_data.pkl", targets_data)
                Writer.savePickle(prepro_path / "features_test.pkl", features_test)
                Writer.savePickle(prepro_path / "targets_test.pkl", targets_test)
                Writer.savePickle(model_path, regressor.getModel())
            else:
                # Use the existing model if found.
                logger.debug(f'Using created model for: Policy({sub_policy.epi_time_points})')
                regressor = self.constructor.createInstance(self.constructor, model, sub_policy, train, test)

                # Load preprocessing data.
                features_data = Writer.loadPickle(prepro_path / "features_data.pkl")
                targets_data = Writer.loadPickle(prepro_path / "targets_data.pkl")
                features_test = Writer.loadPickle(prepro_path / "features_test.pkl")
                targets_test = Writer.loadPickle(prepro_path / "targets_test.pkl")

                # Use existing preprocessing data if available, otherwise, calculate new preprocessing data.
                if features_data is not None or targets_data is not None or features_test is not None or targets_test is not None:
                    logger.debug("Using already calculated preprocessing")
                    regressor.setPreprocessing(features_data, targets_data, features_test, targets_test)
                else:
                    logger.debug("Calculating new preprocessing")
                    regressor.initialize_and_train_model()
                    (features_data, targets_data, features_test, targets_test) = regressor.getPreprocessing()
                    Writer.savePickle(prepro_path / "features_data.pkl", features_data)
                    Writer.savePickle(prepro_path / "targets_data.pkl", targets_data)
                    Writer.savePickle(prepro_path / "features_test.pkl", features_test)
                    Writer.savePickle(prepro_path / "targets_test.pkl", targets_test)

            # Store the trained regressor in the policy classifiers dictionary.
            self.policy_classifiers[sub_policy] = regressor

    def __calculate_score(self, policy: Policy) -> Score:
        """
        Calculate the cost of a policy and all its sub-policies based on regression under each simulation
        :param policy: Policy to find costs for
        :return: Costs of the policy and its sub-policies
        """
        # Keep track of the costs of all simulations that terminate in a certain policy
        # TODO: figure out if costs are being calculated properly

        # Go through all sub-policies and ignore the empty policy
        sub_policies = [p for p in policy.sub_policies]
        sub_policy_costs: dict[Policy, dict[tuple[int, int], float]] = defaultdict(dict)

        if len(sub_policies) == 0:
            return Score.create_missing()

        latenesses = []
        n_false_positives = 0
        n_false_negatives = 0

        for iter, simulation in enumerate(self.test_simulations):
            key = (simulation.scenario.id, simulation.id)
            logger.debug(f"{iter}/{len(self.test_simulations)} with simulation {key}")

            signal_year = None

            for sub_policy in sub_policies:
                classifier = self.policy_classifiers[sub_policy]

                # Continue with epidemiological surveys as long as resistance does not seem to be a problem yet
                epi_signal = classifier.predict(simulation)

                if epi_signal is None:  # cannot use simulations that have incomplete data
                    costs = simulation.calculate_cost(sub_policy)
                    # costs += RESISTANCE_NOT_FOUND_COSTS
                    # policy_simulation_costs[sub_policy].append(costs)
                    logger.debug(f"Simulation {key} -> {sub_policy} with costs {costs} [No epi data]")

                    sub_policy_costs[sub_policy][key] = costs
                    break

                if epi_signal >= DRUG_EFFICACY_THRESHOLD:  # skip drug efficacy survey when signal is still fine
                    continue

                # Otherwise, verify whether resistance is a problem by scheduling a drug efficacy the year after
                drug_signal = simulation.verify(sub_policy)

                # If no drug efficacy data is available, penalize the policy for not finding a signal sooner
                if drug_signal is None:
                    costs = simulation.calculate_cost(sub_policy)
                    # costs += RESISTANCE_NOT_FOUND_COSTS
                    # policy_simulation_costs[sub_policy].append(costs)
                    logger.debug(
                        f"Simulation {key} -> {sub_policy} with costs {costs} [Epi < {DRUG_EFFICACY_THRESHOLD}, no drug data]")

                    sub_policy_costs[sub_policy][key] = costs
                    break

                # If data is available and resistance is indeed a problem, stop the simulation and register its cost
                elif drug_signal < 0.85:
                    drug_policy = sub_policy.with_drug_survey()
                    costs = simulation.calculate_cost(drug_policy)
                    logger.debug(
                        f"Simulation {key} -> {drug_policy} with costs {costs} [Epi < {DRUG_EFFICACY_THRESHOLD}, drug < 0.85]")
                    # policy_simulation_costs[drug_policy].append(costs)
                    sub_policy_costs[sub_policy][key] = costs
                    signal_year = drug_policy.last_year + 1
                    break

            # If resistance never becomes a problem under the policy, register its costs without drug efficacy surveys
            else:
                costs = simulation.calculate_cost(policy)
                logger.debug(
                    f"Simulation {key} -> {policy} with costs {costs} [Epi>= {DRUG_EFFICACY_THRESHOLD}, drug >= 0.85]")
                sub_policy_costs[policy][key] = costs
                self.sub_policy_simulations[policy].add(key)

            # Only penalise miss classifications for resistance modes different from 'none' when needed
            if simulation.scenario.res_mode != 'none':
                # Find the first year for which the ERR value is not nan
                first_year = None
                ERR = simulation.drug_efficacy_s['ERR'].reset_index(drop=True)
                target = simulation.monitor_age['target'].reset_index(drop=True)
                for time, (epi_signal, drug_signal) in enumerate(zip(target, ERR)):
                    if isnan(epi_signal) or isnan(drug_signal):
                        break

                    if epi_signal >= DRUG_EFFICACY_THRESHOLD or drug_signal >= 0.85:
                        continue

                    first_year = time
                    break

                # Determine lateness of the policy
                # Note that the absolute value is needed, regressor can find a signal before it occurs in the monitor_age data
                if first_year is None:
                    lateness = 0
                else:
                    lateness = (20 if signal_year is None else signal_year) - first_year
                    lateness = abs(lateness)

                logger.debug(f"Simulation {key} has lateness {lateness}")
                latenesses.append(lateness)

                # Verify whether the simulation was wrongly classified
                
                # Find the last year for which the true drug_efficacy/ERR values are below
                last_year = None
                
                for time, (epi_signal, drug_signal) in enumerate(zip(target[::-1], ERR[::-1])):
                    if isnan(epi_signal) or isnan(drug_signal):
                        continue

                    if epi_signal >= DRUG_EFFICACY_THRESHOLD or drug_signal >= 0.85:
                        continue

                    last_year = time
                    break

                if last_year is not None and signal_year is None:
                    logger.debug(f"Simulation {key} was wrongly classified as False Negative")
                    n_false_negatives += 1
                elif last_year is None and signal_year is not None:
                    logger.debug(f"Simulation {key} was wrongly classified as False Positive")
                    n_false_positives += 1

        # Calculate final costs and display
        return Score(policy=policy,
                     n_simulations=len(self.test_simulations),
                     n_false_positives=n_false_positives,
                     n_false_negatives=n_false_negatives,
                     responses=latenesses,
                     sub_policy_costs=sub_policy_costs)

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

        logger.debug("Created train/test")
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
    from emc.data.neighborhood import flip_neighbors, swap_neighbors, identity_neighbors, fixed_interval_neighbors, flip_out_neighbors

    # TODO: adjust scenario before running the policy manager
    worm = Worm.HOOKWORM.value
    frequency = 2
    strategy = 'sac'
    regresModel = GradientBoosterDefault

    # Use the policy manager
    logger.info(f"-- {worm}: {strategy} with {frequency} --")
    neighborhoods = [fixed_interval_neighbors]  # also swap_neighbors

    loader = DataLoader(worm)
    all_scenarios = loader.load_scenarios()

    scenarios = [
        s for s in all_scenarios
        if s.mda_freq == frequency and s.mda_strategy == strategy
    ]

    init_policy = Policy.from_every_n_years(5)
    manager = PolicyManager(scenarios, strategy, frequency, worm, regresModel, neighborhoods, init_policy)

    # Register best policy and save all costs
    best_score, policy_scores = manager.manage()
    json_costs = {str(policy.epi_time_points): score.as_dict() for policy, score in policy_scores.items()}
    path = Paths.data('policies') / f"{worm}{frequency}{strategy}.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w') as file:
        json.dump(json_costs, file, allow_nan=True, indent=4)

    policy, val = best_score.policy, float(best_score)
    logger.info(f"Optimal policy is {policy} with score {val}")


if __name__ == '__main__':
    main()
