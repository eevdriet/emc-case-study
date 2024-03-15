import pandas as pd
import numpy as np
from math import isnan
import math

from emc.util import Paths
from emc.data.constants import *
from emc.log import setup_logger
from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs

logger = setup_logger(__name__)


def set_costs() -> None:
    """
    Computes the expected infected level for all simulations
    :return: Nothing, just updates the given CSV with an `exp_inf_level` column
    This column is based on
    - At a given time step (time), what is the expected infected level
    given the infection level at the PREVIOUS step (prev)
    - The expected infected level is based on the levels from the relevant scenario with 5% buckets
    """
    for worm in Worm:
        worm = worm.value

        # Load monitor age data
        path = Paths.worm_data(worm, 'drug_efficacy')
        df = pd.read_csv(path)

        # Assert required columns are present
        req_cols = ["total_useful_tests", "skipped_NaN_tests", "true_a_pre", "true_a_post"]
        assert all(col in df.columns for col in req_cols), "Make sure to merge the drug efficacy data"

        # Add cost column based on the other columns
        df['cost'] = df.apply(calculate_drug_cost, axis=1)

        # Write new dataframe
        df.to_csv(path, index=False)


def calculate_drug_cost(row: pd.Series) -> float:
    """
    Calculate the cost of scheduling a drug efficacy survey in the given year
    :param de_survey: Data to base costs on
    :param year: Year to schedule if any, otherwise take an average over all years
    :return: Cost of scheduling the survey
    """
    # Logging
    scenario = row['scenario']
    simulation = row['simulation']
    logger.info(f"{scenario} {simulation}")

    # Parameters
    total_useful_tests = row['total_useful_tests']
    skipped_NaN_tests = row['skipped_NaN_tests']
    true_a_pre = row['true_a_pre']
    true_a_post = row['true_a_post']

    if any(isnan(var) for var in (total_useful_tests, skipped_NaN_tests, true_a_pre, true_a_post)):
        return np.nan

    costs = 0
    costs += __consumable(total_useful_tests, skipped_NaN_tests)

    days = __days(total_useful_tests, skipped_NaN_tests, true_a_pre, true_a_post)
    costs += __personnel(days)
    costs += __transportation(days)

    return costs


def __consumable(total_useful_tests: int, skipped_NaN_tests: int) -> float:
    """
    Calculate the consumable costs
    :param de_survey: Survey data to base costs on
    :param year: Year to schedule if any, otherwise take an average over all years
    :return: Consumable costs
    """
    # TODO: handle missing observations before calculating costs to avoid this error prevention below

    # Calculate costs
    N_baseline = total_useful_tests + skipped_NaN_tests
    N_follow_up = total_useful_tests
    baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
    follow_up_costs = N_follow_up * (
            Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
    return baseline_costs + follow_up_costs


def __personnel(days: int) -> float:
    """
    Calculate the personnel costs of a drug efficacy survey
    :param de_survey: Survey data to base costs on
    :param year: Year to schedule if any, otherwise take an average over all years
    :return: Personnel costs
    """
    return 4 * 22.5 * days


def __transportation(days: int) -> int:
    """
    Calculate the transportation costs of a drug efficacy survey
    :param de_survey: Survey data to base costs on
    :param year: Year to schedule if any, otherwise take an average over all years
    :return: Transportation costs
    """
    return 90 * days


def __days(total_useful_tests: int, skipped_NaN_tests: int, true_a_pre: float, true_a_post: float) -> int:
    """
    Calculate the number of days required to take a drug efficacy survey
    :param de_survey: Survey data to base the calculation on
    :param year: Year to schedule if any, otherwise take an average over all years
    :return: Survey days
    """
    # Set parameters
    workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
    timeAvailable = workers * 4 * 60 * 60  # In seconds

    # Calculate costs
    N_baseline = total_useful_tests + skipped_NaN_tests
    N_follow_up = total_useful_tests
    counting_time = 0
    for i in range(N_baseline):
        c_pre = 680
        counting_time += Time_Costs.countKK(c_pre)

    for i in range(N_follow_up):
        c_post = 400
        counting_time +=
    c_pre = true_a_pre  # TODO: Use average egg observations per time stamp AND include duplicate KK
    c_post = true_a_post  # TODO: This is true number of eggs in individual, aliquots is on observed

    count_pre = Time_Costs.countKK(c_pre)
    count_post = Time_Costs.countKK(2 * c_post)
    time_pre = N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                             Time_Costs.KATO_KATZ.get('single_record')) + count_pre
    time_post = N_follow_up * (Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                               Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post
    return math.ceil((time_pre + time_post) / timeAvailable)


if __name__ == '__main__':
    set_costs()
