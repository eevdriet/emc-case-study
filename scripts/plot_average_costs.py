from emc.data.data_loader import DataLoader
from emc.data.constants import *
from emc.log import setup_logger
from emc.util import Paths
from emc.model.policy import Policy

from collections import defaultdict
import json

logger = setup_logger(__name__)


def main():
    # TODO: adjust scenario before running the policy manager
    for worm in Worm:
        worm = worm.value

        loader = DataLoader(worm)
        scenarios = loader.load_scenarios()
        simulations = [simulation for scenario in scenarios for simulation in scenario]

        costs = defaultdict(dict)

        for simulation in simulations:
            logger.info(f"{simulation.id}/{len(simulations)}")
            policy = Policy.from_timepoints([0, 1])
            avg_cost = simulation.calculate_cost(policy)
            costs[simulation][0] = avg_cost

            for year in range(N_YEARS - 1):
                policy = Policy.from_timepoints([year]).with_drug_survey()
                cost = simulation.calculate_cost(policy, allow_average=False)

                costs[simulation][year] = cost

        json_costs = {str(simulation): {str(time): cost} for simulation, costs in costs.items() for time, cost in
                      costs.items()}
        path = Paths.data('policies') / f"{worm}_costs.json"
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'w') as file:
            json.dump(json_costs, file, allow_nan=True, indent=4)

        print("End")


if __name__ == '__main__':
    main()
