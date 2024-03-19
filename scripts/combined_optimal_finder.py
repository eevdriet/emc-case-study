from emc.data.constants import *
from emc.model.score import ScoreType
from emc.util import Paths, Writer


def find_combined_optimal():
    frequencies = [1, 2]
    strategies = ['community', 'sac']
    score_types = [ScoreType.TOTAL_COSTS, ScoreType.FINANCIAL_COSTS, ScoreType.RESPONSIVENESS]

    for strategy in strategies:
        for frequency in frequencies:
            for score_type in score_types:
                path1 = Paths.data(
                    'policies') / f"{Worm.HOOKWORM.value}{frequency}{strategy}" / f"{score_type.value}_fixed_interval.json"
                data1 = Writer.read_json_file(path1)
                path2 = Paths.data(
                    'policies') / f"{Worm.ASCARIS.value}{frequency}{strategy}" / f"{score_type.value}_fixed_interval.json"
                data2 = Writer.read_json_file(path2)

                # Calculate the sum of total_costs for each key in both JSON inputs
                total_costs_sum = {}

                for key, value in data1.items():
                    total_costs_sum[key] = value["total_costs"]

                for key, value in data2.items():
                    if key in total_costs_sum:
                        total_costs_sum[key] += value["total_costs"]
                    else:
                        total_costs_sum[key] = value["total_costs"]

                # Find the key with the lowest total cost
                min_key = min(total_costs_sum, key=total_costs_sum.get)

                print(f"For freq: {frequency}, strat: {strategy}, score_type: {score_type.value}")
                print(f"Key with lowest total cost: {min_key}")


if __name__ == '__main__':
    find_combined_optimal()
