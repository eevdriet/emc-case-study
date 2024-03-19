from emc.util import *

worms = ['hookworm', 'ascaris']
frequencies = ['1', '2']
strategies = ['sac', 'community']
# methods = ['financial_costs', 'total_costs']
# methods = ['responsiveness']
methods =['responsiveness']
fixed_intervals = ['']

lengths = []

for worm in worms:
    for frequency in frequencies:
        for strategy in strategies:
            for method in methods:
                for fixed_interval in fixed_intervals:
                    json_path = Paths.data('policies') / f"{worm}{frequency}{strategy}" / f"{method}{fixed_interval}.json"
                    json_data = Writer.read_json_file(json_path)
                    print(f"{worm} {frequency} {strategy} {method} {fixed_interval}")
                    lengths.append(len(json_data))
                    print(len(json_data))

print(sum(lengths) / len(lengths))