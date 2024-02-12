from emc.data import DataLoader
from emc.classifiers import GradientBoosting
from emc.data import LabelGenerator
from collections import Counter


def main():
    loader = DataLoader('ascaris')
    scenarios = loader.load_scenarios()
    df = scenarios[0]._epi_data
    print(df)

    labelGenerator = LabelGenerator(scenarios)
    labelGenerator.generate()

    # Classifiers bouwen
    gb = GradientBoosting()
    result = gb.run(scenarios[0]._epi_data)
    print(Counter(result))

    print("succes")

if __name__ == "__main__":
    main()
