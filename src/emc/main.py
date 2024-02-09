from emc.data import DataLoader
from emc.classifiers import GradientBoostingClassifier, LSTMClassifier
from emc.data import LabelGenerator
from collections import Counter


def main():
    loader = DataLoader('ascaris')
    scenarios = loader.load_scenarios()
    df = scenarios[0]._epi_data

    labelGenerator = LabelGenerator(scenarios)
    labelGenerator.generate()

    # Classifiers bouwen
    lstm = LSTMClassifier()
    model_result, y_test = lstm.run(scenarios[0]._epi_data)
    print(model_result)
    print(y_test)

    print("succes")


if __name__ == "__main__":
    main()
