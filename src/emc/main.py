from emc.data import DataLoader


def main():
    loader = DataLoader('ascaris')
    scenarios = loader.load_scenarios()
    df = scenarios[0].epidemiological_data

    print("succes")


if __name__ == "__main__":
    main()
