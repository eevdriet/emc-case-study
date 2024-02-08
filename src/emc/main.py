from emc.data import DataLoader


def main():
    loader = DataLoader('ascaris')
    scenarios = loader.load_scenarios()

    print("succes")


if __name__ == "__main__":
    main()
