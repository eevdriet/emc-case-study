import pandas as pd
from pathlib import Path
import json


def main() -> None:
    """
    Computes the initial target level for all simulations
    :return: Nothing, just updates the given monitor age CSV with an `target` column
    This column is based on the ERR in the corresponding drug efficacy survey
    Note that if the PC strategy is 2/year, the maximal ERR is chosen for that year
    E.g. for year 13 in the epidemiological survey, years 13.0 and 13.5 are compared
    """

    worm = 'ascaris'
    data_path = Path.cwd() / '..' / 'data'
    path = data_path / f'{worm}_monitor_age_merged.csv'
    monitor_age = pd.read_csv(path)
    drug_efficacy = pd.read_csv(data_path / f'{worm}_drug_efficacy.csv').reset_index(drop=True)

    df = monitor_age.sort_values(['scenario', 'simulation', 'time']).reset_index(drop=True)
    df2 = drug_efficacy.groupby(['scenario', 'simulation'])

    with open(data_path / f'{worm}_metadata.json', 'r') as file:
        metadata = json.load(file)

    n_age_cats = 1 if 'merged' in str(path) else 4
    n_years = 21

    for scen in range(16):
        # Get right levels
        data = metadata[scen]
        mda_freq = data['mda_freq']

        for sim in range(1000):
            start_ma = n_years * n_age_cats * (1000 * scen + sim)
            df3 = df2.get_group((scen + 1, sim + 1)).reset_index(drop=True)

            print(scen, sim)

            for time in range(n_years):
                start_de = mda_freq * time
                target = df3.loc[start_de:start_de + mda_freq, 'ERR'].max()

                for age_cat in range(n_age_cats):
                    df.loc[start_ma + n_age_cats * time + age_cat, 'target'] = target

    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
