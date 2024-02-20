import pandas as pd

from emc.data.constants import *
from emc.util import Paths


def save_pretty() -> None:
    """
    Reformat the CSV files to have the right column order and integer columns where needed
    :return: Nothing, just reformat existing CSV files
    """

    for worm in Worm:
        worm = worm.value

        for data_type in ['monitor_age', 'drug_efficacy']:
            print(f"Reformatting the {worm} {data_type} data...")
            path = Paths.worm_data(worm, data_type, use_merged=True)
            df = pd.read_csv(path).reset_index(drop=True)

            cols = DE_COLUMNS if data_type == 'drug_efficacy' else MA_COLUMNS
            int_cols = DE_INT_COLUMNS if data_type == 'drug_efficacy' else MA_INT_COLUMNS

            df = df[cols]
            df[int_cols] = df[int_cols].astype('Int64', errors="ignore")
            df.to_csv(path, index=False)


if __name__ == '__main__':
    save_pretty()
