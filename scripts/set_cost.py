import pandas as pd

from emc.util import Paths
from emc.data.constants import *
from emc.log import setup_logger
from emc.data.cost_calculator import CostCalculator, CostTechnique

logger = setup_logger(__name__)


def set_costs():
    for worm in Worm:
        worm = worm.value

        path = Paths.worm_data(worm, 'drug_efficacy')
        df = pd.read_csv(path)

        # Calculate the costs for different techniques
        # NOTE: set to only hosts as average technique is invalid
        for technique in [CostTechnique.FROM_INDIVIDUAL_HOSTS]:
            out_path = path.with_stem(f'{worm}_{technique.value}')
            if out_path.exists():
                logger.info(f"Skippped {worm}")
                continue

            # Average costs
            calculator = CostCalculator(worm, technique=technique)
            df = calculator.calculate_from_df(df)
            df.to_csv(out_path, index=False)


if __name__ == '__main__':
    set_costs()
