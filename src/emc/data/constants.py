from enum import Enum


class Worm(Enum):
    ASCARIS = 'ascaris'
    HOOKWORM = 'hookworm'

    def __str__(self):
        return self.name


"""
Numerical constants
"""

N_SCENARIOS = 16
N_SIMULATIONS = 1000
N_YEARS = 21
N_AGE_CATEGORIES = 4

"""
Other constants
"""

AGE_CATEGORIES = [15, 5, 2, 0]

"""
Columns
"""

COLUMNS = ['scenario', 'simulation', 'time']
INT_COLUMNS = ['scenario', 'simulation']

# Monitor age
MA_COLUMNS = COLUMNS + ['n_host',
                        'n_host_eggpos',
                        'exp_inf_level',
                        'inf_level',
                        'a_epg_obs',
                        'a_drug_efficacy_true',
                        'ERR',
                        'target']

MA_INT_COLUMNS = INT_COLUMNS + ['time', 'n_host', 'n_host_eggpos']

# Drug efficacy
DE_COLUMNS = COLUMNS + ['total_useful_tests', 'skipped_NaN_tests',
                        'true_a_pre', 'true_a_post', 'true_total_pre', 'true_total_post',
                        'ERR', 'EPG_change']
DE_INT_COLUMNS = INT_COLUMNS + ['total_useful_tests', 'skipped_NaN_tests']

# Levels
BUCKET_SIZES = [5, 10, 20]
MDA_FREQUENCIES = [1, 2]
MDA_STRATEGIES = ['sac', 'community']

# Other
SEED = 76

RESISTANCE_MODES = ["none", "dominant", "codominant", "recessive"]
RESISTANCE_NOT_FOUND_COSTS = 1_000
ACCURACY_VIOLATED_COSTS = 1_000
ACCURACY_VIOLATED_THRESHOLD = 0.80
DRUG_EFFICACY_THRESHOLD = 0.85
MAX_MISCLASSIFICATION_FRACTION = 1
MC_EVALUATION_NUM = 50

# Color scheme
YELLOW = '#FFC107'
MAGENTA = '#D81B60'
BLUE = '#1E88E5'
GREEN = '#004D40'
VIOLET = '#7B1FA2'
ORANGE = '#FF5722'
