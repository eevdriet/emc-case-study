from collections import OrderedDict

from add_features import add_features
from emc.data.level_builder import build_levels
from emc.log import setup_logger
from merge_age_cats import merge_age_cats
from merge_monitor_age_csv import merge_csv
from save_pretty import save_pretty
from set_expected_inf_level import set_expected_infection_level
from set_init_target import set_target

logger = setup_logger(__name__)


class Pipeline:
    """
    Executes the various scripts to load the data into one CSV file and add interesting features
    Note that when
    """

    __NAMED_PROCESSES = OrderedDict([
        # Data set creation
        ("merge_csv", merge_csv),
        ("merge_age_cats", merge_age_cats),

        # Feature generation
        ("add_features", add_features),
        ("build_levels", lambda: build_levels(overwrite=True)),
        ("add_exp_level", set_expected_infection_level),
        ("add_target", set_target),

        # Data set saving
        ("save_pretty", save_pretty)
    ])

    def __init__(self, processes: list[str] = __NAMED_PROCESSES.keys()):
        assert all(p in self.__NAMED_PROCESSES for p in processes), "Provide only valid processes"

        # Sort the processes in the right order
        keys = list(self.__NAMED_PROCESSES.keys())
        self.names = sorted(processes, key=lambda name: keys.index(name))

    def run(self):
        for name in self.names:
            # Information banner
            header = f"\n\n\n{'-' * 20} {name} {'-' * 20}"
            logger.info(header)

            # Execute named process
            process = self.__NAMED_PROCESSES[name]
            process()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
