import math

class Time:
    """
    Helper static class with the time constants that are used throughout the surveys
    """

    # Average time required from 'Cost' paper in seconds
    KATO_KATZ: dict[str, float] = {
        'demography': 15,
        'single_prep': 67,
        'duplicate_prep': 135,
        'single_record': 9,
        'duplicate_record': 18
    }

    MINI_FLOTAC: dict[str, float] = {
        'demography': 15,
        'single_prep': 131,
        'duplicate_prep': 197,
        'single_record': 9,
        'duplicate_record': 18
    }

    FECPAK_G2: dict[str, float] = {
        'demography': 34,
        'single_prep': 596,
        'duplicate_prep': 1050,
        'single_record': 0,
        'duplicate_record': 0
    }

    @staticmethod
    def __countKK(count: int):
        return pow(10, 2.3896 + 0.0661 * pow(math.log10(count + 1), 2))

    @staticmethod
    def __countMF(count: int):
        return pow(10, 2.5154 + 0.0661 * pow(math.log10(count + 1), 2))

    @staticmethod
    def __countFEC(count: int):
        return pow(10, 1.8349 + 0.1731 * pow(math.log10(count + 1), 2))

