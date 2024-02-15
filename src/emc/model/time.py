class Costs:
    """
    Helper static class with the time constants that are used throughout the surveys
    """

    # Average time required from 'Cost' paper in seconds
    KATO_KATZ: dict[str, float] = {
        'demography': 15,
        'single_sample': 67,
        'duplicate_sample': 135,
    }

    MINI_FLOTAC: dict[str, float] = {
        'demography': 15,
        'single_sample': 131,
        'duplicate_sample': 197
    }

    FECPAK_G2: dict[str, float] = {
        'demography': 34,
        'single_sample': 596,
        'duplicate_sample': 1050
    }
