class Costs:
    """
    Helper static class with the cost constants that are used throughout the surveys
    """

    # Fixed costs
    EQUIPMENT: float = 0.57
    FIXED_COST: float = 0.60

    # Variable costs
    KATO_KATZ: dict[str, float] = {
        'aliquot': 1.37,  # from 'Cost' paper
        'single_sample': 0.78,  # from 'Erasmus MC_cost data'
        'duplicate_sample': 0.91
    }

    MINI_FLOTAC: dict[str, float] = {
        'aliquot': 1.51,
        'single_sample': 0.92,
        'duplicate_sample': 1.27
    }

    FECPAK_G2: dict[str, float] = {
        'aliquot': 1.69,
        'single_sample': 1.09,
        'duplicate_sample': 2.13
    }
