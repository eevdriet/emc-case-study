class Costs:
    """
    Helper static class with the cost constants that are used throughout the surveys
    """

    # Fixed costs
    EQUIPMENT: float = 0.57
    FIXED_COST: float = 0.60

    # Variable costs
    KATO_KATZ: dict[str, float] = {
        'single': 0.78,
        'duplicate': 0.91
    }

    MINI_FLOTAC: dict[str, float] = {
        'single': 0.92,
        'duplicate': 1.27
    }

    FECPAK_G2: dict[str, float] = {
        'single': 1.09
        'duplicate': 2.13
    }
