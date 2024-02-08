from attrs import define


@define
class Policy:
    # At which moments in time to conduct a survey or not
    surveys: list[bool] = [False] * 21
