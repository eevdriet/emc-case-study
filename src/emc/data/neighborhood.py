from emc.model.policy import Policy
from emc.data.constants import N_YEARS
from typing import Generator, Callable

Neighbor = Generator[Policy, Policy, None]
Neighborhood = Callable[[Policy], Neighbor]


def swap_neighbors(policy: Policy) -> Neighbor:
    yield policy.copy()

    for left in range(N_YEARS):
        for right in range(N_YEARS):
            new_policy = policy.copy()

            new_policy[left] = policy[right]
            new_policy[right] = policy[left]

            yield new_policy


def flip_neighbors(policy: Policy) -> Neighbor:
    yield policy.copy()

    for year in range(N_YEARS):
        new_policy = policy.copy()
        new_policy[year] = not policy[year]

        yield new_policy
