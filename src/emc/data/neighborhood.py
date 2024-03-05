from emc.model.policy import Policy
from emc.data.constants import N_YEARS
from typing import Generator, Callable

Neighbor = Generator[Policy, Policy, None]
Neighborhood = Callable[[Policy], Neighbor]


def swap_neighbors(policy: Policy) -> Neighbor:
    yield policy.copy()

    for left in range(1, N_YEARS - 1):  # ignore first and last year when swapping
        for right in range(1, N_YEARS - 1):
            new_policy = policy.copy()

            new_policy[left] = policy[right]
            new_policy[right] = policy[left]

            yield new_policy


def flip_neighbors(policy: Policy) -> Neighbor:
    yield policy.copy()

    for year in range(1, N_YEARS - 1):  # ignore first and last year when flipping
        new_policy = policy.copy()

        new_policy[year] = not policy[year]
        yield new_policy


def flip_out_neighbors(policy: Policy) -> Neighbor:
    yield policy.copy()

    for year in range(1, N_YEARS - 1):
        if policy.epi_surveys[year]:
            new_policy = policy.copy()

            new_policy[year] = False
            yield new_policy


def identity_neighbors(policy: Policy) -> Neighbor:
    for _ in range(1, N_YEARS - 1):
        yield policy.copy()


def fixed_interval_neighbors(policy: Policy) -> Neighbor:
    for interval in range(1, N_YEARS - 1):
        yield Policy.from_every_n_years(interval)


if __name__ == '__main__':
    policy = Policy.from_timepoints([0, 2, 8, 16, 20])
    print(list(p for p in fixed_interval_neighbors(policy)))
