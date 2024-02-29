from emc.model.policy import Policy, create_every_n_years_policy
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


def identity_neighbors(policy: Policy) -> Neighbor:
    for _ in range(1, N_YEARS - 1):
        yield policy.copy()

def fixed_interval_neighbors(policy: Policy) -> Neighbor:
    for i in range(1, N_YEARS - 1):
        yield create_every_n_years_policy(i)

if __name__ == '__main__':
    times = [True] * N_YEARS
    for idx in range(1, N_YEARS - 1):
        times[idx] = False

    policy = Policy(times)
    print(list(p for p in fixed_interval_neighbors(policy)))
