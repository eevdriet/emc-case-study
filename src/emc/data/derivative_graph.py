from emc.util import Paths
from emc.log import setup_logger

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger(__name__)


def fit_polynomial_with_uncertainty(x, y, y_err, degree):
    # Fit the polynomial with weights and get covariance matrix
    weights = 1 / y_err ** 2
    coeffs, cov_matrix = np.polyfit(x, y, degree, w=weights, cov=True)

    # Polynomial and its derivative
    p = np.poly1d(coeffs)
    p_derivative = p.deriv()
    logger.info(p)
    logger.info(p_derivative)

    return p, p_derivative, cov_matrix


def derivative_variance(x, degree, cov_matrix):
    # Calculate the matrix of derivatives of the polynomial w.r.t its coefficients
    J = np.array([x ** i for i in range(degree, -1, -1)]).T

    # The variance of the derivative is J * Cov * J.T (matrix multiplication)
    var_derivative = np.sum((J @ cov_matrix) * J, axis=1)
    return var_derivative


def main():
    # loader = DataLoader('ascaris')
    # scenarios = loader.load_scenarios()
    # df = loader.monitor_age

    # NOTE: uncomment to generate new levels
    # Levels are currently saved in data, so you can instead retrieve them directly
    # tree = InfectionTree(scenarios, loader.monitor_age)

    path = Paths.data() / 'levels.txt'
    if not path.exists():
        logger.warning("Levels does not exist, generate from InfectionTree")
        return

    with open(path, 'r') as file:
        level_simulations = eval(file.read())

    means, sds, mins, maxs = zip(*level_simulations[3])
    times = np.array(range(len(means)))
    sds = np.array(sds)
    means = np.array(means)
    # logger.info(times)
    # logger.info(means)
    # logger.info(sds)

    degree = 3  # Degree of polynomial

    p, p_derivative, cov_matrix = fit_polynomial_with_uncertainty(times, means, sds, degree)
    derivative_values = p_derivative(times)

    # Calculate the variance of the derivative
    var_derivative = derivative_variance(times, degree, cov_matrix)
    std_derivative = np.sqrt(var_derivative)  # Standard deviation

    upper_bound = derivative_values + std_derivative
    lower_bound = derivative_values - std_derivative

    # Set the Seaborn theme for better aesthetics
    sns.set_theme(style="darkgrid")

    # Plot the derivative line
    plt.plot(times, derivative_values, label='Derivative', color='blue')

    # Add the confidence band
    plt.fill_between(times, lower_bound, upper_bound, color='blue', alpha=0.3, label='Confidence Band')

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Infection level change')
    plt.title('Infection level change per time step. Baseline: 0.3-0.4')
    plt.legend()

    # Show the plot
    plt.ylim(-0.05, 0.05)
    plt.show()


if __name__ == "__main__":
    main()
