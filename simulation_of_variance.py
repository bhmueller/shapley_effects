"""Capabilities for computation of model variances.

This module contains functions to simulate the variance of models.

"""
import chaospy as cp
import numpy as np
#import warnings


def simulate_variance(model, cov, mean, n_sim):
    """
    Calculate the variance of a model.

    Args:
        model (string): Model for which to calculate the variance.
        cov (ndarray): Covariance matrix of the random variables.
        mean (array): Vector of means.
        n_sim (int): Number of samples for Monte Carlo simulation.

    Returns:
        float: The variance of the model.
    """

    n_inputs = len(mean)

    # Get the function for sampling.
    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    # Initialize model inputs for evaluating the model.
    model_inputs = np.zeros(
        (n_sim, n_inputs),
    )
    np.random.seed(123)
    model_inputs[:n_sim, :] = x_all(n_sim).T

    # Calculate the output Y.
    output = model(model_inputs)
    # if output.shape != (n_sim,):
    #   raise ValueError
    return np.var(output)