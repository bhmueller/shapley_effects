import chaospy as cp
import numpy as np
import pandas as pd
from functools import partial
from numpy.testing import assert_array_almost_equal as aaae

from econsa.shapley import _r_condmvn
from econsa.shapley import get_shapley


def test_get_shapley_linear_two_inputs():
    """Tests for the module shapley.py for the test case in section 3.2 in Iooss and 
        Prieur (2019): Linear model with two Gaussian inputs."""
    def linear_model(x):
        beta = np.array([[beta_1], [beta_2]])
        return x.dot(beta)

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
            distribution = cp.MvNormal(mean[subset_j], cov_int)
            return distribution.sample(n)
        else:
            return _r_condmvn(
                n,
                mean=mean,
                cov=cov,
                dependent_ind=subset_j,
                given_ind=subsetj_conditional,
                x_given=xjc,
            )

    np.random.seed(1234)
    beta_1 = 1.3
    beta_2 = 1.5
    var_1 = 16
    var_2 = 4
    rho = 0.3

    # Calculate analytical Shapley effects.
    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    covariance = rho * np.sqrt(var_1) * np.sqrt(var_2)
    var_y = component_1 + 2 * covariance * beta_1 * beta_2 + component_2
    share = 0.5 * (rho**2)
    true_shapley_1 = (component_1 * (1 - share) + covariance * beta_1 * beta_2 + component_2 * share)/var_y
    true_shapley_2 = (component_2 * (1 - share) + covariance * beta_1 * beta_2 + component_1 * share)/var_y

    n_inputs = 2
    mean = np.zeros(n_inputs)
    cov = np.array(
        [[var_1, covariance],
        [covariance, var_2]]
        )
    method = "exact"
    n_perms = None
    n_output = 10 ** 7
    n_outer = 10 ** 5
    n_inner = 10 ** 2

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [true_shapley_1, true_shapley_2],
            [0, 0],
            [true_shapley_1, true_shapley_2],
            [true_shapley_1, true_shapley_2],
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        linear_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated["Shapley effects"], expected["Shapley effects"], 3)


def test_get_shapley_linear_three_inputs():
    """Tests for the module shapley.py for the test case in section 3.3 in Iooss and 
        Prieur (2019): Linear model with two Gaussian inputs."""
    def linear_model(x):
        beta = np.array([[beta_1], [beta_2], [beta_3]])
        return x.dot(beta)

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
            distribution = cp.MvNormal(mean[subset_j], cov_int)
            return distribution.sample(n)
        else:
            return _r_condmvn(
                n,
                mean=mean,
                cov=cov,
                dependent_ind=subset_j,
                given_ind=subsetj_conditional,
                x_given=xjc,
            )

    np.random.seed(123)
    n_inputs = 3
    mean = np.zeros(n_inputs)
    var_1 = 16
    var_2 = 4
    var_3 = 9
    
    # rho is the correlation coefficient, and thus, in range [-1,1]. Correlation between X2 and X3 only, i.e. rho = Corr[X2, X3].
    rho = 0.3
    covariance = rho * np.sqrt(var_2) * np.sqrt(var_3)
    beta_1 = 1.3
    beta_2 = 1.5
    beta_3 = 2.5
    #beta = (beta_1, beta_2, beta)

    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    component_3 = beta_3 ** 2 * var_3
    var_y = component_1 + component_2 + component_3 + 2 * covariance * beta_2 * beta_3
    share = 0.5 * (rho**2)
    true_shapley_1 = (component_1)/var_y
    true_shapley_2 = (component_2 + covariance * beta_2 * beta_3 + share * (component_3 - component_2))/var_y
    true_shapley_3 = (component_3 + covariance * beta_2 * beta_3 + share * (component_2 - component_3))/var_y

    cov = np.array(
        [[var_1, 0, 0],
        [0, var_2, covariance],
        [0, covariance, var_3]]
        )

    method = "exact"
    n_perms = None
    n_output = 10 ** 7
    n_outer = 10 ** 4
    n_inner = 10 ** 2

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [true_shapley_1, true_shapley_2, true_shapley_3]
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        linear_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated["Shapley effects"], expected["Shapley effects"], 3)


def test_get_shapley_ishigami():
    '''Test case and analytical Shapley values obtained from Plischke, Rabitti, and
    Borgonovo (2020), p. 8. Inputs are independent.'''
    def ishigami_function(x):
        return np.sin(x[:, 0]) * (1 + 0.1 * np.power(x[:, 2], 4)) + 7 * np.power(np.sin(x[:, 1]), 2)

    def x_all(n):
        distribution = cp.Iid(cp.Uniform(lower, upper), n_inputs)
        return distribution.sample(n)

    def x_cond_uniform(n, subset_j, subsetj_conditional, xjc):
        distribution = cp.Iid(cp.Uniform(lower, upper), len(subset_j))
        return distribution.sample(n)

    np.random.seed(123)
    n_inputs = 3
    mean = np.zeros(3)
    # Lower and upper bound of the uniform distribution.
    lower = -np.pi
    upper = np.pi
    variance = (1/12) * ((upper - lower) ** 2)
    # cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
    method = "exact"
    n_perms = None
    n_output = 10 ** 5
    n_outer = 10 ** 4
    n_inner = 10 ** 2

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [0.4358, 0.4424, 0.1218],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        ishigami_function,
        x_all,
        x_cond_uniform,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated['Shapley effects'], expected['Shapley effects'], 2)


def test_get_shapley_additive_three_inputs():
    """Test for the module shapley.py for the test case in section 3.5 in Iooss and 
        Prieur (2019): Linear model with two Gaussian inputs."""
    def additive_model(x):
        return x[:, 0] + x[:, 1] * x[:, 2]

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
            distribution = cp.MvNormal(mean[subset_j], cov_int)
            return distribution.sample(n)
        else:
            return _r_condmvn(
                n,
                mean=mean,
                cov=cov,
                dependent_ind=subset_j,
                given_ind=subsetj_conditional,
                x_given=xjc,
            )

    np.random.seed(123)
    n_inputs = 3
    mean = np.zeros(n_inputs)
    var_1 = 1
    var_2 = 1
    var_3 = 1
    rho = 0.3
    covariance = rho * np.sqrt(var_1) * np.sqrt(var_3)
    # Variance obtained analytically by myself.
    var_y = var_1 + var_2 * var_3
    
    cov = np.array(
        [[var_1, 0, covariance],
        [0, var_2, 0],
        [covariance, 0, var_3]]
        )
    
    true_shapley_1 = ((var_1 * (1 - ((rho ** 2) / 2))) + (((var_2 * var_3) * (rho ** 2)) / 6)) / var_y
    true_shapley_2 = (((var_2 * var_3) * (3 + (rho ** 2))) / 6) / var_y
    true_shapley_3 = (((var_1 * (rho ** 2)) / 2) + (((var_2 * var_3) * (3 - (2 * (rho ** 2)))) / 6)) / var_y

    method = "exact"
    n_perms = None
    n_output = 10 ** 5
    n_outer = 10 ** 4
    n_inner = 10 ** 3

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [true_shapley_1, true_shapley_2, true_shapley_3],
            [0, 0, 0],
            [true_shapley_1, true_shapley_2, true_shapley_3],
            [true_shapley_1, true_shapley_2, true_shapley_3],
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        additive_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated["Shapley effects"], expected["Shapley effects"], 2)


def compute_confidence_intervals(param_estimate, variance, critical_value):
    '''Compute confidence intervals (ci). Note assumptions about the distributions apply.
    
    Parameters
    ----------
    param_estimate: float
        Parameter estimate for which ci should be computed
    variance: float
        Variance of parameter estimate.
    critical_value: float
        Critical value of the t distribution, e.g. for the 95-percent-ci it's 1.96.

    Returns
    -------
    confidence_interval_dict: dict
        Lower (upper) bound of the ci can be accessed by the key 'lower_bound' ('upper_bound').
    
    '''
    confidence_interval_dict = {}
    confidence_interval_dict['lower_bound'] = param_estimate - critical_value * np.sqrt(variance)
    confidence_interval_dict['upper_bound'] = param_estimate + critical_value * np.sqrt(variance)
    return confidence_interval_dict


def get_simulated_shapley(n_replicates, method, model, x_all, x_cond, n_perms, n_inputs, n_output, n_outer, n_inner):

    shapley_effects_simulated = {}
    for i in np.arange(n_replicates):
        np.random.seed(i)
        exact_shapley = get_shapley(method, model, x_all, x_cond, n_perms, n_inputs, n_output, n_outer, n_inner)
        shapley_effects_simulated[i] = exact_shapley

    descriptives_data = np.zeros((n_inputs, 4))
    crit_value = 1.96
    for i in np.arange(n_inputs):
        k = i + 1
        variance = np.var([shapley_effects[i]['Shapley effects'][f'X{k}'] for i in np.arange(n_replicates)])
        mean = np.mean([shapley_effects[i]['Shapley effects'][f'X{k}'] for i in np.arange(n_replicates)])
        
        ci = compute_confidence_intervals(mean, variance, crit_value)
                                       
        std_error = np.sqrt(variance)
        descriptives_data[i, :] = np.array([mean, 
                                            std_error, 
                                    ci['lower_bound'], 
                                    ci['upper_bound']
                                    ]
                                    )

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    descriptives_shapley_effects = pd.DataFrame(descriptives_data, 
                                                columns=["Shapley effects", "std. errors", "CI_min", "CI_max"], 
                                                index=col
                                                )
    
    return descriptives_shapley_effects



def test_std_error_ishigami():
    np.random.seed(1234)

    lower = -np.pi
    upper = np.pi
    
    method = 'exact'
    n_perms = None
    n_inputs = 3
    n_output = 10 ** 3
    n_outer = 10 ** 2
    n_inner = 10

    def ishigami_function(x):
        return np.sin(x[:, 0]) * (1 + 0.1 * np.power(x[:, 2], 4)) + 7 * np.power(np.sin(x[:, 1]), 2)

    def x_all(n):
        distribution = cp.Iid(cp.Uniform(lower, upper), n_inputs)
        return distribution.sample(n)

    def x_cond_uniform(n, subset_j, subsetj_conditional, xjc):
        distribution = cp.Iid(cp.Uniform(lower, upper), len(subset_j))
        return distribution.sample(n)

    n_replicates = 10 ** 3
    shapley_simulated = get_simulated_shapley(n_replicates, method, ishigami_function, x_all, x_cond_uniform, n_perms, n_inputs, n_output, n_outer, n_inner)