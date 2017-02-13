# Import packages and load the data
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import os

# For PART 1c
def norm_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a normal distribution with
    mean mu and standard deviation sigma. These draws correspond to an (N x S)
    matrix of randomly generated draws from a uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the normal distribution
    sigma     = scalar > 0, standard deviation of the normal distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()

    OBJECTS CREATED WITHIN FUNCTION:
    unif2_vals  = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  rescaled uniform derived from original.
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma) and cutoffs
                  (cut_lb, cut_ub)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''

    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)

    return norm_draws

def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, income data, "real" and simulated.
            Real world data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        std_data = xvals.std()
    else:
        mean_data = xvals.mean(axis=0)
        std_data = xvals.std(axis=0)

    return mean_data, std_data

def err_vec(data_vals, sim_vals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    moms_model = np.array([[mean_model], [std_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 3 tuple,
                (xvals, unif_vals, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_draws()
        np.exp()

    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    sim_vals   = (N, S) matrix of the simulated data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = np.exp(norm_draws(unif_vals, mu, sigma))
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val
