# Import packages
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt

def model_moments(data, beta0, beta1, beta2, beta3):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM.
    --------------------------------------------------------------------
    INPUTS:
    data = data in numpy array format
    beta0 = scalar
    beta1 = scalar
    beta2 = scalar
    beta3 = scalar

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mod_mom_array = array containing calculated model moments

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mod_mom_array
    --------------------------------------------------------------------
    '''
    mod_mom_array = np.ones(200,)
    for x in range(200):
        mod_mom_array[x,] = beta0 + beta1 * data[x,1] + beta2 * data[x,2] + beta3 * data[x,3]

    return mod_mom_array

def error_vec(data, beta0, beta1, beta2, beta3):
    '''
    --------------------------------------------------------------------
    This function computes the error vector for GMM.
    --------------------------------------------------------------------
    INPUTS:
    data = data in numpy array format
    beta0 = scalar
    beta1 = scalar
    beta2 = scalar
    beta3 = scalar

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    model_moments

    OBJECTS CREATED WITHIN FUNCTION:
    dat_mom_array = array containing actual data values
    mod_mom_array = array containing calculated model moments
    err_vec = array containing the simple difference between data and model moments

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    dat_mom_array = data[:,0]
    mod_mom_array = model_moments(data, beta0, beta1, beta2, beta3)
    err_vec = mod_mom_array - dat_mom_array

    return err_vec

def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:

    params = (4,) vector, ([beta0, beta1, beta2, beta3])
        data = data in numpy array format
        beta0 = scalar
        beta1 = scalar
        beta2 = scalar
        beta3 = scalar
    args   = length 2 tuple, (data, W)
    W  = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    error_vec

    OBJECTS CREATED WITHIN FUNCTION:
    err_vec_array  = (200,) matrix, column vector of two moment error
                     functions
    crit_val       = scalar > 0, GMM criterion function value

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3 = params
    data, W = args
    err_vec_array = error_vec(data, beta0, beta1, beta2, beta3)
    crit_val = np.dot(np.dot(err_vec_array.T, W), err_vec_array)

    return crit_val
