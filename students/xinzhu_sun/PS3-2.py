
import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import scipy.stats as sts
import scipy.integrate as intgr
import pandas as pd


# Problem 2: Linear regression and GMM
sick = pd.read_csv('sick.txt')

def model_moments(df, betas):
    '''
    --------------------------------------------------------------------
    Given the dataset and the hypothesized beta values compute the model
    moment
    --------------------------------------------------------------------
    INPUTS:
    df: the dataframe
    betas: hypothesized beta values

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    fit: the predicted value using the hypothesized dataframe
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: fit
    --------------------------------------------------------------------
    '''
    fit = (betas * df.iloc[:, 1:].assign(const=1)).sum(axis = 1)
    
    return fit
        
def err_vec(df, betas):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    df: the dataframe
    betas: a numpy array, the hypothesized beta values
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    err_vec = df.iloc[:, 0] - model_moments(df, betas)
    
    return err_vec

def criterion(betas, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    betas: a numpy array containing all of the beta values
    *args: the input containting the dataframe and the weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    df, W = args
    err = err_vec(df, betas)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

betas_init = np.array([0, 0, 0, 1])
W_hat = np.eye(sick.shape[0])
gmm_args = (sick, W_hat)
results = opt.minimize(criterion, betas_init, args = (gmm_args), method = 'L-BFGS-B')
beta1, beta2, beta3, beta0 = results.x
crit_val = criterion(results.x, *gmm_args)
print('2(a). The estimated model parameters beta1, beta2, beta3, beta0 are as following:')
print('      ', beta1, ', ', beta2, ', ', beta3, ', ', beta0, '.')
print('      The value of the criterion function is ', crit_val, '.')


