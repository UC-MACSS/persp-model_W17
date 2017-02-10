# Import packages and load the data
import numpy as np
import pandas as pd
import os
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#--------------------------------------------------------------------
#Problem 1a
#--------------------------------------------------------------------
def f1a():
    '''
    Problem 1a
    --------------------------------------------------------------------
    This function plots a histogram of percentages of the incomes.txt data
    with 30 bins. The bins are weighted using the normed=True option.
    --------------------------------------------------------------------
    INPUT: none
    RETURN: none
    --------------------------------------------------------------------
    '''
    fig, ax = plt.subplots(figsize=(10,6)) # Set the size of the window of the figure to be plotted.
    count, bins, ignored = plt.hist(pts, 30, normed=True)
    plt.title('Annual Incomes of MACSS graduates: 2018-2020', fontsize=20)
    plt.xlabel('income')
    plt.ylabel('percent of income')


#--------------------------------------------------------------------
#Problem 1b
#--------------------------------------------------------------------
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    This function generates pdf values from the lognormal pdf with mean 
    mu and standard deviation sigma.
    --------------------------------------------------------------------
    INPUT: 
        xvals: either a one-dimensional (N; ) vector or an N by S matrix
               of simulated data where each column is a simulation of the
               N data points on income.
        mu: the mean of the normal distribution on which the lognormal is
            based.
        sigma: the standard deviation of the normal distribution on which
               the lognormal is based.
    RETURN: pdf_vals: an numpy array that takes the same size as xvals, 
                      and represents the lognormal PDF values of each 
                      element of xvals given the parameters mu and sigma.
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1: # if xvals is a one-dimention vector
        pdf_vals  = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
    elif xvals.ndim == 2: # if xvals is a 2-dimention array
        val = lambda x : (1/(x * sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2)) # val is a function
        pdf_vals = np.apply_along_axis(val, axis = 1, arr = xvals)  # axis: 1-column wise, 0-row wise, arr: input matrix

    return pdf_vals


#--------------------------------------------------------------------
#Problem 1c
#--------------------------------------------------------------------
def LN_draws(mu, sigma, N, S, seed):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of incomes from a lognormal distribution based on
    the mean mu and standard deviation sigma.
    --------------------------------------------------------------------
    INPUTS:
    mu        = scalar, mean of the lognormal distribution
    sigma     = scalar > 0, standard deviation of lognormal distribution 
    N         = scalar > 0, the number of rows of the matrix
    S         = scalar > 0, the number of columns of the matrix
    seed      = scalar > 0, for generating random values

    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        np.random.normal()
    
    OBJECTS CREATED WITHIN FUNCTION:
    LN_darws
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: draws
    --------------------------------------------------------------------
    '''
    np.random.seed(seed)
    normal_dist = rnd.normal(mu, sigma, size = (N, S)) # draw an N by S matrix of values from a normal distribution with mu and sigma.
    draws = np.exp(normal_dist) # transform those draws into the lognormal distribution by exponentiating them.
    
    return draws


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    std_data  = scalar > 0 or (S,) vector, standrad variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        std_data = xvals.std()
    elif xvals.ndim == 2:
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
    std_data   = scalar > 0, standard variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    std_model  = scalar > 0, estimated standard variance from model
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
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    draw      = (N, S) matrix
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    N         = scalar > 0, the number of rows of the matrix
    S         = scalar > 0, the number of columns of the matrix
    seed      = scalar > 0, for generating random values
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, SMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W_hat, N, S, seed = args
    sim_vals = LN_draws(mu, sigma, N, S, seed)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val

def f1c():
    '''
    --------------------------------------------------------------------
    This function estimates the parameters of the lognormal distribution
    by simulated method of moments (SMM)
    --------------------------------------------------------------------
    INPUT: none
    RETURN: mu_SMM1, sig_SMM1, sim_vals
    --------------------------------------------------------------------
    '''
    pts = np.loadtxt('incomes.txt') # load txt file
    N = 200.0
    S = 300.0
    seed = 1234
    mu = 9.0
    sigma = 0.3 
    params_init = np.array([mu, sigma])
    W_hat = np.eye(2)
    args = (pts, W_hat, N, S, seed)
    results_c = opt.minimize(criterion, params_init, args=(args), method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
    mu_SMM1, sig_SMM1 = results_c.x
    # Report the estimated parameter values:
    print('1c. mu_SMM1=', mu_SMM1, ' sig_SMM1=', sig_SMM1)

    # Calculate and report the value of the GMM criterion function at the estimated parameter values:
    params_SMM = np.array([mu_SMM1, sig_SMM1])
    value = criterion(params_SMM, *args)[0][0]
    print('The value of the SMM criterion function at the estimated parameter values is', value)

    # Calculate and report and compare my two data moments against my two model moments at the estimated parameter values.
    sim_vals = LN_draws(mu_SMM1, sig_SMM1, N, S, seed)
    mean_data, std_data = data_moments(pts)
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    print('Mean of incomes =', mean_data, ', Standard deviation of incomes =', std_data)
    print('Mean of model =', mean_model, ', Standard deviation of model =', std_model)
    # print(results_c)-success

    # Plot the estimated lognormal PDF against the histogram from part (a):
    f1a()
    dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1), linewidth=2, color='k', label='$\mu_{b}$: 11.331,$\sigma_{b}$: 0.210')
    plt.legend(loc='upper right')

    return mu_SMM1, sig_SMM1, sim_vals

#--------------------------------------------------------------------
#Problem 1d
#--------------------------------------------------------------------
def f1d(mu_SMM1, sig_SMM1, sim_vals):
    '''
    --------------------------------------------------------------------
    This function performs the two-step SMM estimator by using the 
    estimates from part (c) with two moments to generate an estimator for
    the variance covariance matrix omega hat 2step, which then will be 
    used to get the two-step estimator for the optimal weighting matrix W
    hat 2step.
    --------------------------------------------------------------------
    INPUT: mu_SMM1, sig_SMM1, sim_val
    RETURN: none
    --------------------------------------------------------------------
    '''
    pts = np.loadtxt('incomes.txt') # load txt file
    S = 300.0
    N = 200.0
    seed = 1234
    err1 = err_vec(pts, sim_vals, mu_SMM1, sig_SMM1, False)
    VCV2 = np.dot(err1, err1.T) / pts.shape[0]
    W_hat2 = lin.pinv(VCV2)
    mu = 11
    sigma = 0.2
    params_init = np.array([mu_SMM1, sig_SMM1])
    args = (pts, W_hat2, N, S, seed)
    results_d = opt.minimize(criterion, params_init, args=(args), method='Nelder-Mead', bounds=((None, None), (1e-10, None)))
    mu_SMM2, sig_SMM2 = results_d.x
    # Report the estimated parameter values:
    print('1d. mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)

    # Calculate and report the value of the GMM criterion function at the estimated parameter values:
    params_SMM = np.array([mu_SMM2, sig_SMM2])
    value = criterion(params_SMM, *args)[0][0]
    print('The value of the SMM criterion function at the estimated parameter values is', value)

    # Calculate and report and compare my two data moments against my two model moments at the estimated parameter values.
    sim_vals = LN_draws(mu_SMM2, sig_SMM2, N, S, seed)
    mean_data, std_data = data_moments(pts)
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    print('Mean of incomes =', mean_data, ', Standard deviation of incomes =', std_data)
    print('Mean of model =', mean_model, ', Standard deviation of model =', std_model)
    # print(results_d)-success

    # Plot the estimated lognormal PDF against the histogram from part (a) and the estimated PDF from part (c):
    f1a()
    #part (c)
    dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1), linewidth=2, color='k', label='$\mu_{b}$: 11.331,$\sigma_{b}$: 0.210')
    plt.legend(loc='upper right')
    #part (d)
    dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2, sig_SMM2), '--', linewidth=2, color='r', label='$\mu_{b}$: 11.330,$\sigma_{b}$: 0.211')
    plt.legend(loc='upper right')


def save_figure(filename):
    '''
    --------------------------------------------------------------------
    This function saves figures into the images/ directory. 
    --------------------------------------------------------------------
    INPUT: a string represents the filename of a figure to be saved
    RETURN: none
    --------------------------------------------------------------------
    '''
    fn = os.path.split(filename)[1]
    if not os.path.exists('images'):
        os.mkdir('images')
    fn = os.path.join('images', fn)
    plt.savefig(fn)
    print(fn + ' saved.')


# Plots produced in problem 1a, are saved in the "images" folder.
if '__main__'==__name__:
    # plt.ion()
    pts = np.loadtxt('incomes.txt') # load txt file
    # Problem 1a:
    f1a(); save_figure('Fig_1a');  plt.close()
    # Problem 1b:  
    xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
    mu = 5.0 
    sigma = 1.0
    pdf_vals = LN_pdf(xvals, mu, sigma)
    print(pdf_vals)
    # Problem 1c:
    mu_SMM1, sig_SMM1, sim_vals = f1c(); save_figure('Fig_1c');  plt.close()
    # Problem 1d:
    f1d(mu_SMM1, sig_SMM1, sim_vals); save_figure('Fig_1d');  plt.close()
    pass