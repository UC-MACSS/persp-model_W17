# Problem set 4 script
# Yiqing Zhu


import os
import numpy as np
import pandas as pd
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


data = np.loadtxt('incomes.txt')

'''
-------------------------------------------------------------------------------
Problem 1a:
Plot a histogram of percentages of the income.txt data with 30 bins. 
Make sure that the bins are weighted using the normed=True option.
Make sure your plot has correct x-axis and y-axis labels as well as a plot title.
-------------------------------------------------------------------------------
'''
print('Problem 1a:')
plot_1a = True

if plot_1a:
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram
    fig = plt.figure()
    # fig, ax = plt.subplots()
    # weights = (1 / data.shape[0]) * np.ones_like(data)
    plt.hist(data, bins = 30, normed = True)
    plt.xlim([20000, 160000])
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1a')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()

    
'''
-------------------------------------------------------------------------------
Problem 1b:
Write your own function for the lognormal PDF above called LN_pdf().

Have this function take as inputs xvals, mu, and sigma, where xvals is either 
a one-dimensional (N,) vector or an N × S matrix of simulated data where 
each column is a simulation of the N data points on income, μ is the mean of 
the normal distribution on which the lognormal is based, and σ is the standard 
deviation of the normal distribution on which the lognormal is based. 
This function should return an array pdf vals that is the same size as xvals 
and represents the lognormal PDF values of each element of xvals given 
the parameters μ and σ. 

Test your function by inputting the matrix 
xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
with parameter values μ = 5.0 and σ = 1.0.
-------------------------------------------------------------------------------
'''

def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and standard
    deviation sigma. This function does the same thing as 
    sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N, S) matrix, or (N,) vector, simulated data
    mu     = scalar, mean of the normal distribution
    sigma  = scalar > 0, standard deviation of the normal distribution
             
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:

    LN_pdf_vals = (N, S) matrix, or (N,) vector, log normal PDF values 
                  for mu and sigma corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: LN_pdf_vals
    --------------------------------------------------------------------
    '''
            
    LN_pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
                  np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))

    return LN_pdf_vals

# Test the LN_pdf function
print('\nProblem 1b:')
test_vals = LN_pdf(xvals=np.array([[200.0, 270.0], [180.0, 195.5]]), mu=5.0, sigma=1.0)
print(test_vals)


'''
-------------------------------------------------------------------------------
Problem 1c:
Estimate the parameters of the lognormal distribution by simulated method of 
moments (SMM).

Create S = 300 simulations, each with N = 200 observations on income from the 
lognormal distribution above. Use the average income and standard deviation of 
income as your two moments. Use the identity matrix as your weighting matrix Wˆ.
To draw your N × S matrix of incomes from the lognormal distribution within 
your criterion function, set the seed to 1234 in the random number generator 
by typing np.random.seed(seed=1234). Then draw an N × S matrix of values from 
a normal distribution with mean μ and standard deviation σ. Then transform 
those draws into the lognormal distribution by exponentiating them numpy.exp().

Plot your estimated lognormal PDF against the histogram from part (a).
Report the value of your SMM criterion function at the estimated parameter values.
Report and compare your two data moments against your two model moments at the 
estimated parameter values. 
-------------------------------------------------------------------------------
''' 

def LN_draws(N, S, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N, S) matrix of random draws from a lognormal distribution 
    with mean mu and standard deviation sigma.
    --------------------------------------------------------------------
    INPUTS:
    N      = scalar, the number of observations
    S      = scalar, the number of simulations
    mu     = scalar, mean of the normal distribution
    sigma  = scalar > 0, standard deviation of the normal distribution
   
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        np.random.seed()
    
    OBJECTS CREATED WITHIN FUNCTION:
    LN_draws = (N, S) matrix of values drawn from lognormal PDF with base
               normal distribution N(mu, sigma)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: LN_draws
    --------------------------------------------------------------------
    '''
    np.random.seed(1234)
    norm_dist = rnd.normal(mu, sigma, size = (N, S))
    LN_draws = np.exp(norm_dist)
    
    return LN_draws

    
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), standard deviation(data)) from both the actual data and 
    from the simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, or (N,) vector, income data, 
            either real world or simulated.
            Real world data will come in the form (N,). 
            Simulated data comes in the form (N,) or (N, S).
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of income data
    std_data  = scalar > 0 or (S,) vector, standard deviation of income data
    
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
    data_vals = (N,) vector, income data
    sim_vals  = (N, S) matrix, S simulations of income data
    mu        = scalar, mean of the normal distribution
    sigma     = scalar > 0, standard deviation of the normal distribution
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
    mean_data, var_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
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
    args      = length 4 tuple,
                (xvals, N, S, W_hat)
    xvals     = (N,) vector, values of the normally distributed random variable
    N         = scalar, the number of observations
    S         = scalar, the number of simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        LN_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, N, S, W_hat = args
    sim_vals = LN_draws(N, S, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val
    
     
print('\nProblem 1c:')

# Estimate the parameters of the lognormal distribution by SMM
mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
S = 300
N = 200
W_hat = np.eye(2)
smm_args = (data, N, S, W_hat)
results_1 = opt.minimize(criterion, params_init, args=(smm_args),
                          method='L-BFGS-B',
                          bounds=((None, None), (1e-10, None)))
mu_SMM1, sig_SMM1 = results_1.x
print('mu_SMM =', mu_SMM1, ' sig_SMM =', sig_SMM1)

# Report the value of the SMM criterion function at the estimated parameter values.
params_SMM1 = np.array([mu_SMM1, sig_SMM1])
value1 = criterion(params_SMM1, *smm_args)[0][0]
print('The value of the SMM criterion function at the estimated parameter values is ',\
      value1)

# Report and compare two data moments against two model moments at the 
# estimated parameter values. 
mean_data, std_data = data_moments(data)
sim_vals1 = LN_draws(N, S, mu_SMM1, sig_SMM1)
mean_sim1, std_sim1 = data_moments(sim_vals1)
mean_model1 = mean_sim1.mean()
std_model1 = std_sim1.mean()
print('mean_data =', mean_data, ' std_data =', std_data)
print('mean_model =', mean_model1, ' std_model =', std_model1)
print('Two data moments and two model moments at the estimated parameter values \
are very close. ')

# Plot the estimated lognormal PDF against the histogram from part (a).
plot_1c = True
if plot_1c:
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram of the data
    fig = plt.figure()
    # fig, ax = plt.subplots()
    # weights = (1 / data.shape[0]) * np.ones_like(data)
    plt.hist(data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    
    # Plot the estimated SMM PDF
    dist_pts = np.linspace(30000, 150000, 10000)
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
         linewidth=2, color='g', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
    plt.legend(loc='upper left')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1c')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()
    

'''
-------------------------------------------------------------------------------
Problem 1d:
Perform the two-step SMM estimator by using your estimates from part (c) with 
two moments to generate an estimator for the variance covariance matrix Ωˆ2step,
which you then use to get the two-step estimator for the optimal weighting 
matrix Wˆ2step. 

Report your estimates as well as the criterion function value at these estimates.
Plot your estimated lognormal PDF against the histogram from part (a) and the 
estimated PDF from part (c). Report and compare your two data moments against 
your two model moments at the estimated parameter values.
-------------------------------------------------------------------------------
''' 
print('\nProblem 1d:')
# Perform the two-step SMM estimator by using the estimates from part (c)
err2 = err_vec(data, sim_vals1, mu_SMM1, sig_SMM1, False)
VCV2 = np.dot(err2, err2.T) / data.shape[0]
W_hat2 = lin.pinv(VCV2)
params_init2 = np.array([mu_SMM1, sig_SMM1])
smm_args2 = (data, N, S, W_hat2)
results2 = opt.minimize(criterion, params_init2, args=(smm_args2),
                         method='TNC',
                         bounds=((None, None), (1e-10, None)))
mu_SMM2, sig_SMM2 = results2.x

# Report the estimates
print('mu_SMM =', mu_SMM2, ' sig_SMM =', sig_SMM2)

# Report the value of the SMM criterion function at the estimated parameter values.
params_SMM2 = np.array([mu_SMM2, sig_SMM2])
value2 = criterion(params_SMM2, *smm_args2)[0][0]
print('The value of the SMM criterion function at the estimated parameter values is ',\
      value2)

# Report and compare two data moments against two model moments at the 
# estimated parameter values. 
sim_vals2 = LN_draws(N, S, mu_SMM2, sig_SMM2)
mean_sim2, std_sim2 = data_moments(sim_vals2)
mean_model2 = mean_sim2.mean()
std_model2 = std_sim2.mean()
print('mean_data =', mean_data, ' std_data =', std_data)
print('mean_model =', mean_model2, ' std_model =', std_model2)
print('Two data moments and two model moments at the estimated parameter values \
are very close. ')

# Plot the estimated lognormal PDF against the histogram from part (a) and the 
# estimated PDF from part (c).
plot_1d = True
if plot_1d:
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram of the data
    fig = plt.figure()
    # fig, ax = plt.subplots()
    # weights = (1 / data.shape[0]) * np.ones_like(data)
    plt.hist(data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    
    # Plot the estimated SMM PDF from part(a)
    dist_pts = np.linspace(30000, 150000, 10000)
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
         linewidth=2, color='g', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
    plt.legend(loc='upper left')

    # Plot the estimated two-step SMM PDF
    dist_pts = np.linspace(30000, 150000, 10000)
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2, sig_SMM2),
         linewidth=2, color='y', label='2: $\mu_{SMM2}$,$\sigma_{SMM2}$')
    plt.legend(loc='upper left')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1d')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()