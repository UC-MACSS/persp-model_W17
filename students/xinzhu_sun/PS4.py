import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
np.random.seed(seed=1234)

# Problem 1: some income data, lognormal distribution and GMM
incomes = np.loadtxt('incomes.txt')

#(a) Plot a histogram of percentages of the data with 30 bins.
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    Fig_1a  = Boolean, = True if make a histogram of percentages of the
                         annual incomes of MACSS graduats
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    count, bins, patches = plt.hist(incomes, 30, normed = True)
    plt.title('Annual incomes of MACSS graduates:2018-2020', fontsize=20)
    plt.xlabel('Annual incomes')
    plt.ylabel('Percent of Graduates')
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()

#(b) Write my own function of lognormal PDF and test it.
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate PDF values from a truncated normal distribution based on a
    normal distribution with mean mu and standard deviation sigma and
    cutoffs (cut_lb, cut_ub).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
             value(s) in the support of s~TN(mu, sig, cut_lb, cut_ub)
    mu     = scalar, mean of the nontruncated normal distribution from
             which the truncated normal is derived
    sigma  = scalar > 0, standard deviation of the nontruncated normal
             distribution from which the truncated normal is derived
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals   = (N, S) matrix, (N,) vector, or scalar in (0,1), PDF
                 values corresponding to xvals from truncated normal PDF
                 with base lognormal normal distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) 
    
    return pdf_vals

xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
print('1(b). The lognormal PDF value is ', LN_pdf(xvals, 5.0, 1.0), '.')

#(c) Perform SMM with 2 moments
def LN_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a truncated normal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    tLN_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                 alues drawn from truncated normal PDF with base
                 lognormal distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: tLN_draws
    --------------------------------------------------------------------
    '''
    draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    tLN_draws = np.exp(draws)

    return tLN_draws

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
    var_data  = scalar > 0 or (S,) vector, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        var_data = xvals.var()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        var_data = xvals.var(axis=0)
    
    return mean_data, var_data

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
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
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
    xvals, unif_vals, W_hat = args
    sim_vals = LN_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 

    return crit_val


S = 300
N = 200
unif_vals = sts.uniform.rvs(0, 1, size=(N, S))
mu_init = 11.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
smm_args = (incomes, unif_vals, W_hat)
results = opt.minimize(criterion, params_init, args=(smm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_SMM, sig_SMM = results.x

# Plot the estimated lognormal PDF against the histogram from (a)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1b  = Boolean, = True if plot the estimated lognormal PDF against
                         the income distribution histogram.
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated SMM PDF
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, LN_pdf(dist_incomes, mu_SMM, sig_SMM),\
             linewidth=2, color='r', label='The lognormal PDF of SMM')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.show()

# Report the value of SMM criterion function
crit_val = criterion(results.x, *smm_args)
print('1(c). The SMM estimator mu is ', mu_SMM, ', sigma is ', sig_SMM, '.')
print('      The value of the SMM criterion function is ', crit_val, '.')
# Report the two data moments and two model moments
mean_data, var_data = data_moments(incomes)
sim_vals = LN_draws(unif_vals, mu_SMM, sig_SMM)
mean_sim, var_sim = data_moments(sim_vals)
mean_model = mean_sim.mean()
var_model = var_sim.mean()
print('      Data Mean is ', mean_data, ', data standard deviation is ', np.sqrt(var_data), '.')
print('      Model Mean is ', mean_model, ', model standard deviation is ', np.sqrt(var_model), '.')

#(d) Perform the two-step SMM
err1 = err_vec(incomes, sim_vals, mu_SMM, sig_SMM, False)
VCV2 = np.dot(err1, err1.T) / incomes.shape[0]
# Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
W_hat2 = lin.pinv(VCV2)
params_init_two_step = np.array([mu_SMM, sig_SMM])
smm_args_two_step = (incomes,unif_vals, W_hat2)
results_two_step = opt.minimize(criterion, params_init_two_step, args=(smm_args_two_step),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
# Report the estimates as well as the criterion function value
mu_SMM_two_step, sig_SMM_two_step = results_two_step.x
crit_val_two_step= criterion(results_two_step.x, *smm_args_two_step)
print('1(d). The two-step SMM estimator mu is ', mu_SMM_two_step, ', sigma is ', sig_SMM_two_step, '.')
print('      The value of the two-step SMM criterion function is ', crit_val_two_step, '.')

# Plot the estimated lognormal PDF against the histogram from (a) and the
# estimated PDF from (c)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1c  = Boolean, = True if plot the estimated lognormal PDF of two-
                         step GMM against the income distribution histog-
                         ram and the estimated lognormal PDF from (c).
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated lognormal PDF from (b)
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, LN_pdf(dist_incomes, mu_SMM, sig_SMM),\
             linewidth=2, color='r', label='The lognormal PDF of SMM')
    # Plot the estimated lognormal PDF of two-step GMM
    plt.plot(dist_incomes, LN_pdf(dist_incomes, mu_SMM_two_step, sig_SMM_two_step),\
             linewidth=2, color='g', label='The lognormal PDF of two-step SMM')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    plt.show()

# Report the two data moments and two model moments of two-step GMM
mean_data_two_step, var_data_two_step = data_moments(incomes)
sim_vals_two_step = LN_draws(unif_vals, mu_SMM_two_step, sig_SMM_two_step)
mean_sim_two_step, var_sim_two_step = data_moments(sim_vals_two_step)
mean_model_two_step = mean_sim_two_step.mean()
var_model_two_step = var_sim_two_step.mean()
print('      Data Mean is ', mean_data_two_step, ', data standard deviation is ', np.sqrt(var_data_two_step), '.')
print('      Model Mean is ', mean_model_two_step, ', model standard deviation is ', np.sqrt(var_model_two_step), '.')










