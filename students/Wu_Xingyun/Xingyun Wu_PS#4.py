# Perspective on Computational Modeling
# Problem Set #4
# Xingyun Wu


# Import packages and load the data
import os
import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt


# Set random seed
np.random.seed(seed = 1234)


# Create directory if images directory does not already exist
'''
--------------------------------------------------------------------
cur_path    = string, path name of current directory
output_fldr = string, folder in current path to save files
output_dir  = string, total path of images folder
output_path = string, path of file name of figure to be saved
--------------------------------------------------------------------
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
    

# Read in the data
income = np.loadtxt('incomes.txt')


# PROBLEM (a)
print("Problem (a)")
plot_1a = True
if plot_1a:
    num_bins = 30
    weights = (1/len(income))*np.ones_like(income)
    count, bins, ignored = plt.hist(income, num_bins, weights = weights, \
                                    color='pink', normed=True)
    # set title and labels
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()
print()


# PROBLEM (b)
print("Problem (b)")
# Define function that gives PDF values from lognormal distribution
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

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals   = (N, S) matrix, (N,) vector, or scalar in (0,1), PDF
                 values corresponding to xvals from truncated normal PDF
                 with base normal normal distribution N(mu, sigma) and
                 cutoffs (cut_lb, cut_ub)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    
    pdf_vals = 1 / (xvals * sigma * np.sqrt(2 * np.pi)) * \
                 np.exp(-(np.log(xvals) - mu)**2 / (2 * sigma ** 2))      
    
    return pdf_vals

# function test
xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
mu_test = 5.0
sig_test = 1.0
print("testing the function with given xvals and parameters: ")
print(LN_pdf(xvals, mu_test, sig_test))
print()


# PROBLEM (c)
print("Problem (c)")

# Define function that draws N x S test score values from a truncated
# normal distribution
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
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    norm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                 values drawn from truncated normal PDF with base
                 normal distribution N(mu, sigma) and cutoffs
                 (cut_lb, cut_ub)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''

    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    ln_draws = np.exp(norm_draws)
    
    return ln_draws


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
    std_data  = scalar > 0 or (S,) vector, standard deviation of test scores data
    
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
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
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


# set parameters
N = 200
S = 300
mu_init = 9.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
unif_vals_c = sts.uniform.rvs(0, 1, size=(N, S))
W_hat_c = np.eye(2)
smm_args_c = (income, unif_vals_c, W_hat_c)

# smm estimation
results_c = opt.minimize(criterion, params_init, args=(smm_args_c),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM_c, sig_SMM_c = results_c.x
print('mu_SMM_c=', mu_SMM_c, ' sig_SMM_c=', sig_SMM_c)

mean_data, std_data = data_moments(income)
print("mean_data = ", mean_data, ", std_data = ", std_data)
sim_vals_c = LN_draws(unif_vals_c, mu_SMM_c, sig_SMM_c)
mean_sim_c, std_sim_c = data_moments(sim_vals_c)
#print("mean_sim = ", mean_sim, ", std_sim = ", std_sim)
mean_mod_c = mean_sim_c.mean()
std_mod_c = std_sim_c.mean()
print('Estimated model mean =', mean_mod_c)
print('Estimated model standard deviation =', std_mod_c)
# get value of criterion function
params_SMM_c = np.array([mu_SMM_c, sig_SMM_c])
crit_val_c = criterion(params_SMM_c, *smm_args_c)[0][0]
print('Value of my SMM criterion function at the estimated parameter values:',\
      crit_val_c)

cutoff = 150000
# plot (c)
plot_1c = True
if plot_1c:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the estimated SMM PDF
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, LN_pdf(dist_inc, mu_SMM_c, sig_SMM_c),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'\
                                                .format(mu_SMM_c, sig_SMM_c))
    plt.legend(loc='upper left')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    plt.show()
    plt.close()

print()


# PROBLEM (d): 2 step estimation
print("Problem (d)")
err2_d = err_vec(income, sim_vals_c, mu_SMM_c, sig_SMM_c, False)
VCV2_d = np.dot(err2_d, err2_d.T) / income.shape[0]
print("The variance-covariance matrix is: ")
print(VCV2_d)
W_hat_d = lin.pinv(VCV2_d)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print("The optimal weighting matrix is: ")
print(W_hat_d)

#unif_vals_d = sts.uniform.rvs(0, 1, size=(N, S))
smm_args_d = (income, unif_vals_c, W_hat_d)

# smm estimation
results_d = opt.minimize(criterion, params_SMM_c, args=(smm_args_d),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM_d, sig_SMM_d = results_d.x
print('mu_SMM_d=', mu_SMM_d, ' sig_SMM_d=', sig_SMM_d)

mean_data, std_data = data_moments(income)
print("mean_data = ", mean_data, ", std_data = ", std_data)
sim_vals_d = LN_draws(unif_vals_c, mu_SMM_d, sig_SMM_d)
mean_sim_d, std_sim_d = data_moments(sim_vals_d)
#print("mean_sim = ", mean_sim, ", std_sim = ", std_sim)
mean_mod_d = mean_sim_d.mean()
std_mod_d = std_sim_d.mean()
print('Model mean of one-step estimation =', mean_mod_c)
print('Model standard deviation of one-step estimation = ', std_mod_c)
print('Model mean of two-step estimation = ', mean_mod_d)
print('Model standard deviation of two-step estimation = ', std_mod_d)
# get value of criterion function
params_SMM_d = np.array([mu_SMM_d, sig_SMM_d])
crit_val_d = criterion(params_SMM_d, *smm_args_d)[0][0]
print('Value of my SMM criterion function at the estimated parameter values:',\
      crit_val_d)

# plot (c)
plot_1c = True
if plot_1c:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the 1st step estimated SMM PDF
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, LN_pdf(dist_inc, mu_SMM_c, sig_SMM_c),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'\
                                                .format(mu_SMM_c, sig_SMM_c))
    plt.legend(loc='upper left')
    # Plot the 2nd step estimated SMM PDF
    plt.plot(dist_inc, LN_pdf(dist_inc, mu_SMM_d, sig_SMM_d),\
         linewidth=2, color='b', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'\
                                                .format(mu_SMM_d, sig_SMM_d))
    plt.legend(loc='upper left')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1d')
    plt.savefig(output_path)
    plt.show()
    plt.close()

print()