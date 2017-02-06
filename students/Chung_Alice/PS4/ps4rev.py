# MACS30100: Problem Set #4
# Dr. Evans
#
# Name : Alice Mee Seon Chung 
#       (alice.chung@uchicago.edu)
# 

import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

np.seterr(all='ignore')

...
graph = True
...
if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    year_vec    = (lf_years,) vector, years from beg_year to
                  beg_year + lf_years
    individual  = integer in [0, numdraws-1], index of particular series
                  to plot
    --------------------------------------------------------------------
    '''
    # create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

'''
----------------------------------------------------------------------------
Problem 1. Some income data, lognormal distribution, and GMM
----------------------------------------------------------------------------
'''
# load the file
income = np.loadtxt('incomes.txt')

# set the seed
np.random.seed(1234)

# 1 - a

# plot the histogram
plt.figure(figsize=(11,10))
num_bins = 30 
plt.hist(income, num_bins, normed = True, color = 'grey')
plt.title('A histogram of annual incomes of MACSS students in 2018-2020 ',
             fontsize=15)
plt.xlabel('Annual Incomes')
plt.ylabel('PDF Values')
output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
#plt.show()
plt.close()


# 1 - b

def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate lognormal pdf values mean mu and standard deviation sigma. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    lognorm_pdf_vals = (( 1/( xvals *sigma * np.sqrt(2 * np.pi)) *
                        np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))))

    return lognorm_pdf_vals

# 1 - b test

mu = 5.0
sigma = 1.0
test_value = np.array([[200.0, 270.0], [180.0, 195.5]])

print("1-b.", )
print("Test function my putting given matrix is", LN_pdf(test_value, mu, sigma))

# 1- c

def lognorm_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws based on a normal distribution 
    with mean mu and standard deviation sigma. These draws correspond 
    to an (N x S) matrix of randomly generated draws from a uniform 
    distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    RETURNS: draws
    --------------------------------------------------------------------
    '''

    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    draws = np.exp(norm_draws)
    
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

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = lognorm_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple = False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val
   
# initial values 
N = 200
S = 300
mu_init = 11.0
sig_init = 0.2

#uniform values
np.random.seed(seed=1234)
unif_vals = sts.uniform.rvs(0, 1, size=(N, S))

# calculate SMM
params_init_1 = np.array([mu_init, sig_init])
W_hat1 = np.eye(2)
smm_args1 = (income, unif_vals, W_hat1)
results1= opt.minimize(criterion, params_init_1, args=(smm_args1),
                           method='L-BFGS-B',
                           bounds=((1e-10, None), (1e-10, None)))

# get the first SMMs
mu_SMM1, sig_SMM1 = results1.x

# CV
params = np.array([mu_SMM1, sig_SMM1])
args = (income, unif_vals, W_hat1)
crit_test = criterion(params,*args)[0][0]

# actual data moments
mean_data, std_data = data_moments(income)

# simulated data moments
sim_vals_1 = lognorm_draws(unif_vals, mu_SMM1, sig_SMM1)
mean_sim_1, std_sim_1 = data_moments(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
std_model_1 = std_sim_1.mean()

# plot the histogram

# 1-a
plt.figure(figsize=(11,10))
num_bins = 30 
plt.hist(income, num_bins, normed = True, color = 'grey')

# 1-c
dist_income = np.linspace(0, 150000, 500)
plt.plot(dist_income, LN_pdf(dist_income, mu_SMM1, sig_SMM1), linewidth=2, color='k',
         label='(c): $\mu_{SMM1} = 11.3306$,$\sigma_{SMM1} = 0.2092$')

plt.title('PDF plots of MACSS students in 2018-2020 ', fontsize=15)
plt.xlabel('Annual Incomes')
plt.ylabel('PDF Values')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
#plt.show()
plt.close()

print()
print('1-c.')
print('Mu for SMM is', mu_SMM1, ' sigma for SMM s', sig_SMM1)
print("The value of SMM criterion function at the estimated parameter values is",
         crit_test)
print('Data mean of scores =', mean_data,
 ', Data standard deviation of scores =', std_data)
print('Model mean of SMM =', mean_model_1, 
      ', Model standard deviation of SMM =', std_model_1)
print('Data moments and model moments are very close. We can say that SMM estimates well.')


# 1 - d
# two-step

err2_1 = err_vec(income, sim_vals_1, mu_SMM1, sig_SMM1, False)
VCV2_1 = np.dot(err2_1, err2_1.T) / income.shape[0]
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

# two-step SMM
params_init2_1 = np.array([mu_SMM1, sig_SMM1])

smm_args2_1 = (income, unif_vals, W_hat2_1)
results2_1 = opt.minimize(criterion, params_init2_1, args=(smm_args2_1),
                         method='TNC',
                         bounds=((1e-10, None), (1e-10, None)))

# two-step SMMs
mu_SMM2_1, sig_SMM2_1 = results2_1.x

# CV with two-step SMMs
params2_1 = np.array([mu_SMM2_1, sig_SMM2_1])
args2_1 = (income, unif_vals, W_hat2_1) 
crit_test2_1 = criterion(params2_1,*args2_1)[0][0]

# actual data moments
mean_data_1, std_data_1 = data_moments(income)

# simulated model moments 
sim_vals_3 = lognorm_draws(unif_vals, mu_SMM2_1, sig_SMM2_1)
mean_sim_3, std_sim_3 = data_moments(sim_vals_3)
mean_model3 = mean_sim_3.mean()
std_model3 = std_sim_3.mean()


print()
print('1-d.')
print('Mu for two-step SMM is', mu_SMM2_1, 
    ' sigma for two-step SMM s', sig_SMM2_1)
print("The value of SMM criterion function at the estimated parameter values is"
    , crit_test2_1)
print('Data mean of scores =', mean_data_1, 
    ', Data standard deviation of scores =', std_data_1)
print('Model mean of SMM =', mean_model3, 
', Model standard deviation of SMM =', std_model3)
print('Data moments and model moments are very close. We can say that two-step ',
    'SMM estimates well.')

# plot
# 1-a
plt.figure(figsize=(11,10))
num_bins = 30 
plt.hist(income, num_bins, normed = True, color = 'grey')

# 1-c
dist_income = np.linspace(0, 150000, 500)
plt.plot(dist_income, LN_pdf(dist_income, mu_SMM1, sig_SMM1), linewidth=4, color='k',
         label='(c): $\mu_{SMM1} = 11.3306$,$\sigma_{SMM1} = 0.2092$')

# 1-d
dist_income = np.linspace(0, 150000, 500)
plt.plot(dist_income, LN_pdf(dist_income, mu_SMM2_1, sig_SMM2_1), linewidth=2, color='pink',
         label='(d): $\mu_{SMM2} = 11.3306$,$\sigma_{SMM2} = 0.2092$')

plt.title('PDF plots of MACSS students in 2018-2020 ', fontsize=15)
plt.xlabel('Annual Incomes')
plt.ylabel('PDF Values')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
#plt.show()
plt.close()



