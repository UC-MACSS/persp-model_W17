
import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
cmap1 = matplotlib.cm.get_cmap('summer')

income = np.loadtxt('incomes.txt')

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
    
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
'''
------------------------------------------------
Exercise 1a: Plot a histogram of the income data
------------------------------------------------
'''
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
# fig1a, ax1a = plt.subplots()

plt.hist(income, num_bins, weights = weights,normed=True)

plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')
    
output_path = os.path.join(output_dir, '1a_inc_hist')
plt.savefig(output_path)
#plt.show()
plt.close

'''
------------------------------------------------------------
Exercise 1b: Write own function for the lognormal PDF above called LN pdf()
Test your function by inputting the matrix 
xvals = np.array([[200.0, 270.0], [180.0, 195.5]]) 
with parameter values μ = 5.0 and sigma = 1.0.
------------------------------------------------------------
'''

def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the truncated normal pdf with mean mu and
    standard deviation sigma. If the cutoff is finite, then the PDF
    values are inflated upward to reflect the zero probability on values
    above the cutoff. If there is no cutoff given or if it is given as
    infinity, this function does the same thing as
    sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
   
    
    log_pdf_vals = ((1/(xvals*sigma * np.sqrt(2 * np.pi)) *
                        np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))) 
    return log_pdf_vals

print('Exercise 1b:')

xvals = np.array([[200.0, 270.0], [180.0, 195.5]]) 
print('Test result of the testing matrix using LN_pdf is:', LN_pdf(xvals,5.0,1.0))
print()
def trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub): #inverse CDF
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
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                  cutoff of truncated normal distribution
    cut_lb_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                  cutoff of truncated normal distribution
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
    # No cutoffs: truncated normal = normal
    if (cut_lb == None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb != None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb == None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb != None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    
    unif2_vals = unif_vals * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    tnorm_draws = sts.norm.ppf(unif2_vals, loc=mu, scale=sigma)
    draws_1_exp=np.exp(tnorm_draws)

    return draws_1_exp



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


def err_vec(data_vals, sim_vals, mu, sigma, cut_lb, cut_ub, simple):
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
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    std_data   = scalar > 0, standard deviation of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    std_model  = scalar > 0, estimated standard deviation from model
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
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
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
    xvals, unif_vals, cut_lb, cut_ub, W_hat = args
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    err = err_vec(xvals, sim_vals, mu, sigma, cut_lb, cut_ub,
                  simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val


#Estimate the parameters using identity matrix

np.random.seed(seed=1234)
N = 200
S = 300
mu_init = 11.0
sig_init = 0.2
cut_lb = 0.0
cut_up =  1500000
unif_vals_2 = sts.uniform.rvs(0, 1, size=(N, S))

params_init_1 = np.array([mu_init, sig_init])
W_hat1 = np.eye(2)
smm_args1 = (income, unif_vals_2, cut_lb, cut_up, W_hat1)
results1= opt.minimize(criterion, params_init_1, args=(smm_args1),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM1, sig_SMM1 = results1.x

print('Exercise 1c:')
#Report the estimates
print('The estimators mu_SMM1 is:',mu_SMM1)
print('The estimators sig_SMM1 is:',sig_SMM1)

#Report and compare two data moments against two model moments at the estimated parameter values
mean_data, std_data = data_moments(income)
print('Data mean of scores =', mean_data, ', Data standard deviation of scores =', std_data)
sim_vals_1 = trunc_norm_draws(unif_vals_2, mu_SMM1, sig_SMM1, cut_lb, cut_up)
mean_sim_1, std_sim_1 = data_moments(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
std_model_1 = std_sim_1.mean()
print('Model mean 1 =', mean_model_1, ', Model standard deviation 1 =', std_model_1)
params_SMM = np.array([mu_SMM1, sig_SMM1])
value = criterion(params_SMM,*smm_args1)[0][0]
print('The value of my SMM criterion function at the estimated parameter values is:',value)
print()

# Plot the histogram of the data
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1a, ax1a = plt.subplots()

plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

# Plot the estimated SMM PDF
dist_pts = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
         linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.legend(loc='upper left')

output_path = os.path.join(output_dir, '1c_SMM')
plt.savefig(output_path)
#plt.show()
plt.close


'''
------------------------------------------------------------
Exercise 1d: Perform the two-step SMM estimator 
by using estimators mu_SMM1,sig_SMM1
------------------------------------------------------------
'''
err_1 = err_vec(income, sim_vals_1, mu_SMM1, sig_SMM1, cut_lb, cut_up,
                False)
VCV2_1 = np.dot(err_1, err_1.T) / income.shape[0]
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned


smm_args2_1 = (income, unif_vals_2, cut_lb, cut_up, W_hat2_1)
results2_1 = opt.minimize(criterion, params_SMM, args=(smm_args2_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2, sig_SMM2 = results2_1.x

print('Exercise 1d:')
#Report the estimates
print('The estimators mu_SMM2 is:',mu_SMM2)
print('The estimators sig_SMM2 is:',sig_SMM2)
print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)

#Report and compare two data moments against two model moments at the estimated parameter values
sim_vals_2 = trunc_norm_draws(unif_vals_2, mu_SMM2, sig_SMM2, cut_lb, cut_up)
mean_sim_2, std_sim_2 = data_moments(sim_vals_2)
mean_model_2 = mean_sim_2.mean()
std_model_2 = std_sim_2.mean()
print('Data mean of scores =', mean_data, ', Data standard deviation of scores =', std_data)
print('Model mean 2 =', mean_model_2, ', Model standard deviation 2 =', std_model_2)
# print('Error vector 1 =', err_1)
# print(results1_1)
params_SMM_2 = np.array([mu_SMM2, sig_SMM2])
value2 = criterion(params_SMM_2,*smm_args2_1)[0][0]
print('The value of my SMM criterion function at the estimated parameter values is:',value2)


# Plot the histogram of the data
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1a, ax1a = plt.subplots()

plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

# Plot the estimated SMM PDF
dist_pts = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
         linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.legend(loc='upper left')

#Plot the estimate SMM PDF using two step estimator
dist_pts = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2, sig_SMM2),
         linewidth=2, color='g', label='2: $\mu_{SMM2}$,$\sigma_{SMM2}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, '1d_SMM_Two_step')
plt.savefig(output_path)
#plt.show()
plt.close




