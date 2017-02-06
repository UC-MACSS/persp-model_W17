##ps4

##Zhuo Leng

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')


'''
--------------------------------------------------------------------
problem 1(a)histogram
--------------------------------------------------------------------
'''

incomes = np.loadtxt('incomes.txt')

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath('ps3.py'))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
	os.makedirs(output_dir)

##plot a histogram
fig = plt.figure(figsize = (10,10))
# weights = (1 / data.shape[0]) * np.ones_like(data)
count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=20)
plt.xlabel(r'annual incomes(\$)')
plt.ylabel(r'PDF value of annual incomes')

output_path = os.path.join(output_dir, 'Fig_1a')
plt.savefig(output_path)
#plt.show()
plt.close()



'''
--------------------------------------------------------------------
problem 1(b)
--------------------------------------------------------------------
'''

def LN_pdf(xvals, mu, sigma):
	'''
    --------------------------------------------------------------------
    Generate PDF values from a log normal distribution.
    --------------------------------------------------------------------
    '''

	pdf_vals  = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))

	return pdf_vals

##test case
print()
print('1(b)')
x_test = np.array([[200.0,270.0],[180.0,195.5]])
mu_test = 5
sig_test = 1
result_test = LN_pdf(x_test, mu_test, sig_test)
print ('function testing result is',result_test)



'''
--------------------------------------------------------------------
problem 1(c) SMM
--------------------------------------------------------------------
'''

# Define function that draws N x S test score values from a truncated
# normal distribution
def log_norm_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a log normal. These draws
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
    
    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''

    np.random.seed(seed = 1234)
    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    lognorm_draws = np.exp(norm_draws)
    
    return lognorm_draws


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

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = log_norm_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val


mu_init_1 = 11
sig_init_1 = 0.2
params_init_1 = np.array([mu_init_1, sig_init_1])
W_hat1_1 = np.eye(2)
unif_vals_2 = sts.uniform.rvs(0, 1, size=(200, 300))
smm_args1_1 = (incomes, unif_vals_2, W_hat1_1)
results1_1 = opt.minimize(criterion, params_init_1, args=(smm_args1_1),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM1_1, sig_SMM1_1 = results1_1.x

print()
print('1(c)')
print('mu_SMM1_1= {:.4f}, sig_SMM1_1={:.4f}'.format(mu_SMM1_1, sig_SMM1_1))

mean_data, var_data = data_moments(incomes)
print('Data mean of scores = {:.4f}, Data variance of scores ={:.4f}'.format(mean_data, var_data))

sim_vals_1 = log_norm_draws(unif_vals_2, mu_SMM1_1, sig_SMM1_1)
mean_sim_1, var_sim_1 = data_moments(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
var_model_1 = var_sim_1.mean()
val = criterion(np.array([mu_SMM1_1, sig_SMM1_1]), incomes, unif_vals_2, np.eye(2))[0][0]

print('Model mean 1 = {:.4f}, Model variance 1 = {:.4f}'.format(mean_model_1, var_model_1))
print('value of SMM criterion function at the value of estimate parameter = ', val)

###plot histogram
fig = plt.figure(figsize = (10,10))
count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=20)
plt.xlabel(r'annual incomes(\$)')
plt.ylabel(r'Precent of incomes')
plt.xlim([0, 150000])

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=2, color='k', label='1: $SMM1\mu$ = {:.4f}, $SMM1\sigma$ = {:.4f}'.format(mu_SMM1_1, sig_SMM1_1))
plt.legend(loc='upper left')

output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
#plt.show()
plt.close()



'''
--------------------------------------------------------------------
problem 1(d) Two step weighing matrix
--------------------------------------------------------------------
'''

err2_1 = err_vec(incomes, sim_vals_1, mu_SMM1_1, sig_SMM1_1, False)
VCV2_1 = np.dot(err2_1, err2_1.T) / incomes.shape[0]
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

params_init2_1 = np.array([mu_SMM1_1, sig_SMM1_1])
smm_args2_1 = (incomes, unif_vals_2, W_hat2_1)
results2_1 = opt.minimize(criterion, params_init2_1, args=(smm_args2_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2_1, sig_SMM2_1 = results2_1.x
print()
print('1(d)')
print('mu_SMM2_1= {:.4f}, sig_SMM2_1= {:.4f}'.format(mu_SMM2_1, sig_SMM2_1))
print('Data mean of scores = {:.4f}, Data variance of scores = {:.4f}'.format(mean_data, var_data))

sim_vals_2 = log_norm_draws(unif_vals_2, mu_SMM2_1, sig_SMM2_1)
mean_sim_2, var_sim_2 = data_moments(sim_vals_2)
mean_model_2 = mean_sim_2.mean()
var_model_2 = var_sim_2.mean()
val2 = criterion(np.array([mu_SMM2_1, sig_SMM2_1]), incomes, unif_vals_2, W_hat2_1)[0][0]

print('Model mean 1 = {:.4f}, Model variance 1 = {:.4f}'.format(mean_model_2, var_model_2))
print('value of SMM criterion function at the value of estimate parameter = ', val2)

###plot histogram
fig = plt.figure(figsize = (10,10))
count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=20)
plt.xlabel(r'annual incomes(\$)')
plt.ylabel(r'Precent of incomes')
plt.xlim([0, 150000])

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=2, color='k', label='1: $SMM1\mu$ = {:.4f}, $SMM1\sigma$ = {:.4f}'.format(mu_SMM1_1, sig_SMM1_1))
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2_1, sig_SMM2_1),
         linewidth=1, color='g', label='1: $SMM2\mu$ = {:.4f}, $SMM2\sigma$ = {:.4f}'.format(mu_SMM2_1, sig_SMM2_1))
plt.legend(loc='upper left')

output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
#plt.show()
plt.close()


