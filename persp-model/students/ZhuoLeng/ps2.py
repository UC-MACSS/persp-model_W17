## ps2

## Zhuo Leng

import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt


'''
--------------------------------------------------------------------
Excercise 1 function needed
--------------------------------------------------------------------
'''

def lognorm_cdf(xvals, mu, sigma):
	

    cdf_vals = sts.lognorm.cdf(xvals, loc = mu, scale = np.exp(mu), s = sigma)
    return cdf_vals


def log_norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values with mean mu and standard deviation sigma. 

    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormally distributed random
             variable
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    --------------------------------------------------------------------
    '''
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * 
    	np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))
    return pdf_vals
 

def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given lognormal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    --------------------------------------------------------------------
    '''
    pdf_vals = log_norm_pdf(xvals, mu, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val
 

def crit_incomes(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable

    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_lognorm(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


'''
--------------------------------------------------------------------
Excercise 1(a) Plot a histogram
--------------------------------------------------------------------
'''

incomes = np.loadtxt('incomes.txt')

plot_1a = True
if plot_1a:

    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath('ps2.py'))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    #plot histogram
    num_bins = 30
    weights = (1 / len(incomes) * np.ones_like(incomes))
    n, bin_cuts, patches = plt.hist(incomes, num_bins, weights=weights)
    plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
    plt.xlabel(r'annual incomes(\$s)')
    plt.ylabel(r'Percent of annual incomes of students')

    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

'''
--------------------------------------------------------------------
Excercise 1(b) Log PDF and log likelihood
--------------------------------------------------------------------
'''

dist_pts = np.linspace(0, 150000, 150000)
mu_1 = 9.0
sig_1 = 0.3
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1, sig_1),
         linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
plt.xlim(0,150000)
plt.xlabel('x')
plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
plt.title('Log normal pdf')
plt.legend(loc= 'upper left')

output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
# plt.show()
plt.close()

print("Excercise 1b")
print('Log-likelihood value: ', log_lik_lognorm(incomes, mu_1, sig_1))
print()


'''
--------------------------------------------------------------------
Excercise 1(c) MLE
--------------------------------------------------------------------
'''

#In order to get vcv_mle and mle estimator, we need to run minimize function twice using 
#different method. The first time, we use SLSQP method to get result of new params.
#Then we use these params as params_init to run minimize function again using BFGS method.

mu_init = 10  
sig_init = 0.2 
params_init = np.array([mu_init, sig_init])
mle_args = (incomes, 150000)
bnds = ((None, None), (0.00001, None))
results = opt.minimize(crit_incomes, params_init, args=(mle_args), method = 'SLSQP', bounds = bnds)

params_init = results.x
results = opt.minimize(crit_incomes, params_init, args=(mle_args), method = 'BFGS', bounds = bnds)
mu_MLE, sig_MLE = results.x
vcv_mle = results.hess_inv


print("Excercise 1c")
print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
print('MLE log-likelihood : ', log_lik_lognorm(incomes, mu_MLE, sig_MLE))
print('variance-covariance matrix', vcv_mle)
print()


plot_1c = True

if plot_1c:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    xx          = (45,) vector, values of xx
    yy          = (45,) vector, values of yy
    --------------------------------------------------------------------
    '''
    fig = plt.figure()
    plt.hist(incomes, 30, normed = True)
    plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=16)
    plt.xlabel(r'annual incomes')
    plt.ylabel(r'Percent of annual incomes of students')
    plt.xlim(0,150000)


    # Plot the test distributions from before
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1, sig_1),
    linewidth=2, color='r', label= '1: $\mu$ = 9.0,$\sigma$ = 0.3')
    plt.legend(loc='upper right')

    # Plot the MLE estimated distribution
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_MLE, sig_MLE),
    linewidth=2, color='g', label='2: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')

    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    #plt.show()
    plt.close()


'''
--------------------------------------------------------------------
Excercise 1d likelihood ratio test
--------------------------------------------------------------------
'''

log_lik_h0 = log_lik_lognorm(incomes, mu_1, sig_1)
log_lik_mle = log_lik_lognorm(incomes, mu_MLE, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)

print('Excercise 1d')
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print()



'''
--------------------------------------------------------------------
Excercise 1e 
--------------------------------------------------------------------
'''

p1 = 1 - lognorm_cdf(100000, mu_MLE, sig_MLE)
p2 = lognorm_cdf(75000, mu_MLE, sig_MLE)
print ('Excercise 1e')
print("Probability that you will earn more than $100,000 is {:.3f}".format(p1))
print("Probability that you will earn less than $75,000 is {:.3f}".format(p2))
print()


'''
--------------------------------------------------------------------
Excercise 2 function needed
--------------------------------------------------------------------
'''

def norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    --------------------------------------------------------------------
    '''

    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2)))) 
    
    return pdf_vals

def log_lik_norm(df, b0, b1, b2, b3, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data df given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    b0, b1, b2, b3 = scalar, parameters
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
   
    --------------------------------------------------------------------
    '''
    error = df['sick'] - b0 - b1*df['age'] - b2*df['children'] - b3*df['temp']
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

def crit_sick(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable

    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3,sigma = params
    df = args[0]
    log_lik_val = log_lik_norm(df, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val 

'''
--------------------------------------------------------------------
Excercise 2a MLE
--------------------------------------------------------------------
'''

sick= pd.read_csv('sick.txt', header = 0, names = ['sick','age','children',
	'temp'])
#set initial estimate value for parameters
#run minimization function twice
b0_init = 1
b1_init = 0
b2_init = 0
b3_init = 0 
sig_init = 1
params_init = np.array([b0_init, b1_init, b2_init, b3_init, sig_init])
bnds = ((None, None),(None, None),(None, None),(None, None),(0.00001, None))
results = opt.minimize(crit_sick, params_init, args=sick, method = 'SLSQP', bounds = bnds)

params_init = results.x
results = opt.minimize(crit_sick, params_init, args=sick, method = 'BFGS', bounds = bnds)
b0_MLE, b1_MLE, b2_MLE, b3_MLE, sigma_MLE = results.x
vcv_mle = results.hess_inv

print( 'Excercise 2a')
print('Beta0 =', b0_MLE, ' Beta1 =', b1_MLE, 'Beta2 =', b2_MLE, 'Beta3 =', b3_MLE, 'Sigma =', sigma_MLE)
print('The variance-covariance matrix is: ',vcv_mle)
print('Log-likelihood: ', log_lik_norm(sick, *results.x))
print()


'''
--------------------------------------------------------------------
Excercise 2b likelihood ratio test
--------------------------------------------------------------------
'''

log_lik_h0 = log_lik_norm(sick,1 ,0, 0, 0, 0.01)
log_lik_mle = log_lik_norm(sick, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sigma_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)

print('Excercise 2b')
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)














