import numpy as np
import pandas as pd
import scipy.stats as sts
from scipy.stats import lognorm
import os
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize as opt

'''
lognorm(s =sig_1, scale =np.exp(mu_1)).pdf(dist_pts)
'''
# Define function that generates values of a normal pdf
def norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''       
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    
    return pdf_vals


# Define log likelihood function for the log normal distribution
def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    
    pdf_vals = lognorm(s =sigma, scale =np.exp(mu)).pdf(xvals)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

def crit(params, *args):
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_lognorm(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
'''
--------------------------------------------------------------------
Question 1: a
--------------------------------------------------------------------
'''
data= np.loadtxt('incomes.txt')
hist = True

if hist:
	#Create directory if images directory does not already exist
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)

	fig = plt.figure()


	plt.hist(data, bins = 30, normed = True)

	plt.title('Histogram of incomes of MACSS graduates', fontsize=15)
	plt.xlabel('Annual Income($)')
	plt.ylabel('Percent of observations in bin')
	output_path = os.path.join(output_dir, 'Fig_1a')
	plt.savefig(output_path)
	#plt.show()
	plt.close()
'''
--------------------------------------------------------------------
Question 1: b
--------------------------------------------------------------------
'''
plot_b = True

if plot_b:
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)

	dist_pts = np.linspace(0, 150000, 500)
	mu_1 = 9.0
	sig_1 = 0.3
	fig = plt.figure()
	#plt.hist(data, bins = 30, normed = True)#, weights=weights)
	plt.plot(dist_pts, lognorm(s =sig_1, scale =np.exp(mu_1)).pdf(dist_pts),linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
	plt.legend(loc='upper right')
	plt.title('Lognormal pdf with $\mu$=9.0 and $\sigma$=0.3', fontsize=15)
	plt.tight_layout()
	plt.xlim(0, 150000)
	plt.xlabel('x')
	plt.ylabel('y')
	
	output_path = os.path.join(output_dir, 'Fig_1b')
	plt.savefig(output_path)
	#plt.show()
	plt.close()

#the log likelihood value for this parameterization of the distribution and given this data
print('Log-likelihood: ', log_lik_lognorm(data, mu_1, sig_1))

'''
--------------------------------------------------------------------
Question 1: c
--------------------------------------------------------------------
'''
#Part c:
#Find MLE
'''
Note: When I use the minimizer function from scipy without specifying the method that I want to use,
I could not get a stable result. It depends greatly on the initial values that I put in for the parameters.
After specifying "SLSQP" method, I get reasonably results;however, this method does not provide the 
variance-covariance matrix. Thus, I used my estimates of "SLSQP" method as initial values and run 
the minimizer again with "BFGS" method to get the matrix. These are also the resaons for my approach
for question 2(a).
'''
mu_init = 9
sig_init = 0.5
params_init = np.array([mu_init, sig_init])
mle_args = (data, data.max())
results = opt.minimize(crit, params_init, args=(mle_args), method = "SLSQP", bounds=((None, None), (0.001, None)))
#mu_MLE, sig_MLE = results.x

params_init = results.x
results = opt.minimize(crit, params_init, args=(mle_args), method = "BFGS", bounds=((None, None), (0.001, None)))
mu_MLE, sig_MLE = results.x
vcv_mle = results.hess_inv
#stderr_mu_mle = np.sqrt(vcv_mle[0,0])
#stderr_sig_mle = np.sqrt(vcv_mle[1,1])
print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
print('VCV(MLE) = ', vcv_mle)
#print('Standard error for mu estimate = ', stderr_mu_mle)
#print('Standard error for sigma estimate = ', stderr_sig_mle)
print('Log-likelihood of MLE estimate: ', log_lik_lognorm(data, mu_MLE, sig_MLE))
#print(results)

plot_c = True
if plot_c:
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)

	mu_1 = 9.0
	sig_1 = 0.3
	fig = plt.figure()
	plt.hist(data, bins = 30, normed = True)#, weights=weights)
	plt.plot(dist_pts, lognorm(s =sig_1, scale =np.exp(mu_1)).pdf(dist_pts),linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
	plt.plot(dist_pts, lognorm(s =sig_MLE, scale =np.exp(mu_MLE)).pdf(dist_pts),linewidth=2, color='G', label='2: $\mu$ and $\sigma$ estimated by MLE')
	plt.legend(loc='upper right')
	plt.title('Histogram of incomes of MACSS graduates', fontsize=15)
	plt.xlabel('Annual Income($)')
	plt.ylabel('Percent of observations in bin')
	output_path = os.path.join(output_dir, 'Fig_1c')
	plt.savefig(output_path)
	plt.close()

'''
--------------------------------------------------------------------
Question 1: d
--------------------------------------------------------------------
'''
#perform a ratio test to determine the probability that the data in incomes.txt 
#came from the distribution in part b

log_lik_h0 = log_lik_lognorm(data, mu_1, sig_1)
log_lik_mle = log_lik_lognorm(data, mu_MLE, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = {:.5}'.format(pval_h0))


'''
--------------------------------------------------------------------
Question 1: e
--------------------------------------------------------------------
'''
#P(income>100000):
P_1 = lognorm.sf(100000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print("Probability that you will earn more than $100,000 is", P_1)
#P(income<75000):
P2 = lognorm.cdf(75000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print("Probability that you will earn less than $75,000 is", P2)

'''
--------------------------------------------------------------------
Question 2: 
--------------------------------------------------------------------
'''
sick_data= pd.read_csv('sick.txt', header = 0)
def norm_pdf(xvals, mu, sigma):        
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) 
    
    return pdf_vals

# Define log likelihood function
def log_like_sick(data, b0, b1, b2, b3, sigma):
    error = data['sick']- (b0+ b1*data['age'] + b2*data['children']+ b3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

def crit_sick(params, *args):
    b0, b1, b2, b3,sigma = params
    data = args[0]
    log_lik_val = log_like_sick(data, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
'''
--------------------------------------------------------------------
Question 2: a
--------------------------------------------------------------------
'''
#Find MLE
beta0_init = 1
beta1_init = 0
beta2_init = 0
beta3_init = 0
sig_init = .1
params_init = np.array([beta0_init, beta1_init, beta2_init, beta3_init, sig_init])
bounds = ((None, None),(None, None),(None, None),(None, None),(0.001, None))
results = opt.minimize(crit_sick, params_init, args=sick_data, method = "SLSQP", bounds=bounds)
b0, b1, b2, b3, sigma = results.x
results = opt.minimize(crit_sick,  results.x, args=sick_data, method = "L-BFGS-B", bounds=bounds)
b0, b1, b2, b3, sigma = results.x
print('Beta0 =', b0, ' Beta1 =', b1, 'Beta2 =', b2, 'Beta3 =', b3, 'Sigma =', sigma)
print('The variance-covariance matrix is: ',results.hess_inv.todense())
print('Log-likelihood: ', log_like_sick(sick_data, *results.x))

'''
--------------------------------------------------------------------
Question 2: b
--------------------------------------------------------------------
'''

log_lik_h0 = log_like_sick(sick_data, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_sick(sick_data, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('chi squared of H0 with 5 degrees of freedom p-value = {:.5}'.format(pval_h0))

