
#Yuqing Zhang
#Problem Sets 2

import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import os
import math
import scipy.optimize as opt

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

income = np.loadtxt('incomes.txt')

'''
------------------------------------------------
Exercise 1a: Plot a histogram of the income data
------------------------------------------------
'''
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1a, ax1a = plt.subplots()

plt.hist(income, num_bins, weights = weights)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')
    
output_path = os.path.join(output_dir, '1a_inc_hist')
plt.savefig(output_path)
#plt.show()
plt.close



'''
------------------------------------------------------------
Exercise 1b: Plot the log normal PDF for mu=9 and sigma=0.3 
------------------------------------------------------------
'''
sigma = 0.3
mu = 9.0
cutoff = 150000

def log_norm_pdf(xvals, mu_init, sigma_init, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu_init     = scalar, mean of the normally distributed random variable
    sigma_init  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu_init, scale=sigma_init)
            
    log_pdf_vals    = ((1/(xvals*sigma_init * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu_init)**2 / (2 * sigma_init**2))) /
                    prob_notcut)
    
    return log_pdf_vals

dist_pts = np.linspace(0.000001, 150000, 150000)
ln_dist_pts=np.log(dist_pts)
fig1b, ax1b = plt.subplots()

plt.plot(dist_pts, log_norm_pdf(dist_pts, mu, sigma, cutoff),
         linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
plt.title('Log normal PDF plot', fontsize=20)
plt.xlabel(r'Annual income (\$s)')
plt.ylabel(r'Percent of students')

plt.legend(loc='upper right')
    
output_path = os.path.join(output_dir, '1b_lognormal')
plt.savefig(output_path)
#plt.show()
plt.close

def log_lik_norm(x_vals, mu_init, sigma_init,cutoff):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu_init     = scalar, mean of the normally distributed random variable
    sigma_init  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
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

    pdf_vals = log_norm_pdf(x_vals,mu_init,sigma_init,cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val
print('Exercise 1b:')
print('Log-likelihood for mu=9,sigma=0.3 is: ', log_lik_norm(income, mu, sigma,cutoff))
print()

'''
------------------------------------------------------------
Exercise 1c: Compare 3 plots 
------------------------------------------------------------
'''
print('Exercise 1c:')
def crit(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu_init, sigma_init])
    mu_init     = scalar, mean of the normally distributed random variable
    sigma_init  = scalar > 0, standard deviation of the normally distributed
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
    mu_init, sigma_init = params
    xvals,cutoff= args
    log_lik_val = log_lik_norm(xvals, mu_init, sigma_init,cutoff)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val



mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
mle_args = (income, cutoff)
results=opt.minimize(crit,params_init,args=(mle_args),bounds=((None,None), (0.00000000001,None))) 
mu_MLE, sig_MLE = results.x
print('ML estimate for mu is:', mu_MLE)
print('ML estimate for sigma is:', sig_MLE)
vcv_mle = results.hess_inv.todense()
stderr_mu_mle = np.sqrt(vcv_mle[0,0])
stderr_sig_mle = np.sqrt(vcv_mle[1,1])
print('The value for the maximum likelihood function is:',log_lik_norm(income, mu_MLE, sig_MLE,cutoff))
print('The variance-covariance matrix is:')
print(vcv_mle)
print('Standard error for mu estimate = ', stderr_mu_mle)
print('Standard error for sigma estimate = ', stderr_sig_mle)



# Plot the histogram of the data
fig1c,ax1c=plt.subplots()
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
plt.hist(income, num_bins, weights = weights,normed=1)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

# Plot the two test distributions from before
dist_pts = np.linspace(0.000001, 150000, 150000)
ln_dist_pts=np.log(dist_pts)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu, sigma, cutoff),
         linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper right')

# Plot the MLE estimated distribution
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_MLE, sig_MLE, cutoff),
         linewidth=2, color='k', label='2: $\mu$=mu_MLE,$\sigma$=sig_MLE')
plt.legend(loc='upper right')

output_path = os.path.join(output_dir, '1c_compare')
plt.savefig(output_path)
#plt.show()
plt.close
print()

'''
------------------------------------------------------------
Exercise 1d: likelihood ratio test
------------------------------------------------------------
'''
print('Exercise 1d:')
log_lik_h0 = log_lik_norm(income, mu, sigma, cutoff)
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, cutoff)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)

print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print('p-value equals to 0, which provides strong evidence against null hypothesis. So we can reject the hypothesis that the data in incomes.txt came from distribution in part b.')
print()
'''
------------------------------------------------------------
Exercise 1e: Estimate probability using MLE
------------------------------------------------------------
'''
probility1 = sts.lognorm.cdf(75000, s=sig_MLE, loc=mu_MLE, scale=np.exp(mu_MLE))#probility of earning less than 75,000
probility2 = 1 - sts.lognorm.cdf(100000, s=sig_MLE, loc=mu_MLE, scale=np.exp(mu_MLE))
print('Exercise 1e:')
print('The probability that I will earn more than $100,000:',round(probility2,4)*100,'%')
print('The probability that I will earn less than $75,000:',round(probility1,4)*100,'%')

print()



sick = np.loadtxt('sick.txt',skiprows=1,delimiter=',')
sick_i = sick[:,0]
age_i = sick[:,1]
children_i=sick[:,2]
temp_winter_i=sick[:,3]

'''
------------------------------------------------------------
Exercise 1a: Estimate the parameters of a linear regression
------------------------------------------------------------
'''

def norm_pdf_2(xvals,sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu=0 and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, normal PDF values for mu=0 and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''   
    pdf_vals = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp(-(xvals)**2 / (2 * sigma**2)))  
   
    return pdf_vals



def log_lik_norm_2(xvals,sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf_2()
    
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
    pdf_vals = norm_pdf_2(xvals,sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val



def crit_2(params,*args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (5,) vector, ([sigma,beta_0,beta_1,beta_2,beta_3])
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    args   = length 4 tuple, (sick_i,age_i,children_i,temp_winter_i)
    xvals  = (N,) vector, values of the normally distributed random
             variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_norm_2()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''    
    sigma,beta_0,beta_1,beta_2,beta_3 = params
    sick_i,age_i,children_i,temp_winter_i = args
    xvals = sick_i - beta_0-beta_1*age_i-beta_2*children_i-beta_3*temp_winter_i

    log_lik_val = log_lik_norm_2(xvals,sigma)

    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val



import scipy.optimize as opt

sig_init = 1
beta_0_init=0.1
beta_1_init=0.1
beta_2_init=0.1
beta_3_init=0.1

params_init = np.array([sig_init,beta_0_init,beta_1_init,beta_2_init,beta_3_init])
bound = ((0.000001, None),(None, None),(None, None), (None, None), (None, None))
mle_args = (sick_i, age_i, children_i, temp_winter_i) 

results = opt.minimize(crit_2, params_init,args=(mle_args), bounds=bound,method='SLSQP')

sig_MLE,beta_0_MLE,beta_1_MLE,beta_2_MLE,beta_3_MLE = results.x
#Now that we know the MLE of parameters, we use another method to generate the variance-covariance matrix
params_init2 = np.array([sig_MLE,beta_0_MLE, beta_1_MLE, beta_2_MLE, beta_3_MLE])
results2 = opt.minimize(crit_2, params_init2, args=(mle_args), bounds=bound, method='L-BFGS-B')
vcv_mle_2 = results2.hess_inv.todense() 

print('Exercise 2a')
print('sigma_mle:', sig_MLE)
print('beta_0_mle:',beta_0_MLE)
print('beta_1_mle:',beta_1_MLE)
print('beta_2_mle:',beta_2_MLE)
print('beta_3_mle:',beta_3_MLE)
xvals_mle=sick_i - beta_0_MLE-beta_1_MLE*age_i-beta_2_MLE*children_i-beta_3_MLE*temp_winter_i

print('The value of the maximized loglikelihood function:',log_lik_norm_2(xvals_mle,sig_MLE))
print('The estimated variable-covariance matrix is:')
print(vcv_mle_2)
print()
'''
------------------------------------------------------------
Exercise 2b: likelihood ratio test
------------------------------------------------------------
'''
print('Exercise 2b:')

beta_0_h0=1
beta_1_h0=0
beta_2_h0=0
beta_3_h0=0
sigma_h0=math.sqrt(0.01)
xvals_h0= sick_i - beta_0_h0-beta_1_h0*age_i-beta_2_h0*children_i-beta_3_h0*temp_winter_i
log_lik_h0_2 = log_lik_norm_2(xvals_h0,sigma_h0)
log_lik_mle_2 = log_lik_norm_2(xvals_mle,sig_MLE)
LR_val_2 = 2 * (log_lik_mle_2 - log_lik_h0_2)
pval_h0_2 = 1.0 - sts.chi2.cdf(LR_val_2, 2)

print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0_2)
print('p-value equals to 0, which provides strong evidence against null hypothesis. So we can reject the hypothesis that age,number of children,and average winter temperature have on effect on the  number of sick days.')





