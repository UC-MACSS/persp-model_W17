#########################
# Problem Set 4
# MACS 30100, Dr. Evans
# Yinxian Zhang
#########################

import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.integrate as intgr
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
Create an 'images' folder, if not existed, to save picture outputs when running this script
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
Question 1
--------------------------------
Lognormal distribution and SMM
--------------------------------
'''    

# load income data
inc = np.loadtxt('incomes.txt')
# len(inc)

# Q1(a): plot a histogram of the income data
#---------------------------------------------
plot = True   
if plot: 
    bins = 30 # 30 bins
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram using normed=True option. 
    plt.xlim([0, 160000]) # set the two ends of x axis 
    plt.title('Income Distribution of MACS graduates') 
    plt.xlabel('Income(\$)')
    plt.ylabel('Lognormal PDF of incomes')
    
    output = os.path.join(output_dir, 'plot1')
    plt.savefig(output, bbox_inches='tight') 
    plt.show()
    plt.close() 

# Q1(b): write the lognormal PDF function and test it. 
#----------------------------------------------------------

def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate PDF values from a lognormal distribution based on the normal with mean mu and standard deviation sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N, S) matrix, or (N,) vector, in the support of s~N(mu, sig)
    mu     = scalar, mean of the normal distribution from
             which the lognormal is derived
    sigma  = scalar > 0, standard deviation of the normal
             distribution on which the lognormal is based.
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals   = (N, S) matrix or (N,) vector, PDF
                 values corresponding to xvals from lognormal PDF
                 with base normal distribution N(mu, sigma).
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
  
    # pdf_vals = sts.lognorm.pdf(xvals, s=sigma, loc=mu, scale=np.exp(mu))
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))
                    
    return pdf_vals

xvals_test = np.array([[200.0, 270.0], [180.0, 195.5]])
mu_test = 5.0 
sigma_test = 1.0

income_test = LN_pdf(xvals_test, mu_test, sigma_test)

# Q1(c): estimate the lognormal distribution by SMM 
#---------------------------------------------------

def lognorm_draws(mu, sigma, N, S):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a normal distribution with mean mu and standard deviation sigma. Then transform those draws into the lognormal distribution by exponentiating them.
    --------------------------------------------------------------------
    INPUTS:
    mu        = scalar, mean of the normal distribution from which the lognormal is derived
    sigma     = scalar > 0, standard deviation of the 
                normal distribution from which the lognormal is
                derived
    (N, S) =  (N x S) matrix of random draws
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    lognorm_draws = (N, S) matrix or (N,) vector,
                  values drawn from the lognormal PDF with base
                  normal distribution N(mu, sigma).
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: lognorm_draws
    --------------------------------------------------------------------
    '''
    np.random.seed(seed = 1234)
    unif_vals = sts.uniform.rvs(0, 1, (N, S))
    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma) #  values drawn from normal distribution.  
    lognorm_draws = np.exp(norm_draws) # exponentiating them
                           
    return lognorm_draws
    
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), std(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix or (N,) vector, income data, either real world or simulated. 
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of income data
    std_data  = scalar > 0 or (S,) vector, standard deviation of income data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1: # one-dimentional vector with n row. 
        mean_data = xvals.mean()
        std_data = xvals.std()
    elif xvals.ndim == 2: # two-dimentional matrix with n rows and s columns. 
        mean_data = xvals.mean(axis=0) # compute mean for each column along rows, get 1 by S means
        std_data = xvals.std(axis=0) # get 1 by S std
   
    return mean_data, std_data

def err_vec(xvals, sim_vals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, real-world income data
    sim_vals  = (N, S) matrix, S simulations of income data
    mu        = scalar, mean of the normal distribution
                from which the lognormal is derived
    sigma     = scalar > 0, standard deviation of the
                normal distribution from which the lognormal is
                derived
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
    mean_data, std_data = data_moments(xvals)
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
    args      = length 2 tuple,
                (xvals, W_hat)
    xvals     = (N,) vector, values of the normally
                distributed random variable
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        LN_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, SMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W_hat = args
    sim_vals = lognorm_draws(mu, sigma, N, S)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val

# estimate the SMM parameters with a simple Indentity matrix
mu_init = 5.0
sig_init = 1.0
N = 200
S = 300
params_init = np.array([mu_init, sig_init]) 
W_hat = np.eye(2) # use identity matrix as the weighting matrix
smm_args = (inc, W_hat) # plug in data and weighting matrix

results = opt.minimize(criterion, params_init, args=(smm_args), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) # the denominator should not be zero
mu_SMM, sig_SMM = results.x

# get criterion function values at the estimated SMM parameter values
params_SMM = np.array([mu_SMM, sig_SMM])
SMM_Value = criterion(params_SMM, *smm_args)[0][0]

# compare model moments and data moments
sim_vals = lognorm_draws(mu_SMM, sig_SMM, N, S) # get simulated income data
mean_data, std_data = data_moments(inc)
mean_sim, std_sim = data_moments(sim_vals)
mean_mod = mean_sim.mean() # get average of simulation means
std_mod = std_sim.mean()

# plot lognormal PDF against the histogram
inc_sim = np.linspace(0.0001, 150000, 200) # x should be strictly positive in the PDF function, therefore the lower bound is larger than 0 
if plot:
    bins = 30 
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, LN_pdf(inc_sim, mu_SMM, sig_SMM), linewidth=2, color='r', label='1: $\mu=\mu_{SMM}$, $\sigma=\sigma_{SMM}$') # plot the lognormal PDF with the SMM estimated parameters.
 
    plt.title('SMM Estimated Distribution and the Histrogram of Income')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot2')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 
    
 
# Q1(d): two-step SMM estimator 
#------------------------------

# step 1: use vector of moment error function from part (c) to get a new estimate of the inverse variance-covariance matrix
err = err_vec(inc, sim_vals, mu_SMM, sig_SMM, False) # plug in SMM estimated parameters to get error vector 
VCV = np.dot(err, err.T) / inc.shape[0] # calculate the sum of squared errors. Note that variance covariance matrix should be devided by N, the number of observations. 
W_hat2 = lin.pinv(VCV)  # Use the pseudo-inverse calculated by SVD to get the two-step optimal weighting matrix 

#step 2: re-estimate the SMM estimator using the optimal two-step weighting matrix.
params_init_2 = np.array([mu_SMM, sig_SMM]) 
smm_args2 = (inc, W_hat2) # plug in the optimal weighting matrix
results2 = opt.minimize(criterion, params_init_2, args=(smm_args2), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) # run once again the minimizer
mu_SMM2, sig_SMM2 = results2.x # get the two-step estimated parameters

# get criterion function values at the estimated SMM parameter values
params_SMM2 = np.array([mu_SMM2, sig_SMM2])
SMM_Value2 = criterion(params_SMM2, *smm_args2)[0][0]

# compare simulation moments and data moments
sim_vals2 = lognorm_draws(mu_SMM2, sig_SMM2, N, S)
mean_data, std_data = data_moments(inc)
mean_sim2, std_sim2 = data_moments(sim_vals2)
mean_mod2 = mean_sim2.mean()
std_mod2 = std_sim2.mean()

# plot two-step estimated lognormal PDF against the PDF in part c and the histogram in part a
if plot:
    bins = 30
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, LN_pdf(inc_sim, mu_SMM, sig_SMM), linewidth=5, color='r', label='1: $\mu=\mu_{SMM}$, $\sigma=\sigma_{SMM}$') # plot the lognormal PDF with the SMM estimated parameters with identity matrix
    
    plt.plot(inc_sim, LN_pdf(inc_sim, mu_SMM2, sig_SMM2), linewidth=2, color='k', label='2: $\mu=\mu_{2step}$, $\sigma=\sigma_{2step}$') # plot the lognormal PDF with the SMM estimated parameters with two-step optimal weighting matrix
 

    plt.title('Identity matrix v.s. two-step optimal weighting matrix SMM estimations')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot3')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 

if __name__ == '__main__':   
    print()
    print ('BEGINNING OF PROBLEM SET 4')
    print()
    print ('MACS 30100, Dr. Evans')
    print ('Yinxian Zhang')
    print()
    print ('Q1(a)')
    print('The histogram is saved')
    print()
    print('Q1(b)')
    print('The lognormal PDF values of the test matrix with mu=5 and sigma=1 is: ', income_test)
    print()
    print('Q1(c)')
    print('The SMM estimated mean and std are:', mu_SMM, sig_SMM)
    print('The value of SMM criterion function at the estimated parameter values is: ', SMM_Value)
    print('Two data moments and two model moments at the estimated parameter values are: ', np.array([mean_data, std_data]), np.array([mean_mod, std_mod]))
    print()
    print('Q1(d)')
    print('The value of SMM criterion function at the estimated parameter values is: ', SMM_Value2 )
    print('The SMM estimated mean and std are:', mu_SMM2, sig_SMM2)
    print('Two data moments and two model moments at the estimated parameter values are: ', np.array([mean_data, std_data]), np.array([mean_mod2, std_mod2]))
    print()
    print ('END OF PROBLEM SET 4')