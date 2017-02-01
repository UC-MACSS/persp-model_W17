#########################
# Problem Set 3
# MACS 30100, Dr. Evans
# Yinxian Zhang
#########################

import numpy as np
import scipy as sp
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
-----------------------------------------
Lognormal distribution of some income data
------------------------------------------
'''    

# load income data
inc = np.loadtxt('income.txt')
# len(inc)

# Q1(a): plot a histogram of the income data
#---------------------------------------------
plot = True   
if plot: 
    bins = 30 # 30 bins
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    plt.xlim([40000, 150000]) # set the two ends of x axis 
    plt.title('Income Distribution of MACS graduates') 
    plt.xlabel('Income(\$)')
    plt.ylabel('Lognormal PDF of income')
    
    output = os.path.join(output_dir, 'plot1')
    plt.savefig(output, bbox_inches='tight') 
    plt.show()
    plt.close() 

# Q1(b): estimate the lognormal distribution parameters by GMM
#----------------------------------------------------------

def trunc_lognorm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values for the lognormal distribution with mean mu and standard deviation sigma. If the cutoff is given, then the PDF values are inflated upward to reflect the zero probability on values above the
    cutoff. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the log normally distributed random
             variable
    mu     = scalar, mean of the log normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0 
    else:
        prob_notcut = sts.lognorm.cdf(cutoff, s=sigma, loc=mu, scale=np.exp(mu)) # use lognormal CDF function
            
    pdf_vals = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut) # log normal distribution function 
    
    return pdf_vals 


def data_moments(xvals): ## compute mean and standard deviation statistics of the real data
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    std_data  = scalar > 0, standard deviation of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()
    
    return mean_data, std_data


def model_moments(mu, sigma, cutoff): # compute mean and standard deviation statistics of model 
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    std_model  = scalar > 0, standard deviation of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * trunc_lognorm_pdf(x, mu, sigma, cutoff)
    (mean_model, m_m_err) = intgr.quad(xfx, 0.000001, cutoff) # xvals should not be zero in lognormal PDF function
    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_lognorm_pdf(x, mu, sigma, cutoff) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0.000001, cutoff)
    std_model = np.sqrt(var_model)
    
    return mean_model, std_model


def err_vec(xvals, mu, sigma, cutoff, simple): # call data and model statistics to compute error
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]]) # data moments
    mean_model, std_model = model_moments(mu, sigma, cutoff)
    moms_model = np.array([[mean_model], [std_model]]) # model moments
    if simple: 
        err_vec = moms_model - moms_data # simple difference between data and model moments
    else:
        err_vec = (moms_model - moms_data) / moms_data # percent deviation between data and model moments
    
    return err_vec

def criterion(params, *args): # compute the criterion function
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
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
    xvals, cutoff, W = args
    err = err_vec(xvals, mu, sigma, cutoff, simple=False) # get error variance-covariance matrix
    crit_val = np.dot(np.dot(err.T, W), err)  # plug in weighting matrix
    
    return crit_val

# estimate the GMM parameters with a simple Indentity matrix
mu_init = 11
sig_init = 0.3
params_init = np.array([mu_init, sig_init]) # set up inital guess of parameters

W_hat = np.eye(2) # two by two identity matrix
gmm_args = (inc, 150000.0, W_hat) # plug in data and weighting matrix
results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) # the divided number should not be zero
mu_GMM1, sig_GMM1 = results.x

# get criterion function values at the estimated GMM parameter values
params_GMM = np.array([mu_GMM1, sig_GMM1])
GMM_Value = criterion(params_GMM, *gmm_args)[0][0]

# compare model moments and data moments
mean_data, std_data = data_moments(inc)
mean_model, std_model = model_moments(mu_GMM1, sig_GMM1, 150000)

# plot lognormal PDF against the histogram
inc_sim = np.linspace(0.0001, 150000, 200) # simulate income data, x should be strictly positive in the PDF, therefore the lower bound is larger than 0 
if plot:
    bins = 30 
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM1, sig_GMM1, 150000), linewidth=2, color='r', label='1: $\mu$=11.33359685,$\sigma$=0.213255509') # plot the lognormal PDF with the GMM estimated parameters and cutoff of 150000
 
    plt.title('GMM Estimated Distribution and the Histrogram of Income')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot2')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 
    
 
# Q1(c): two-step GMM estimator 
#------------------------------

# step 1: use vector of moment error function from part b to get a new estimate of the inverse variance-covariance matrix
err1 = err_vec(inc, mu_GMM1, sig_GMM1, 150000, False) # plug in GMM estimated parameters to get error vector 
VCV2 = np.dot(err1, err1.T) / inc.shape[0] # calculate the sum of squared errors 
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD to get the two-step optimal weighting matrix 

#step 2: re-estimate the GMM estimator using the optimal two-step weighting matrix.
params_init_2 = np.array([mu_GMM1, sig_GMM1]) 
gmm_args2 = (inc, 150000.0, W_hat2) # plug in the optimal weighting matrix
results2 = opt.minimize(criterion, params_init_2, args=(gmm_args2), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) # run once again the minimizer
mu_GMM2, sig_GMM2 = results2.x # get the two-step estimated parameters

# get criterion function values at the estimated GMM parameter values
params_GMM2 = np.array([mu_GMM2, sig_GMM2])
GMM2_Value = criterion(params_GMM2, *gmm_args2)[0][0]

# compare model moments and data moments
mean_data, std_data = data_moments(inc)
mean_model2, std_model2 = model_moments(mu_GMM2, sig_GMM2, 150000)

# plot two-step estimated lognormal PDF against the PDF in part b and the histogram in part a
if plot:
    bins = 30
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM1, sig_GMM1, 150000), linewidth=5, color='r', label='1: $\mu$=11.33359685,$\sigma$=0.213255509') # plot the lognormal PDF with the GMM estimated parameters with identity matrix
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM2, sig_GMM2, 150000), linewidth=2, color='k', label='2: $\mu$=11.333596294,$\sigma$=0.213255452') # plot the lognormal PDF with the GMM estimated parameters with two-step optimal weighting matrix
 
 
    plt.title('Identity matrix v.s. two-step optimal weighting matrix GMM estimations')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot3')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 


# Q1(d): GMM estimation with three moments
#------------------------------------------

def data_3moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 75,000
    bpct_2_dat = scalar in [0, 1], percent of observations
                 75,000 <= x < 100,000
    bpct_3_dat = scalar in [0, 1], percent of observations
                 x > 100,000
                 
    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1 = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2 = (xvals[(xvals >=75000) & (xvals <= 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3 = xvals[xvals > 100000].shape[0] / xvals.shape[0]
    
    return bpct_1, bpct_2, bpct_3

def model_3moments(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the three model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_lognorm_pdf()
        xfx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_mod = scalar in [0, 1], percent of model observations in
                 bin 1
    bp_1_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_1_mod
    bpct_2_mod = scalar in [0, 1], percent of model observations in
                 bin 2
    bp_2_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_2_mod
    bpct_3_mod = scalar in [0, 1], percent of model observations in
                 bin 3
    bp_3_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_3_mod
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod
    --------------------------------------------------------------------
    '''
    xfx = lambda x: trunc_lognorm_pdf(x, mu, sigma, cutoff)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0.00001, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 150000)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_3vec(xvals, mu, sigma, cutoff, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments4()
        model_moments4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    moms_data  = (3, 1) matrix, column vector of three data moment
    moms_model = (3, 1) matrix, column vector of three model moments
    err_vec    = (3, 1) matrix, column vector of three moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_3moments(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_3moments(mu, sigma, cutoff)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec


def criterion3(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_3vec()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (3, 1) matrix, column vector of three moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_3vec(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val    

# GMM estimation 
mu_init = 11
sig_init = 0.3
params_init = np.array([mu_init, sig_init]) # set up inital parameters

W_hat3 = np.eye(3) # use simple identity matrix as the weighting matrix for GMM estimation
gmm_args3 = (inc, 150000.0, W_hat3) # plug in data and weighting matrix
results3 = opt.minimize(criterion3, params_init, args=(gmm_args3),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) 
mu_GMM1_3, sig_GMM1_3 = results3.x

# get criterion function values at the estimated GMM parameter values
params_GMM3 = np.array([mu_GMM1_3, sig_GMM1_3])
GMM3_Value = criterion3(params_GMM3, *gmm_args3)[0][0]

# compare model moments and data moments
bpct_1, bpct_2, bpct_3 = data_3moments(inc)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_3moments(mu_GMM1_3, sig_GMM1_3, 150000)

# plot the lognormal PDF against the histogram in part a
if plot:
    bins = 30
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM1_3, sig_GMM1_3, 150000), linewidth=2, color='r', label='1: $\mu$=11.336705158,$\sigma$=0.211513244') # plot the lognormal PDF with the GMM estimated parameters and cutoff of 150000
    
 
    plt.title('GMM estimated income distribution using three moments')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot4')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 

# Q1(e): two-step GMM estimator with three moments. 
#------------------------------------------------------------------------
# step 1: use vector of moment error function from part b to get a new estimate of the inverse variance-covariance matrix
err1_3 = err_3vec(inc, mu_GMM1_3, sig_GMM1_3, 150000, False) # plug in GMM estimated parameters to get error vector 
VCV2_3 = np.dot(err1_3, err1_3.T) / inc.shape[0] # calculate the sum of squared errors 
W_hat2_3 = lin.pinv(VCV2_3)  # Use the pseudo-inverse calculated by SVD to get the two-step optimal weighting matrix 

#step 2: re-estimate the GMM estimator using the optimal two-step weighting matrix.
#params_init = np.array([mu_init, sig_init]) 
params_init_3 = np.array([mu_GMM1_3, sig_GMM1_3]) 
gmm_args2_3 = (inc, 150000.0, W_hat2_3) # plug in the optimal weighting matrix
results3_2 = opt.minimize(criterion3, params_init_3, args=(gmm_args2_3), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None))) # run once again the minimizer
mu_GMM2_3, sig_GMM2_3 = results3_2.x # get the two-step estimated parameters

# get criterion function values at the estimated GMM parameter values
params_GMM2_3 = np.array([mu_GMM2_3, sig_GMM2_3])
GMM4_Value = criterion3(params_GMM2_3, *gmm_args2_3)[0][0]

# compare model moments and data moments
bpct_1, bpct_2, bpct_3 = data_3moments(inc)
bpct_1_mod2, bpct_2_mod2, bpct_3_mod2 = model_3moments(mu_GMM2_3, sig_GMM2_3, 150000)

# plot two-step estimated lognormal PDF against the PDF in part d and the histogram in part a
if plot:
    bins = 30
    n, bins, ignored = plt.hist(inc, bins=bins, normed=True) # plot histogram   
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM1_3, sig_GMM1_3, 150000), linewidth=5, color='r', label='1: $\mu$=11.336705158,$\sigma$=0.211513244') # plot the lognormal PDF estimated with three moments and identity matrix
    
    plt.plot(inc_sim, trunc_lognorm_pdf(inc_sim, mu_GMM2_3, sig_GMM2_3, 150000), linewidth=2, color='k', label='2: $\mu$=11.336778366,$\sigma$=0.211591543') # plot the lognormal PDF estimated with three moments and optimal weighting matrix
 
 
    plt.title('GMM estimations of income distribution using three moments')
    plt.xlabel('Income(\$)')
    plt.ylabel('Lognormal PDF')
    plt.legend(loc='upper left')
    
    output = os.path.join(output_dir, 'plot5')
    plt.savefig(output, bbox_inches='tight')  
    plt.show()
    plt.close() 

 
'''
Question 2
---------------------------
Linear regression and GMM
---------------------------
'''    

# load data
df = pd.read_csv('sick.txt', header=0)


def err_vec_lin(xvals, b0, b1, b2, b3): # call data and model statistics to compute error matrix
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in simple difference) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
   
    
    OBJECTS CREATED WITHIN FUNCTION:
    moms_data  = (200, 1) matrix, column vector of 200 data moments
    moms_model = (200, 1) matrix, column vector of 200 model moments
    err_vec    = (200, 1) matrix, column vector of 200 moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    moms_data = np.array([xvals['sick']])
    moms_model = b0 + b1*xvals['age'] + b2*xvals['children'] + b3*xvals['avgtemp_winter']
    moms_model = np.array([moms_model])
    
    err_vec = moms_model - moms_data
    
    return err_vec

def criterion_lin(params, *args): # compute the criterion function
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([b0, b1, b2, b3])
    args   = length 2 tuple, (xvals, W)
    xvals  = (N,) vector, values of the variable
    W  = (R, R) matrix, estimate of optimal weighting matrix
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (200, 1) matrix, column vector of 200 moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3 = params
    xvals, W = args
    err = err_vec_lin(xvals, b0, b1, b2, b3) # get error matrix
    crit_val = np.dot(np.dot(err, W), err.T)  
    
    return crit_val

# set parameters for the minimizer
b0_init = 1 # initial guess for four parameters
b1_init = 0
b2_init = 0
b3_init = 0
  
params_init_lin = np.array([b0_init, b1_init, b2_init, b3_init]) 
W_hat_lin = np.eye(200) # use simple identity matrix as weighting matrix
gmm_args_lin = (df, W_hat_lin) # plug in data and the weighting matrix)

results_lin = opt.minimize(criterion_lin, params_init_lin, args=(gmm_args_lin), method='L-BFGS-B') 
b0_gmm, b1_gmm, b2_gmm, b3_gmm = results_lin.x # unpack estimated parameters

# get criterion function values at the estimated GMM parameter values
params_GMM_lin = np.array([b0_gmm, b1_gmm, b2_gmm, b3_gmm])
GMM_Value_lin = criterion_lin(params_GMM_lin, *gmm_args_lin)[0][0]

if __name__ == '__main__':   
    print()
    print ('BEGINNING OF PROBLEM SET 3')
    print()
    print ('MACS 30100, Dr. Evans')
    print ('Yinxian Zhang')
    print()
    print ('Question 1')
    print('plot 1-5 are saved')
    print()
    print('Q1(b)')
    print('The value of GMM criterion function at the estimated parameter values is: ', GMM_Value )
    print('The GMM estimated mean and variance are:', mu_GMM1, sig_GMM1)
    print('Two data moments and two model moments at the estimated parameter values are: ', np.array([mean_data, std_data]), np.array([mean_model, std_model]))
    print()
    print('Q1(c)')
    print('The GMM estimated mean and variance are:', mu_GMM2, sig_GMM2)
    print('The value of GMM criterion function at the estimated parameter values is: ', GMM2_Value )
    print('Two data moments and two model moments at the estimated parameter values are: ', np.array([mean_data, std_data]), np.array([mean_model2, std_model2]))
    print()
    print('Q1(d)')
    print('The value of GMM criterion function at the estimated parameter values is: ', GMM3_Value )
    print('The GMM estimated mean and variance are:', mu_GMM1_3, sig_GMM1_3)
    print('Three data moments and Three model moments at the estimated parameter values are: ', np.array([bpct_1, bpct_2, bpct_3]), np.array([bpct_1_mod, bpct_2_mod, bpct_3_mod]))
    print()
    print('Q1(e)')
    print('The GMM estimated mean and variance are:', mu_GMM2_3, sig_GMM2_3)
    print('The value of GMM criterion function at the estimated parameter values is: ', GMM4_Value )
    print('Three data moments and Three model moments at the estimated parameter values are: ', np.array([bpct_1, bpct_2, bpct_3]), np.array([bpct_1_mod2, bpct_2_mod2, bpct_3_mod2]))
    print()
    print ('Question 2')
    print('The GMM estimated parameters are:', b0_gmm, b1_gmm, b2_gmm, b3_gmm)
    print('The value of GMM criterion function at the estimated parameter values is: ', GMM_Value_lin) 
    print ()
    print ('END OF PROBLEM SET 3')