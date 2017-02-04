# Perspectives on Computational Modeling
# Problem Set 3
# Name: Xinzhu Sun

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import scipy.stats as sts
import scipy.integrate as intgr
import pandas as pd
import os


# Problem 1
# (a) Plot the Histogram
income = np.loadtxt("incomes.txt", comments="#", delimiter=",", unpack=False)
count, bins, patches = plt.hist(income, 30, normed = True)
plt.xlabel("Income")
plt.ylabel("Density")
plt.title("Income for MACSS Students Histogram", fontsize = 20)
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "hist_income")
plt.savefig(output_path, bbox_inches='tight')
plt.show()


def log_norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and
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
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
            
    pdf_vals    = 1/(xvals * sigma * np.sqrt(2 * np.pi)) *\
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)) 
    return pdf_vals


def data_moments(xvals):
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
    var_data  = scalar > 0, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    var_data = xvals.var()
    
    return mean_data, var_data


def model_moments(mu, sigma):
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
    var_model  = scalar > 0, variance of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, var_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 1e-50, 15000000, limit = 1000)
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 1e-50, 15000000, limit = 1000)
    return mean_model, var_model

def err_vec(xvals, mu, sigma, simple):
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
    mean_data, var_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_model, var_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion(params, *args):
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
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)    
    return crit_val

mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
gmm_args = (income, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='Nelder-Mead')
mu_GMM1, sig_GMM1 = results.x
error1 = err_vec(income, mu_GMM1, sig_GMM1, False)
crit_val = criterion(results.x, income, W_hat)
print(r"1b): The GMM estimator is: $\mu$ = {}, $\sigma$ = {}"\
     .format(mu_GMM1, sig_GMM1))
print("    The value of the criterion function is {}".format(crit_val))
print("    For two data moment: {}".format(data_moments(income)))
print("    For two model moment: {}".format(model_moments(mu_GMM1, sig_GMM1)))
print()

dist_pers = np.linspace(0, 140000, 1000000)
income = np.loadtxt("incomes.txt", comments="#", delimiter=",", unpack=False)
count, bins, patches = plt.hist(income, 30, normed = True)
plt.plot(dist_pers, log_norm_pdf(dist_pers, mu_GMM1, sig_GMM1), \
         label = 'Orginal, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM1, sig_GMM1))
plt.xlabel("Income")
plt.ylabel("Density")
plt.legend(loc = "upper left", prop={'size':8})
plt.title('Income PDF GMM', fontsize = 15)
output_path = os.path.join(output_dir, "1b")
plt.savefig(output_path, bbox_inches='tight')


W_two = np.linalg.pinv(1 / len(error1) * np.dot(error1, error1.T))
gmm_args = (income, W_two)
params_init = np.array([mu_GMM1, sig_GMM1])
results_two = opt.minimize(criterion, params_init,  args = (gmm_args), tol = 1e-2,\
                       method = 'Nelder-Mead', options = {'maxiter': 1000})
mu_GMM2, sig_GMM2 = results_two.x
crit_val2 = criterion(results_two.x, income, W_two)
print(r"1c): The GMM estimator is: $\mu$ = {}, $\sigma$ = {}"\
     .format(mu_GMM2, sig_GMM2))
print("    The value of the criterion function is {}".format(crit_val2))
print("    For two data moment: {}".format(data_moments(income)))
print("    For two model moment: {}".format(model_moments(mu_GMM2, sig_GMM2)))
print()
plt.title("Income for MACSS Students Histogram", fontsize = 20)
plt.plot(dist_pers, log_norm_pdf(dist_pers, mu_GMM2, sig_GMM2), \
         label = 'TWO-STEP, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM2, sig_GMM2))
plt.xlim((0, None))
plt.legend(loc = "upper left", prop={'size':8})
output_path = os.path.join(output_dir, "1c")
plt.title('Income PDF GMM (TWO-STEP)', fontsize = 15)
plt.savefig(output_path, bbox_inches='tight', linestyle = '--')
plt.show()

def data_moments3(xvals):
    '''
    Given data, copute the data moment using three moment conditions
    
    Input:
        xvals: the dataset
        
    Output:
        np.array([per1_model, per2_model, per3_model]):
            A numpy array that contains the three data moment value
    '''
    per1 = len(xvals[xvals <= 75000]) / len(xvals)
    per2 = len(xvals[(xvals <= 100000) & (xvals > 75000)]) / len(xvals)
    per3 = len(xvals[xvals > 100000]) / len(xvals)
    return np.array([per1, per2, per3])

def model_moments3(mu, sigma):
    '''
    Given the expectation and variance, compute the model moment.
    
    Input:
        mu: the expectation of the model
        sigma: the variance of the model
        
    Output:
        np.array([per1_model, per2_model, per3_model]): a numpy array that
        contains the three model moment value
    '''
    xfx = lambda x: log_norm_pdf(x, mu, sigma)
    (per1_model, m_m_err1) = intgr.quad(xfx, 1e-50, 75000, limit = 1000)
    (per2_model, m_m_err2) = intgr.quad(xfx, 75000, 100000, limit = 1000)
    (per3_model, m_m_err3) = intgr.quad(xfx, 100000, 15000000, limit = 1000)
    return np.array([per1_model, per2_model, per3_model])

def err_vec3(xvals, mu, sigma, simple):
    '''
    Given the dataset and hypothesized expectation and variance
    compute the error vector of the model and the data
    
    Input:
        xvals: the dataset
        mu: the expectation
        sigma: the standard deviation of the model
        simple: a boolean. True, then use the simple error
        False, then use the percent deviatoin
    '''
    moms_data = data_moments3(xvals).reshape(3, 1)
    moms_model = model_moments3(mu, sigma).reshape(3, 1)
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
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
    xvals, W = args
    err = err_vec3(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    return crit_val

mu_init = 10.5
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(3)
gmm_args = (income, W_hat)
results3 = opt.minimize(criterion3, params_init,  args = (gmm_args), \
                       method = 'L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu3_GMM, sig3_GMM = results3.x
error1 = err_vec3(income, mu3_GMM, sig3_GMM, False)
crit3_val = criterion3(results3.x, income, W_hat)
print(r"1d): The GMM estimator using three moments condition is: $\mu$ = {}, $\sigma$ = {}"\
     .format(mu3_GMM, sig3_GMM))
print("    The value of the criterion function is {}".format(crit3_val))
print("    For three data moment: {}".format(data_moments3(income)))
print("    For three model moment: {}".format(model_moments3(mu3_GMM, sig3_GMM)))
print()

##Plot the PDF
dist_pers = np.linspace(0, 140000, 1000000)
income = np.loadtxt("incomes.txt", comments="#", delimiter=",", unpack=False)
count, bins, patches = plt.hist(income, 30, normed = True)
plt.plot(dist_pers, log_norm_pdf(dist_pers, mu3_GMM, sig3_GMM),\
         label = 'Orginal, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu3_GMM, sig3_GMM))
plt.xlabel("Income")
plt.ylabel("PDF")
plt.title('Income PDF Using Three Momemt Conditions', fontsize = 15)
plt.legend(loc = "upper left", prop={'size':8})
output_path = os.path.join(output_dir, "1d")
plt.savefig(output_path, bbox_inches='tight')

cov = 1 / len(income) * np.dot(error1, error1.T)
W3_two = np.linalg.pinv(cov)
gmm_args = (income, W3_two)
params_init = np.array([mu3_GMM, sig3_GMM])
result3_two = opt.minimize(criterion3, params_init,  args = (gmm_args), \
                       method = 'Nelder-Mead')
mu3_GMM2, sig3_GMM2= result3_two.x
error2 = err_vec3(income, mu3_GMM2, sig3_GMM2, False)
crit3_val2 = criterion3(result3_two.x, income, W_hat)
print(r"1e): The TWO-STEP GMM estimator using three moments condition is: $\mu$ = {}, $\sigma$ = {}"\
     .format(mu3_GMM2, sig3_GMM2))
print("    The value of the criterion function is {}".format(crit3_val2))
print("    For three model moment: {}".format(model_moments3(mu3_GMM2, sig3_GMM2)))
print()

plt.plot(dist_pers, log_norm_pdf(dist_pers, mu3_GMM2, sig3_GMM2), linestyle = '--',\
         label = 'TWO-STEP, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu3_GMM2, sig3_GMM2))
plt.title('Income PDF Using Three Momemt Conditions(TWO-STEP)', fontsize = 15)
output_path = os.path.join(output_dir, "1e")
plt.legend(loc = "upper left", prop={'size':8})
plt.savefig(output_path, bbox_inches='tight')
plt.show()

# Problem 2
df_sick = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter'])
if True:
    def data_moments_sick(xdf):
        moments_data = xdf['sick']
        return moments_data

    def model_moments_sick(xdf, b0, b1, b2, b3):
        moments_model = b0 + b1 * xdf['age'] + b2 * xdf['children'] + b3 * xdf['avgtemp_winter']
        return moments_model

    def err_vec_sick(xdf, b0, b1, b2, b3, simple):
        moms_data = data_moments_sick(xdf)
        moms_model = model_moments_sick(xdf, b0, b1, b2, b3)
        if simple:
            err_vec = moms_model - moms_data
        else:
            err_vec = (moms_model - moms_data) / moms_data
        return err_vec

    def criterion_sick(params, *args):
        b0, b1, b2, b3 = params
        xdf, W = args
        err = err_vec_sick(xdf, b0, b1, b2, b3, simple=True)
        crit_val = np.dot(np.dot(err.T, W), err)
        return crit_val

    b0_init, b1_init, b2_init, b3_init = (0, 0, 0, 1)
    params_init = np.array([b0_init, b1_init, b2_init, b3_init])
    W_hat = np.eye(200)
    gmm_args = (df_sick, W_hat)
    results_sick = opt.minimize(criterion_sick, params_init, args=(gmm_args),
                                method='L-BFGS-B')
    b0_GMM1, b1_GMM1, b2_GMM1, b3_GMM1 = results_sick.x
    print('2.(a)')
    print('b0_GMM1=', b0_GMM1, 'b1_GMM1=', b1_GMM1,
          'b2_GMM1=', b2_GMM1, 'b3_GMM1=', b3_GMM1)
    print('criterion_GMM1=',
          criterion_sick((b0_GMM1, b1_GMM1, b2_GMM1, b3_GMM1), df_sick, W_hat))