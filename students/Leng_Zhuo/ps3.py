##Perspective

##Zhuo Leng

##PS3

# Import packages and load the data
import numpy as np
import pandas as pd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os


'''
--------------------------------------------------------------------
Excercise 1 function needed
--------------------------------------------------------------------
'''
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
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
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
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    
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


def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 220
    bpct_2_dat = scalar in [0, 1], percent of observations
                 220 <= x < 320
    bpct_3_dat = scalar in [0, 1], percent of observations
                 320 <= x < 430
    bpct_4_dat = scalar in [0, 1], percent of observations
                 430 <= x <= 450

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3, bpct_4
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = xvals[xvals >= 100000].shape[0] / xvals.shape[0]

    return bpct_1_dat, bpct_2_dat, bpct_3_dat

def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3, binpct_4).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
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
    bpct_4_mod = scalar in [0, 1], percent of model observations in
                 bin 4
    bp_4_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_4_mod
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod, bpct_4_mod
    --------------------------------------------------------------------
    '''
    xfx = lambda x: log_norm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 150000)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod

def err_vec3(xvals, mu, sigma, simple):
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
        data_moments4()
        model_moments4()
    
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
    bpct_1_dat, bpct_2_dat, bpct_3_dat =  data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
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
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (4, 1) matrix, column vector of four moment error
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



'''
--------------------------------------------------------------------
Excercise 1a histogram
--------------------------------------------------------------------
'''


incomes = np.loadtxt('incomes.txt')

plot_1a = True
if plot_1a:

    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath('ps3.py'))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed=True)
    plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
    plt.xlabel(r'annual incomes(\$s)')
    plt.ylabel(r'PDF of annual incomes of students')

    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    #plt.show()
    plt.close()



'''
--------------------------------------------------------------------
Excercise 1b GMM Estimation
--------------------------------------------------------------------
'''

mu_init1 = 10.0
sig_init1 = 0.2
params_init1 = np.array([mu_init1, sig_init1])
W_hat = np.eye(2)
gmm_args = (incomes, W_hat)
results1 = opt.minimize(criterion, params_init1, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results1.x
print()
print('1b')
print('mu_GMMb={:.3f},  sig_GMMb={:.3f}'.format(mu_GMM1, sig_GMM1))

mean_data, var_data = data_moments(incomes)
mean_model, var_model = model_moments(mu_GMM1, sig_GMM1)
params_GMM = np.array([mu_GMM1, sig_GMM1])
crt = criterion(params_GMM, *gmm_args)[0][0]
print('Mean of points = {:.3f}, Variance of points ={:.3f}'.format(mean_data, var_data))
print('Mean of model = {:.3f}, Variance of model = {:.3f}'.format(mean_model, var_model))
print('The value of GMM criterion function at the estimated parameter values is: {}'.format(crt))

fig = plt.figure()
# Plot the estimated GMM PDF
dist_pts = np.linspace(0, 150000, 1000000)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='$\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM1, sig_GMM1))
plt.legend(loc='upper left')

#plot part(a) histogram
count, bins, ignored = plt.hist(incomes, 30, normed=True)
#weights = (1 / len(incomes) * np.ones_like(incomes))
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
plt.xlabel(r'annual incomes(\$s)')
plt.ylabel(r'PDF of annual incomes of students')

output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
#plt.show()
plt.close()


'''
--------------------------------------------------------------------
Excercise 1c Two_step GMM
--------------------------------------------------------------------
'''

err1 = err_vec(incomes, mu_GMM1, sig_GMM1, False)
VCV2 = np.dot(err1, err1.T) / incomes.shape[0]
W_hat2 = lin.pinv(VCV2)

gmm_args = (incomes, W_hat2)
params_init_GMM1 = np.array([mu_GMM1, sig_GMM1])
results2 = opt.minimize(criterion, params_init_GMM1, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results2.x
print()
print('1c')
print('mu_GMMc={:.3f},  sig_GMMc={:.3f}'.format(mu_GMM2, sig_GMM2))
mean_model, var_model = model_moments(mu_GMM2, sig_GMM2)
params_GMM = np.array([mu_GMM2, sig_GMM2])
crt = criterion(params_GMM, *gmm_args)[0][0]
print('Mean of points = {:.3f}, Variance of points ={:.3f}'.format(mean_data, var_data))
print('Mean of model = {:.3f}, Variance of model = {:.3f}'.format(mean_model, var_model))
print('The value of GMM criterion function at the estimated parameter values is: {}'.format(crt))

fig = plt.figure()
# Plot the estimated GMM PDF in part1(b)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=1.5, color='g', label='Origin.$\mu$ = {:.5f}, $\sigma$ = {:.5f}'.format(mu_GMM1, sig_GMM1))
# Plot the estimated GMM PDF of two step
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM2, sig_GMM2),
         linewidth=1.5, color='r', linestyle = '-.', label='Two-step.$\mu$ = {:.5f}, $\sigma$ = {:.5f}'.format(mu_GMM2, sig_GMM2))
plt.legend(loc='upper left')

#plot part(a) histogram
count, bins, ignored = plt.hist(incomes, 30, normed=True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
plt.xlabel(r'annual incomes(\$s)')
plt.ylabel(r'PDF of annual incomes of students')

output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
#plt.show()
plt.close()

'''
--------------------------------------------------------------------
Excercise 1d different moment estimation 
--------------------------------------------------------------------
'''
#set new init value
mu_init2 = 11.0
sig_init2 = 0.3
params_init2 = np.array([mu_init2, sig_init2])
W_hat = np.eye(3)

gmm_args = (incomes, W_hat)
results_4 = opt.minimize(criterion3, params_init2, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None),(1e-10, None)))
mu_GMM1_4, sig_GMM1_4 = results_4.x
print()
print('1d')
print('mu_GMMd={:.3f},  sig_GMMd={:.3f}'.format(mu_GMM1_4, sig_GMM1_4))
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM1_4, sig_GMM1_4)
params_GMM = np.array([mu_GMM1_4, sig_GMM1_4])
crt = criterion3(params_GMM, *gmm_args)[0][0]
print('Moment of data: {:.3f}, {:.3f}, {:.3f}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('moment of model:{:.3f}, {:.3f}, {:.3f}'.format(bpct_1_mod, bpct_2_mod, bpct_3_mod))
print('The value of GMM criterion function at the estimated parameter values is: {}'.format(crt))



fig = plt.figure()
#plot part(a) histogram
count, bins, ignored = plt.hist(incomes, 30, normed=True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
plt.xlabel(r'annual incomes(\$s)')
plt.ylabel(r'PDF of annual incomes of students')

# Plot the estimated GMM PDF
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1_4, sig_GMM1_4),
         linewidth=2, color='k', label='$\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM1_4, sig_GMM1_4))
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
#plt.show()
plt.close()

'''
--------------------------------------------------------------------
Excercise 1e Two step GMM
--------------------------------------------------------------------
'''

err1e = err_vec3(incomes, mu_GMM1_4, sig_GMM1_4, False)
VCV2 = np.dot(err1e, err1e.T) / incomes.shape[0]
W_hat2 = lin.pinv(VCV2)

gmm_args = (incomes, W_hat2)
params_init_4e = np.array([mu_GMM1_4, sig_GMM1_4])
results_1e = opt.minimize(criterion3, params_init_4e, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM2_4, sig_GMM2_4 = results_1e.x
print()
print('1e')
print('mu_GMMe={:.3f},  sig_GMMe={:.3f}'.format(mu_GMM2_4, sig_GMM2_4))
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM2_4, sig_GMM2_4)
params_GMM2 = np.array([mu_GMM2_4, sig_GMM2_4])
crt = criterion3(params_GMM2, *gmm_args)[0][0]
print('Moment of data: {:.3f}, {:.3f}, {:.3f}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('moment of model:{:.3f}, {:.3f}, {:.3f}'.format(bpct_1_mod, bpct_2_mod, bpct_3_mod))
print('The value of GMM criterion function at the estimated parameter values is: {}'.format(crt))


fig = plt.figure()
# Plot the estimated GMM PDF in part1(d)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1_4, sig_GMM1_4),
         linewidth=2, color='k', label='origin.$\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM1_4, sig_GMM1_4))
##plot the estimated GMM PDF of two-step
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM2_4, sig_GMM2_4),
         linewidth=2, color='r', label='two-step.$\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_GMM2_4, sig_GMM2_4))
plt.legend(loc='upper left')

#plot part(a) histogram
count, bins, ignored = plt.hist(incomes, 30, normed=True)
plt.title('Annual incomes of MACSS graduates (2018-2020)', fontsize=17)
plt.xlabel(r'annual incomes(\$s)')
plt.ylabel(r'PDF of annual incomes of students')

output_path = os.path.join(output_dir, 'Fig_1e')
plt.savefig(output_path)
plt.show()
plt.close()

'''
--------------------------------------------------------------------
Excercise 2 Function needed
--------------------------------------------------------------------
'''


def criterion2a(params, *args):
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
    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3 = params
    df, W = args
    err = df['sick'] - b0 - b1*df['age'] - b2*df['children'] - b3*df['temp']
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


'''
--------------------------------------------------------------------
Excercise 2a GMM
--------------------------------------------------------------------
'''

sick= pd.read_csv('sick.txt', header = 0, names = ['sick','age','children',
	'temp'])

#set initial estimate value for parameters
b0_init = 0
b1_init = 0
b2_init = 0
b3_init = 0
params_init2a = np.array([b0_init, b1_init, b2_init, b3_init])
W_hat = np.eye(200)
gmm_args = (sick, W_hat)
results = opt.minimize(criterion2a, params_init2a, args=(gmm_args), method='L-BFGS-B')
b0_GMM1, b1_GMM1, b2_GMM1, b3_GMM1 = results.x
params_GMM = np.array([b0_GMM1,b1_GMM1,b2_GMM1,b3_GMM1])
crt = criterion2a(params_GMM, *gmm_args)
print()
print('2a')
print('b0_GMM= {:.4f}, b1_GMM= {:.4f}, b2_GMM= {:.4f}, b3_GMM= {:.4f}'.format(b0_GMM1,b1_GMM1,b2_GMM1,b3_GMM1)) 
print('Criterion function value with GMM2a estimates = ',crt)
 
