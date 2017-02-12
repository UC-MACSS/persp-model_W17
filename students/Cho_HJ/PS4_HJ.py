'''
---------- ---------- ---------- ---------- ----------
MACS 30100 - Perspectives on Computational Modeling
HyungJin Cho
---------- ---------- ---------- ---------- ----------
'''

# Importing Pacakges
import os
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# < Data >
incomes = np.loadtxt('incomes.txt')

# < Directory >
def directory():
    DIRECTORY = os.path.split(os.path.abspath('__file__'))[0]
    FOLDER = 'images'
    DIRECTORY_FOLDER = os.path.join(DIRECTORY, FOLDER)
    if not os.access(DIRECTORY_FOLDER, os.F_OK):
        os.makedirs(DIRECTORY_FOLDER)
    return(DIRECTORY_FOLDER)

# < Function >
def F_LOGNORMAL(UNIFORM_VALUE, MU, SIGMA):
    '''
    --------------------------------------------------------------------
    This function computes a draw (N x S) matrix of random draws from a lognormal distribution based on a normal distribution with mean mu and standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws correspond to an (N x S) matrix of randomly generated draws from a uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    UNIFORM_VALUE = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    MU        = scalar, mean of the lognormal distribution
    SIGMA     = scalar > 0, standard deviation of the lognormal distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.lognorm.ppf()

    OBJECTS CREATED WITHIN FUNCTION:
    LOGNORMAL = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from lognormal PDF

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: LOGNORMAL_VALUE
    --------------------------------------------------------------------
    '''
    NORMAL = sts.norm.ppf(UNIFORM_VALUE, loc=MU, scale=SIGMA)
    LOGNORMAL = np.exp(NORMAL)
    return LOGNORMAL

def F_DATA_MOMENT(X_VALUE):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    X_VALUE = (N, S) matrix, (N,) vector, or scalar,
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    DATA_MEAN = scalar or (S,) vector, mean value of test scores data
    DATA_VAR  = scalar > 0 or (S,) vector, variance of test scores data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: DATA_MEAN, DATA_VAR
    --------------------------------------------------------------------
    '''
    if X_VALUE.ndim == 1:
        DATA_MEAN = X_VALUE.mean()
        DATA_VAR = X_VALUE.var()
    elif X_VALUE.ndim == 2:
        DATA_MEAN = X_VALUE.mean(axis=0)
        DATA_VAR = X_VALUE.var(axis=0)
    return DATA_MEAN, DATA_VAR

def F_ERROR_VECTOR(DATA_VALUE, SIMULATED_VALUE, SIMPLE):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    DATA_VALUE = (N,) vector, test scores data
    SIMULATED_VALUE  = (N, S) matrix, S simulations of test scores data
    SIMPLE    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        F_DATA_MOMENT()

    OBJECTS CREATED WITHIN FUNCTION:
    DATA_MEAN  = scalar, mean value of data
    DATA_VAR   = scalar > 0, variance of data
    DATA_MOMENT  = (2, 1) matrix, column vector of two data moments
    MODEL_MEAN = scalar, estimated mean value from model
    MODEL_VAR  = scalar > 0, estimated variance from model
    MODEL_MOMENT = (2, 1) matrix, column vector of two model moments
    ERROR_VECTOR    = (2, 1) matrix, column vector of two moment error functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: ERROR_VECTOR
    --------------------------------------------------------------------
    '''
    DATA_MEAN, DATA_VAR = F_DATA_MOMENT(DATA_VALUE)
    DATA_MOMENT = np.array([[DATA_MEAN], [DATA_VAR]])
    SIMULATED_MEAN, SIMULATED_VAR = F_DATA_MOMENT(SIMULATED_VALUE)
    MODEL_MEAN = SIMULATED_MEAN.mean()
    MODEL_VAR = SIMULATED_VAR.mean()
    MODEL_MOMENT = np.array([[MODEL_MEAN], [MODEL_VAR]])
    if SIMPLE:
        ERROR_VECTOR = MODEL_MOMENT - DATA_MOMENT
    else:
        ERROR_VECTOR = (MODEL_MOMENT - DATA_MOMENT) / DATA_MOMENT
    return ERROR_VECTOR

def F_CRITERION(PARAMETER, *ARGS):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    PARAMETER    = (2,) vector, ([mu, sigma])
    MU        = scalar, mean of the normally distributed random variable
    SIGMA     = scalar > 0, standard deviation of the normally
                distributed random variable
    ARGS      = length 5 tuple,
                (xvals, UNIFORM_VALUE, cut_lb, cut_ub, W_hat)
    DATA_VALUE     = (N,) vector, values of the truncated normally
                distributed random variable
    UNIFORM_VALUE = (N, S) matrix, matrix of draws from U(0,1) distribution.
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
    CRIT_VALUE   = scalar > 0, GMM criterion function value

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: CRIT_VALUE
    --------------------------------------------------------------------
    '''
    MU, SIGMA = PARAMETER
    DATA_VALUE, UNIFORM_VALUE, W_hat = ARGS
    SIMULATED_VALUE = F_LOGNORMAL(UNIFORM_VALUE, MU, SIGMA)
    ERROR = F_ERROR_VECTOR(DATA_VALUE, SIMULATED_VALUE, SIMPLE=False)
    CRIT_VALUE = np.dot(np.dot(ERROR.T, W_hat), ERROR)
    return CRIT_VALUE

'''
1.(a)
'''
# < Graph >
n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
plt.title('Histogram of annual incomes of MACSS graudates in 2018-2020', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Percentage of Students')
plt.xlim(0, 150000)
plt.ylim(0, 1.2*n.max())

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(a)'), bbox_inches = 'tight')
plt.close()

'''
1.(b)
'''
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the lognormal pdf values by taking lognormally distributed data, mu, sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals     = (N, S) matrix, (N,) vector, or scalar,
                test scores data, either real world or simulated. Real world
                data will come in the form (N,). Simulated data comes in the
                form (N,) or (N, S).
    mu        = scalar, mean of the lognormal distribution
    sigma     = scalar > 0, standard deviation of the lognormal distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        numpy.log()
        numpy.exp()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  lognormal pdf values

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = (np.exp(-1*(np.log(xvals)-mu)**2/(2*sigma**2)) / (xvals*sigma*np.sqrt(2*np.pi)))
    return pdf_vals

xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
pdf_vals = LN_pdf(xvals, 5.0, 1.0)
print('1.(b): ', pdf_vals)

'''
1.(c)
'''
S = 300
N = 200
W_hat0 = np.eye(2)
np.random.seed(seed=1234)
UNIFORM_VALUE = sts.uniform.rvs(0, 1, size=(N, S))
DATA_MEAN, DATA_VAR = incomes.mean(), incomes.var()
BOUNDS = ((0, None), (1e-10, None))

SIMULATED_VALUE1 = F_LOGNORMAL(UNIFORM_VALUE, 11., .2)
SIMULATED_MEAN1, SIMULATED_VAR1 = F_DATA_MOMENT(SIMULATED_VALUE1)
MODEL_MEAN, MODEL_VAR = SIMULATED_MEAN1.mean(), SIMULATED_VAR1.mean()

PARAMETER = np.array([11., .2])
ARGS = (incomes, UNIFORM_VALUE, W_hat0)

OUTCOME1 = opt.minimize(F_CRITERION, PARAMETER, args=(ARGS),
                       method='L-BFGS-B', bounds=BOUNDS)

# < Results >
SSM_MU, SSM_SIGMA = OUTCOME1.x
SUMULATED_VALUE1 = F_LOGNORMAL(UNIFORM_VALUE, SSM_MU, SSM_SIGMA)
SIMULATED_MEAN1, SIMULATED_VAR1 = F_DATA_MOMENT(SUMULATED_VALUE1)
MODEL_MEAN, MODEL_VAR = SIMULATED_MEAN1.mean(), SIMULATED_VAR1.mean()
DIFFERENCE_MEAN, DIFFERENCE_VAR = (MODEL_MEAN - DATA_MEAN)/DATA_MEAN, (MODEL_VAR - DATA_VAR)/DATA_VAR

print('SSM_MU=', SSM_MU, ' SSM_SIGMA=', SSM_SIGMA)
print('Criterion Function Value: ', OUTCOME1.fun)
print('Data mean:', DATA_MEAN, 'Data variance: ', DATA_VAR)
print('Sim mean:', MODEL_MEAN, 'Sim variance', MODEL_VAR)
print('Mean difference(%): ', DIFFERENCE_MEAN*100, 'Variance difference(%): ', DIFFERENCE_VAR*100)

# < Graph >
n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
dist_pts = np.linspace(1e-10, 150000, 150000)
plt.plot(dist_pts, LN_pdf(dist_pts, SSM_MU, SSM_SIGMA),
         label='$\mu$={},$\sigma$={}'.format(SSM_MU, SSM_SIGMA))
plt.title('Histogram of annual incomes of MACSS graudates in 2018-2020', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Percentage of Students')
plt.xlim(0, 150000)
plt.ylim(0, 1.2*n.max())
plt.legend(loc='upper right')

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(c)'), bbox_inches = 'tight')
plt.close()

'''
1.(d)
'''
ERROR = F_ERROR_VECTOR(incomes, SUMULATED_VALUE1, False)
VCV2 = np.dot(ERROR, ERROR.T) / incomes.shape[0]
W_hat2 = lin.pinv(VCV2)

PARAMETER2 = np.array([SSM_MU, SSM_SIGMA])
ARGS2 = (incomes, UNIFORM_VALUE, W_hat2)

OUTCOME2 = opt.minimize(F_CRITERION, PARAMETER2, args=(ARGS2),
                        method='TNC', bounds=BOUNDS)

# < Results >
SSM_MU2, SSM_SIGMA2 = OUTCOME2.x
SIMULATED_VALUE2 = F_LOGNORMAL(UNIFORM_VALUE, SSM_MU2, SSM_SIGMA2)
SIMULATED_MEAN2, SIMULATED_VAR2 = F_DATA_MOMENT(SIMULATED_VALUE2)
MODEL_MEAN2, MODEL_VAR2 = SIMULATED_MEAN2.mean(), SIMULATED_VAR2.mean()
DIFFERENCE_MEAN2, DIFFERENCE_VAR2 = (MODEL_MEAN2 - DATA_MEAN)/DATA_MEAN, (MODEL_VAR2 - DATA_VAR)/DATA_VAR

print('SMM_MU=', SSM_MU2, ' SSM_SIGMA=', SSM_SIGMA2)
print('Criterion Function Value: ', OUTCOME2.fun)
print('Data mean:', DATA_MEAN, 'Data variance: ', DATA_VAR)
print('Model mean:', MODEL_MEAN2, 'Model variance', MODEL_VAR2)
print('Mean difference(%): ', DIFFERENCE_MEAN2*100, 'Variance difference(%): ', DIFFERENCE_VAR2*100)

# < Graph >
n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
dist_pts = np.linspace(1e-10, 150000, 150000)
plt.plot(dist_pts, LN_pdf(dist_pts, SSM_MU2, SSM_SIGMA2),
         label='$\mu$={},$\sigma$={}'.format(SSM_MU, SSM_SIGMA))
plt.title('Histogram of annual incomes of MACSS graudates in 2018-2020', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Percentage of Students')
plt.xlim(0, 150000)
plt.ylim(0, 1.2*n.max())
plt.legend(loc='upper right')

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(d)'), bbox_inches = 'tight')
plt.close()
