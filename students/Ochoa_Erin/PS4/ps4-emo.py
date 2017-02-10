# Erin M. Ochoa
# Perspectives II
# PS4
# 2017 February 6

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.integrate as intgr
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

SEED = 1234

warnings.filterwarnings('ignore')



def LN_pdf(xvals, mu, sigma):
    '''
    '''

    num_dimensions = len(xvals.shape)

    if num_dimensions == 2:
        rows, cols = xvals.shape

    if num_dimensions == 1 or cols == 1:
        pdf_vals = (1 / (xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - \
                   (np.log(xvals) - mu) ** 2 / (2 * sigma ** 2)))

        return pdf_vals

    else:
        pdf_vals = np.empty([rows,0])

        for col in range(cols):
            new_xvals = xvals[:,col:col + 1]
            col_pdf_vals = LN_pdf(new_xvals,mu,sigma)
            pdf_vals = np.concatenate((pdf_vals, col_pdf_vals), axis = 1)

        return pdf_vals



def LN_draws(mu, sigma, observations, rounds):
    '''
    '''
    distro = sts.norm(mu, sigma)
    draws = distro.rvs((observations, rounds))
    ln_draws = np.exp(draws)

    return ln_draws



def gen_moments(xvals):
    '''  Written by revans.  Edited by emo.
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), std(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mean_xvals = scalar or (S,) vector, mean value of data
    sd_xvals  = scalar > 0 or (S,) vector, standard deviation of data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_xvals = xvals.mean()
        sd_xvals = xvals.std()
    elif xvals.ndim == 2:
        mean_xvals = xvals.mean(axis=0)
        sd_xvals = xvals.std(axis=0)

    return mean_xvals, sd_xvals



def err_vec(data_vals, sim_vals, mu, sigma, simple):
    '''  Written by revans.  Edited by emo.
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
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, sd_data = gen_moments(data_vals)
    moms_data = np.array([[mean_data], [sd_data]])
    mean_sim, sd_sim = gen_moments(sim_vals)
    mean_model = mean_sim.mean()
    sd_model = sd_sim.mean()
    moms_model = np.array([[mean_model], [sd_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec



def criterion(params, *args):
    '''  Written by revans.  Edited by emo.
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

    xvals, w_hat, observations, rounds = args

    sim_vals = LN_draws(mu, sigma, observations, rounds)

    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)

    crit_val = np.dot(np.dot(err.T, w_hat), err)

    return crit_val



'''_________________________________________________________________________'''
'''_________________________________________________________________________'''
'''_________________________________________________________________________'''

# Create directory for images if it does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

print('-----PROBLEM 1-----')
'''1a
Plot a histogram of percentages of the income.txt data with 30 bins. Make sure
that the bins are weighted using the normed=True option. Make sure your plot
has correct x-axis and y-axis labels as well as a plot title.
'''

print('\nPart a')
print('------\n')

incomes = np.loadtxt('incomes.txt')
fig_name_1a = 'Fig_1a'

count, bins, ignored = plt.hist(incomes, 30, normed = True, color = 'cyan')
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1a)
plt.savefig(output_path)

print('Saved {}.\n'.format(fig_name_1a))





'''1b
Write your own function for the lognormal PDF above called LN_pdf(). Have this
function take as inputs xvals, mu, and sigma, where xvals is either a one-
dimensional (N, ) vector or an N × S matrix of simulated data where each column
is a simulation of the N data points on income, μ is the mean of the normal
distribution on which the lognormal is based, and σ is the standard deviation
of the normal distribution on which the lognormal is based. This function
should return an array pdf vals that is the same size as xvals and represents
the lognormal PDF values of each element of xvals given the parameters μ and σ.
Test your function by inputting the matrix xvals = np.array([[200.0, 270.0],
[180.0, 195.5]]) with parameter values μ = 5.0 and σ = 1.0.
'''


print('\nPart b')
print('------\n')

# Defined in the assignment
xvals_1b = np.array([[200.0, 270.0],[180.0, 195.5]])
mu_1b = 5.0
sig_1b = 1.0

pdf_vals = LN_pdf(xvals_1b, mu_1b, sig_1b)

print('Testing the LN_pdf() function with the given array:')
print(pdf_vals)







'''1c
Estimate the parameters of the log-normal distribution by simulated method of
moments (SMM). Create S = 300 simulations, each with N = 200 observations on
income from the lognormal distribution above. Use the average income and
standard deviation of income as your two moments. Use the identity matrix as
your weighting matrix Wˆ. Plot your estimated lognormal PDF against the
histogram from part (a). Report the value of your SMM criterion function at the
estimated parameter values. Report and compare your two data moments against
your two model moments at the estimated parameter values.

To draw your N × S
matrix of incomes from the log-normal distribution within your criterion
function, set the seed to 1234 in the random number generator by typing
np.random.seed(seed=1234). Then draw an N × S matrix of values from a normal
distribution with mean μ and standard deviation σ. Then transform those draws
into the lognormal distribution by exponentiating them numpy.exp().
'''


print('\nPart c')
print('------\n')

np.random.seed(SEED)

# Defined in the assignment
observations = len(incomes)
rounds = 300

# Guesstimating based on values from an earlier assignment
mu_1c = 11
sig_1c = .2

w_1c = np.eye(2)

params_1c = np.array([mu_1c, sig_1c])
args_1c = (incomes, w_1c, observations, rounds)
bds = ((1e-10, None), (1e-10, None))

results_1c = opt.minimize(criterion, params_1c, args=(args_1c),\
                       method='L-BFGS-B', bounds=bds, options={'eps': .007})

mu_smm_1c, sig_smm_1c = results_1c.x

crit_1c = results_1c.fun.sum()

mean_data, sd_data = gen_moments(incomes)

ln_draws = LN_draws(mu_smm_1c, sig_smm_1c, observations, rounds)
mean_sim, sd_sim = gen_moments(ln_draws)

mean_mdl = mean_sim.mean()
sd_mdl = sd_sim.mean()


print('mu_smm_1c  = {:.3f}'.format(mu_smm_1c))
print('sig_smm_1c =  {:.3f}'.format(sig_smm_1c))

print('Value of GMM criterion function with those values:', crit_1c)


print('\nMean of incomes = {:.3f}'.format(mean_data))
print('Mean of model  = {:.3f}'.format(mean_mdl))
print('SD of incomes = {:.3f}'.format(sd_data))
print('SD of model  = {:.3f}\n'.format(sd_mdl))

fig_name_1c = 'Fig_1c'

cutoff = 150000

dist_pts = np.linspace(0.1, cutoff, 500)
pdf_1c = LN_pdf(dist_pts, mu_smm_1c, sig_smm_1c)

plt.plot(dist_pts, pdf_1c, linewidth=1, color='magenta', label='$SMM_1c: ' + \
         '\mu = {:.3f}$, $\sigma ={:.3f}$'.format(mu_smm_1c,sig_smm_1c))
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, fig_name_1c)
plt.savefig(output_path)
#plt.close()

print('Saved {}.\n'.format(fig_name_1c))

print('\nPart d')
print('------\n')

'''1d
Perform the two-step SMM estimator by using your estimates from part (c) with
2 moments to generate an estimator for the variance-covariance matrix Ωˆ2step,
which you then use to get the two-step estimator for the optimal weighting
matrix Wˆ 2step. Report your estimates as well as the criterion function value
at these estimates. Plot your estimated lognormal PDF against the histogram
from part (a) and the estimated PDF from part (c). Report and compare your two
data moments against your two model moments at the estimated parameter values.
'''

error_vector = err_vec(incomes, ln_draws, mu_smm_1c, sig_smm_1c, False)
vcv_1d = np.dot(error_vector, error_vector.T) / len(incomes)
w_1d = np.linalg.pinv(vcv_1d)

print('Variance-covariance matrix:')
for line in vcv_1d:
    print(line)
#print(vcv_1d)

print('\nW-hat-2-step:')
for line in w_1d:
    print(line)


params_1d = np.array([mu_smm_1c, sig_smm_1c])
args_1d = (incomes, w_1d, observations, rounds)

results_1d = opt.minimize(criterion, params_1d, args=(args_1d),\
                       method='L-BFGS-B', bounds=bds, options={'eps': .007})


mu_smm_1d, sig_smm_1d = results_1d.x

crit_1d = results_1d.fun.sum()

ln_draws_1d = LN_draws(mu_smm_1d, sig_smm_1d, observations, rounds)
mean_sim_1d, sd_sim_1d = gen_moments(ln_draws_1d)

mean_mdl_1d = mean_sim_1d.mean()
sd_mdl_1d = sd_sim_1d.mean()

print('\nmu_smm_1d  = {:.3f}'.format(mu_smm_1d))
print('sig_smm_1d =  {:.3f}'.format(sig_smm_1d))

print('Value of GMM criterion function with those values:', crit_1d)


print('\nMean of incomes = {:.3f}'.format(mean_data))
print('Mean of model  = {:.3f}'.format(mean_mdl_1d))
print('SD of incomes = {:.3f}'.format(sd_data))
print('SD of model  = {:.3f}\n'.format(sd_mdl_1d))

fig_name_1d = 'Fig_1d'

pdf_1d = LN_pdf(dist_pts, mu_smm_1d, sig_smm_1d)

plt.plot(dist_pts, pdf_1d, linewidth=1, color='k', label='$SMM_1d: ' + \
         '\mu = {:.3f}$, $\sigma ={:.3f}$'.format(mu_smm_1d,sig_smm_1d))
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, fig_name_1d)
plt.savefig(output_path)
plt.close()

print('Saved {}.\n'.format(fig_name_1d))
