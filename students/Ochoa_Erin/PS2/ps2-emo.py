import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

warnings.filterwarnings('ignore')

def norm_pdf(xvals, mu, sigma, cutoff):
    '''  Written by revans.
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
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

    pdf_vals = ((1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / \
                (2 * sigma**2))) / prob_notcut)

    return pdf_vals



def log_lik_norm(xvals, mu, sigma, cutoff):
    '''  Written by revans.
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
    pdf_vals = norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val



def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''   Written by revans; modified by emo.
    --------------------------------------------------------------------
    Generate pdf values from the log-normal pdf with mean mu and standard
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar
    pdf_vals = (N,) vector, log-normal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)
        '''
        cutoff_log = np.log(cutoff)
        distribution = sts.lognorm([sigma], loc = mu)
        prob_notcut =  distribution.cdf(cutoff_log)
        '''

    pdf_vals = ((1 / (xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - \
                (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut)

    return pdf_vals


def log_lik_log_norm(xvals, mu, sigma, cutoff):
    '''   Written by revans; modified by emo.
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given log-normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the log-normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, log-normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of log-normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val



def crit(params, *args):
    '''   Written by revans.
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
    log_lik_val = log_lik_log_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val


def crit_sicko(params, *args):
    '''
    This function computes the negative of the log-likelihood function
    given parameters and data based on the sick_weeks model as defined in the
    assignment. This is the minimization problem version of the maximum-
    likelihood optimization problem.

    Inputs:
        params:  a parameter list of guesses of coefficients
            beta0_intercept
            beta1_age
            beta2_kids
            beta3_temp

        args:  a list of arguments
            sick_weeks
            age
            kids
            temp
            cutoff
            mu_2a

    Outputs:
        - log_likelihood (float): the negative of the log-likelihood value
                                  computed by the log_likelihood_sicko()
                                  function.
    '''

    beta0_intercept, beta1_age, beta2_kids, beta3_temp, sigma = params

    sick_weeks, age, kids, temp, cutoff, mu_2a = args

    log_likelihood = log_likelihood_sicko(sick_weeks, age, kids, temp, \
                     beta0_intercept, beta1_age, beta2_kids, beta3_temp, \
                     mu_2a, sigma ** 2, cutoff)

    return - log_likelihood

def log_likelihood_sicko(sick_wks,age,kids,temp,b0,b1,b2,b3,mu,sigma,cutoff):
    '''
    This function computes the log-likelihood for the sick_weeks model as
    described in the assignment.

    Inputs:
        sick_wks
        age
        kids
        temp
        b0
        b1
        b2
        b3
        mu
        sigma
        cutoff

    Outputs:
        log_likelihood_val (float):  the log-likelihood value
    '''
    errors = sick_wks - b0 - (b1 * age) - (b2 * kids) - (b3 * temp)

    pdf_vals = norm_pdf(errors, mu, sigma, cutoff)

    log_likelihood_val = np.log(pdf_vals).sum()

    return log_likelihood_val


'''_________________________________________________________________________'''
'''_________________________________________________________________________'''
'''_________________________________________________________________________'''


# Create directory for images if it does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


'''1a'''
print('-----PROBLEM 1-----')
print('\nPart a')
print('------\n')

incomes = np.loadtxt('incomes.txt')
fig_name_1a = 'Fig_1a'

count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')
plt.xlim([40000, 140000])
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1a)
plt.savefig(output_path)

print('Saved {}.\n'.format(fig_name_1a))

'''1b'''
print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nPart b')
print('------\n')

mu_1b = 9
sig_1b = .3
cutoff = 150000

dist_pts = np.linspace(0.1, cutoff, 500)



log_lklhd_1b = log_lik_log_norm(incomes, mu_1b, sig_1b, cutoff)

print('Log-likelihood: {:.3f}'.format(log_lklhd_1b))


fig_name_1b = 'Fig_1b'

log_norm_pdf_1b = log_norm_pdf(dist_pts, mu_1b, sig_1b, cutoff)

plt.plot(dist_pts, log_norm_pdf_1b, linewidth=2, color='c', \
         label='1: $\mu$={},$\sigma$={}'.format(mu_1b,sig_1b))
plt.xlim([0, cutoff])
plt.legend(loc='upper right')
#plt.show()
output_path = os.path.join(output_dir, fig_name_1b)
plt.savefig(output_path)
plt.close()

print('\nSaved {}.\n'.format(fig_name_1b))


'''
1c)
Estimate the parameters of the lognormal distribution by maximum likelihood and
plot its PDF against the PDF from part (b) and the histogram from part (a).
Plot the estimated PDF for 0 ≤ x ≤ 150, 000. Report the ML estimates for μ and
σ, the value of the likelihood function, and the variance-covariance matrix.
'''
print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nPart c')
print('------\n')


mu_init = mu_1b
sig_init = sig_1b

params_init = np.array([mu_init, sig_init])
mle_args = (incomes, cutoff)
bds_1c = ((None, None), (0.00001, None))

results = opt.minimize(crit, params_init, method='SLSQP', args = (mle_args),\
                       bounds = bds_1c)

mu_MLE, sig_MLE = results.x
print('mu_MLE = {:.3f}'.format(mu_MLE))
print('sig_MLE = {:.3f}'.format(sig_MLE))

mu_vals = np.linspace(mu_MLE - .5, mu_MLE + .5, 50)
sig_vals = np.linspace(sig_MLE - .05, sig_MLE + .05, 50)
lnlik_vals = np.zeros((50, 50))
for mu_ind in range(50):
    for sig_ind in range(50):
        lnlik_vals[mu_ind, sig_ind] = log_lik_log_norm(incomes, \
                                    mu_vals[mu_ind], sig_vals[sig_ind], cutoff)

#print('MLE log-likelihood 3: ', log_lik_norm(incomes, mu_MLE, sig_MLE, cutoff))

log_likelihood_norm = log_lik_log_norm(incomes, mu_MLE, sig_MLE, cutoff)

print('MLE log-likelihood: {:.3f}'.format(log_likelihood_norm))

params_init = np.array([1,0,0,0,0])

new_results = opt.minimize(crit, results.x, method='BFGS', args = (mle_args),\
                           bounds = bds_1c)



vcv_mle = new_results.hess_inv
stderr_mu_mle = np.sqrt(vcv_mle[0,0])
stderr_sig_mle = np.sqrt(vcv_mle[1,1])
print('\nVariance-covariance matrix(MLE):')

for row in vcv_mle:
    print('\t', row)

#print('\nStandard error for mu estimate = {:.4f}'.format(stderr_mu_mle))
#print('Standard error for sigma estimate = {:.4f}'.format(stderr_sig_mle))

fig_name_1c = 'Fig_1c'

# Plot the histogram of the data
count, bins, ignored = plt.hist(incomes, 30, normed=True)
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')
plt.tight_layout()
plt.xlim([0, cutoff])
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1b, sig_1b, cutoff), linewidth=2,\
         color='c', label='1: $\mu$={},$\sigma$={}'.format(mu_1b,sig_1b))
plt.legend(loc='upper right')

# Plot the MLE estimated distribution
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_MLE, sig_MLE, cutoff),\
         linewidth=2,color='magenta',label='MLE: $\mu$={:.3f},$\sigma$={:.3f}'\
         .format(mu_MLE,sig_MLE))
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, fig_name_1c)
plt.savefig(output_path)
plt.close()


print('\nSaved {}.\n'.format(fig_name_1c))


'''1d'''

print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nPart d')
print('------\n')

log_lik_h0 = log_lik_log_norm(incomes, mu_1b, sig_1b, cutoff)
log_lik_mle = log_lik_log_norm(incomes, mu_MLE, sig_MLE, cutoff)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('Chi-squared of H0 with two degrees of freedom p-value ' + \
      '= {:.3f}\n'.format(pval_h0))





'''1e'''
'''
What is the probability that you will earn more than $100,000?
What is the probability that you will earn less than $75,000?
'''
print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nPart e')
print('------\n')


exp_mu_MLE = np.exp(mu_MLE)

bank = 100000
broke = 75000

bank_prob = 1 - sts.lognorm(s = sig_MLE, scale = exp_mu_MLE).cdf(bank)
broke_prob = sts.lognorm(s = sig_MLE, scale = exp_mu_MLE).cdf(broke)

print('The probability of a given MACSS graduate earning more than $100,000'\
      + ' is: {:.3f}.\n'.format(bank_prob))
print('The probability of a given MACSS graduate earning less than $75,000'\
      + ' is: {:.3f}.\n'.format(broke_prob))




#----3d plot--------------------------------------------------------------------
print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nBonus plot')
print('----------\n')
cmap1 = cm.get_cmap('winter')

mu_mesh, sig_mesh = np.meshgrid(mu_vals, sig_vals)

fig_name_3d = '3d_plot'

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(sig_mesh, mu_mesh, lnlik_vals, rstride=8,
                cstride=1, cmap=cmap1)
ax.set_title(r'Log likelihood for values of $\mu$ and $\sigma$')
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\mu$')
ax.set_zlabel(r'Log-likelihood')
#plt.show()
output_path = os.path.join(output_dir, fig_name_3d)
plt.savefig(output_path)
plt.close()
print('Saved {}.\n'.format(fig_name_3d))
#------------------------------------------------------------------------------


print('______________________________________________________________________')
print('______________________________________________________________________')
print('\n\n-----PROBLEM 2-----')




'''2a'''
'''
Estimate the parameters of the model (β0,β1,β2,σ2) by maximum likelihood using
the fact that each error term εi is distributed normally N(0, σ2). We can solve
the regression equation for εi which tells us that the following equation is
distributed normally N(0, σ2).

sicki − β0 - β1agei - β2childreni - β3tempwinteri ∼N(0,σ^2)

Estimate (β0,β1,β2,β3,σ^2) to maximize the likelihood of seeing the data in
sick.txt. Report your estimates, the value of the log likelihood function,
and the estimated variance covariance matrix of the estimates.
'''

print('\nPart a')
print('------\n')

sicko = 'sick.txt'
cols = ['Sick_Weeks_Taken', 'Age', 'Children', 'Mean_Winter_Temp_F']

sick_2016 = pd.read_csv(sicko, names = cols, skiprows = 1)

'''Pull out the variables to make things easier.'''
sick_weeks = sick_2016['Sick_Weeks_Taken']
age = sick_2016['Age']
kids = sick_2016['Children']
mean_temp = sick_2016['Mean_Winter_Temp_F']




'''Make some guesses about coefficients.'''
b0_intercept_guess = 0.2
b1_age_guess = .05
b2_kids_guess = 0.9
b3_temp_guess = -0.01
sig_2a_guess = 1.0


'''max(sick_weeks) = 2.79999999.'''
cutoff_2a = 2.8

'''This is defined in the assignment.'''
mu_2a = 0.0

'''Putting it all together to call the crit_sicko function.'''
param_guesses = [b0_intercept_guess, b1_age_guess, b2_kids_guess,\
                 b3_temp_guess, sig_2a_guess]

args_2a = (sick_weeks, age, kids, mean_temp, cutoff_2a, mu_2a)

bds_2a = ((None, None),(None, None),(None, None),(None, None),(0.01, None))

results_2a = opt.minimize(crit_sicko, param_guesses, args = (args_2a),\
                          bounds = bds_2a, method = 'SLSQP')

#print('\n\nresults_2a =\n', results_2a)


new_results_2a = opt.minimize(crit_sicko, results_2a.x, args = (args_2a), \
                              bounds = bds_2a, method = 'BFGS')


#print('\n\nnew_results_2a =\n', new_results_2a)


b0, b1, b2, b3, sig_2a = results_2a.x

log_lklhd_2a = results_2a.fun

vcv_mle_2a = new_results_2a.hess_inv

print('β_0 =  {:.3f}'.format(b0))
print('β_1 =  {:.3f}'.format(b1))
print('β_2 =  {:.3f}'.format(b2))
print('β_3 = {:.3f}'.format(b3))
print('σ^2 =  {:.3f}'.format(sig_2a ** 2))

print('\nLog-likelihood: {:.3f}'.format(log_lklhd_2a))

print('\nVariance-covariance matrix:')

for row in vcv_mle_2a:
    print('\t', row)

print('')

'''2b'''

print('______________________________________________________________________')
print('______________________________________________________________________')
print('\nPart b')
print('------\n')

'''The values given in the question:'''
b0_2b = 1.0
b1_2b, b2_2b, b3_2b = 0, 0, 0
sig_2b = .01

log_lik_h0_2b = log_likelihood_sicko(sick_weeks, age, kids, mean_temp, b0_2b, \
                b1_2b, b2_2b, b3_2b, mu_2a, sig_2b, cutoff)

LR_val_2b = 2 * (log_lklhd_2a - log_lik_h0_2b)
pval_h0_2b = 1.0 - sts.chi2.cdf(LR_val_2b, 5)


print('Chi-squared of H0 with five degrees of freedom p-value ' + \
      '= {:.3f}'.format(pval_h0_2b))
