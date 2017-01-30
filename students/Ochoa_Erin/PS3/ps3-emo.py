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

warnings.filterwarnings('ignore')

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



def data_moments(xvals):
    '''  Written by revans; edited by emo.
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), sd(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value  data
    sd_data  = scalar > 0, standard deviation of data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, sd_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    sd_data = xvals.std()

    return mean_data, sd_data


def model_moments(mu, sigma, cutoff):
    '''Written by revans; edited by emo.
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
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma, cutoff)
    #(mean_model, m_m_err) = intgr.quad(xfx, -np.inf, cutoff)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, cutoff)

    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma, cutoff)
    #(var_model, v_m_err) = intgr.quad(x2fx, -np.inf, cutoff)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, cutoff)

    sd_model = np.sqrt(var_model)

    return mean_model, sd_model


def err_vec(xvals, mu, sigma, cutoff, simple):
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
    mean_data, sd_data = data_moments(xvals)

    moments_data = np.array([[mean_data], [sd_data]])

    mean_model, sd_model = model_moments(mu, sigma, cutoff)

    moments_model = np.array([[mean_model], [sd_model]])

    if simple:
        err_vec = moments_model - moments_data
    else:
        err_vec = (moments_model - moments_data) / moments_data

    return err_vec


def criterion(params, *args):
    '''Written by revans; edited by emo.
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

    xvals, cutoff, w = args

    err = err_vec(xvals, mu, sigma, cutoff, simple=False)

    crit_val = np.dot(np.dot(err.T, w), err)

    return crit_val

def data_moms_1d(xvals):
    low_earners_data = len(incomes[incomes < 75000]) / len(incomes) * 100
    high_earners_data = len(incomes[incomes > 100000]) / len(incomes) * 100
    avg_earners_data = 100 - low_earners_data - high_earners_data

    return low_earners_data, avg_earners_data, high_earners_data

def model_moms_1d(mu, sigma, cutoff):

    xfx = lambda x: x * log_norm_pdf(x, mu, sigma, cutoff)

    (low_earners, l_e_err) = intgr.quad(xfx, 0, 75000)
    (avg_earners, a_e_err) = intgr.quad(xfx, 75000, 100000)
    (high_earners, h_e_err) = intgr.quad(xfx, 100000, cutoff)

    total_earners = low_earners + avg_earners + high_earners

    low_earners_model = low_earners / (total_earners + 1e-20) * 100
    avg_earners_model = avg_earners / (total_earners + 1e-20) * 100
    high_earners_model = high_earners / (total_earners + 1e-20) * 100

    return low_earners_model, avg_earners_model, high_earners_model



def err_vec_1d(xvals, mu, sigma, cutoff, simple):

    low_earners_data,avg_earners_data,high_earners_data = data_moms_1d(incomes)

    moments_data = np.array([[low_earners_data], [avg_earners_data], \
                             [high_earners_data]])

    low_earners_model,avg_earners_model,high_earners_model = model_moms_1d(mu,\
                      sigma, cutoff)

    moments_model = np.array([[low_earners_model], [avg_earners_model],\
                              [high_earners_model]])

    if simple:
        err_vec = moments_model - moments_data
    else:
        err_vec = (moments_model - moments_data) / moments_data * 100

    return err_vec



def criterion_1d(params, *args):
    mu, sigma = params

    xvals, cutoff, w = args

    err = err_vec_1d(xvals, mu, sigma, cutoff, simple=False)

    crit_val = np.dot(np.dot(err.T, w), err)

    return crit_val



def crit_sicko(params, *args):
    sick_2016, cutoff_sicko, w = args

    err = err_vec_sicko(sick_2016, params, cutoff_sicko)

    crit_val = np.dot(np.dot(err.T,w), err)

    return crit_val

def err_vec_sicko(sick_2016, betas, cutoff_sicko):
    '''Pull out the variables to make things easier.'''
    sick_weeks = sick_2016['Sick_Weeks_Taken']
    age = sick_2016['Age']
    kids = sick_2016['Children']
    mean_temp = sick_2016['Mean_Winter_Temp_F']

    b0 = betas[0]
    b1 = betas[1]
    b2 = betas[2]
    b3 = betas[3]

    model = b0 + (b1 * age) + (b2 * kids) + (b3 * mean_temp)

    err_vec = sick_weeks - model

    return err_vec


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
Plot a histogram of percentages of the income.txt data with 30 bins. Make
sure that the bins are weighted using the normed=True option. Make sure your
plot has correct x-axis and y-axis labels as well as a plot title.'''


print('\nPart a')
print('------\n')

incomes = np.loadtxt('incomes.txt')
fig_name_1a = 'Fig_1a'

count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')
#plt.xlim([40000, 140000])
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1a)
plt.savefig(output_path)

print('Saved {}.\n'.format(fig_name_1a))


'''1b
Estimate the parameters of the log normal distribution by generalized method
of moments. Use the average income and standard deviation of income as your two
moments. Use the identity matrix as the weighting matrix Wˆ. Plot the estimated
lognormal PDF against the histogram from part (a). Report the value of your GMM
criterion function at the estimated parameter values. Report & compare your two
data moments against the 2 model moments at the estimated parameter values.'''

print('\nPart b')
print('------\n')

cutoff = 150000

# Start with the values we got from MLE in PA2
mu_1b = 11.331
sig_1b = .212

bds = ((1e-10, None), (1e-10, None))

#trunc_pdf = log_norm_pdf(incomes, mu_1b, sig_1b, cutoff)

params_1b = np.array([mu_1b, sig_1b])
w_identity_1b = np.eye(2)
gmm_args = (incomes, cutoff, w_identity_1b)
results = opt.minimize(criterion, params_1b, args=(gmm_args),
                       method='L-BFGS-B', bounds=bds)
mu_GMM_1b, sig_GMM_1b = results.x
print('mu_GMM_1b  = {:.3f}'.format(mu_GMM_1b))
print('sig_GMM_1b =  {:.3f}'.format(sig_GMM_1b))
crit_GMM1 = results.fun.sum()
print('Value of GMM criterion function with those values:', crit_GMM1)
mean_data, sd_data = data_moments(incomes)
mean_model, sd_model = model_moments(mu_GMM_1b, sig_GMM_1b, cutoff)
err1 = err_vec(incomes, mu_GMM_1b, sig_GMM_1b, cutoff, False).reshape(2,)
print('\nMean of incomes = {:.3f}'.format(mean_data))
print('Mean of model  = {:.3f}'.format(mean_model))
print('SD of incomes = {:.3f}'.format(sd_data))
print('SD of model  = {:.3f}\n'.format(sd_model))
#print('Error vector=', err1)

fig_name_1b = 'Fig_1b'



# Plot the estimated GMM PDF
dist_pts = np.linspace(0.1, cutoff, 500)

pdf_1b = log_norm_pdf(dist_pts, mu_GMM_1b, sig_GMM_1b, cutoff)
plt.plot(dist_pts, pdf_1b, linewidth=3, color='magenta', \
         label='$GMM_1b: \mu = {:.3f}$, $\sigma ={:.3f}$'\
         .format(mu_GMM_1b,sig_GMM_1b))
plt.legend(loc='upper left')
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1b)
plt.savefig(output_path)
#plt.close()
print('Saved {}.\n'.format(fig_name_1b))




'''1c
Perform the two-step GMM estimator by using your estimates from part (b) with
2 moments to generate an estimator for the variance covariance matrix Ωˆ2step,
which you then use to get the two-step estimator for the optimal weighting
matrix Wˆ 2step. Report your estimates as well as the criterion function value
at these estimates. Plot the estimated lognormal PDF against the histogram from
part (a) and the estimated PDF from part (b). Report and compare your two data
moments against your two model moments at the estimated parameter values.'''

print('\nPart c')
print('------\n')


error_vector = err_vec(incomes, mu_GMM_1b, sig_GMM_1b, cutoff, False)

vcv = np.dot(error_vector,error_vector.T) / len(incomes)

w_hat_1c = np.linalg.pinv(vcv)


# Note that this takes a little time because the intgr.quad() commands
# are a little slow

params_1c = np.array([mu_GMM_1b, sig_GMM_1b])
args_1c = (incomes, cutoff, w_hat_1c)
results_1c = opt.minimize(criterion, params_1c, args=(args_1c),
                          method='L-BFGS-B', bounds=bds)
mu_GMM_1c, sig_GMM_1c = results_1c.x
print('mu_GMM_1c  = {:.3f}'.format(mu_GMM_1c))
print('sig_GMM_1c =  {:.3f}'.format(sig_GMM_1c))

crit_GMM_1c = results_1c.fun.sum()
print('Value of GMM criterion function with those values:', crit_GMM_1c)

mean_model_1c, sd_model_1c = model_moments(mu_GMM_1c, sig_GMM_1c, cutoff)

print('\nMean of incomes = {:.3f}'.format(mean_data))
print('Mean of model  = {:.3f}'.format(mean_model_1c))
print('SD of incomes = {:.3f}'.format(sd_data))
print('SD of model  = {:.3f}\n'.format(sd_model_1c))


fig_name_1c = 'Fig_1c'




pdf_1c = log_norm_pdf(dist_pts, mu_GMM_1c, sig_GMM_1c, cutoff)
plt.plot(dist_pts, pdf_1c, linewidth=1, color='cyan', \
         label='$GMM_1c: \mu = {:.3f}$, $\sigma ={:.3f}$'\
         .format(mu_GMM_1c,sig_GMM_1c))
plt.legend(loc='upper left')
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1c)
plt.savefig(output_path)
plt.close()
print('Saved {}.\n'.format(fig_name_1c))


'''1d
Now estimate the lognormal PDF to fit the data by GMM using different moments.
Use percent of individuals who earn less than $75,000, percent of individuals
who earn between $75,000 and $100,000, and percent of individuals who earn more
than $100,000 as your three moments. Use the identity matrix as your estimator
for the optimal weighting matrix. Plot your estimated lognormal PDF against the
histogram from part (a). Report the value of your GMM criterion function at the
estimated parameter values. Report and compare your three data moments against
your three model moments at the estimated parameter values.
'''
print('\nPart d')
print('------\n')

low_earners_data, avg_earners_data, high_earners_data = data_moms_1d(incomes)


params_1d = np.array([mu_1b, sig_1b])

w_identity_1d = np.eye(3)
gmm_args_1d = (incomes, cutoff, w_identity_1d)
results_1d = opt.minimize(criterion_1d, params_1d, args=(gmm_args_1d),
                       method='L-BFGS-B', bounds=bds)
mu_GMM_1d, sig_GMM_1d = results_1d.x


print('mu_GMM_1d  = {:.3f}'.format(mu_GMM_1d))
print('sig_GMM_1d =  {:.3f}'.format(sig_GMM_1d))

crit_GMM_1d = results_1d.fun.sum()
print('Value of GMM criterion function with those values:', crit_GMM_1d)

low_earners_model, avg_earners_model, high_earners_model = model_moms_1d(\
                   mu_GMM_1d, sig_GMM_1d, cutoff)

print('\nPercent of actual earners below $75,000: {:.9f}'.format(\
      low_earners_data))
print('Percent of model earners below $75,000:  {:.9f}'.format(low_earners_model))
print('\nPercent of actual earners between $75,000 and $100,000: {:.9f}'.format(\
      avg_earners_data))
print('Percent of model earners between $75,000 and $100,000:  {:.9f}'.format(\
      avg_earners_model))
print('\nPercent of actual earners above $100,000: {:.9f}'.format(\
      high_earners_data))
print('Percent of model earners above $100,000:  {:.9f}\n'.format(\
      high_earners_model))


fig_name_1d = 'Fig_1d'

pdf_1d = log_norm_pdf(dist_pts, mu_GMM_1d, sig_GMM_1d, cutoff)

count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')

plt.plot(dist_pts, pdf_1d, linewidth=2, color='turquoise', \
         label='$GMM_1d: \mu = {:.3f}$, $\sigma ={:.3f}$'\
         .format(mu_GMM_1d,sig_GMM_1d))
plt.legend(loc='upper left')
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1d)
plt.savefig(output_path)
plt.close()
print('Saved {}.\n'.format(fig_name_1d))


'''1e
Perform the two-step GMM estimator by using the estimates from part (d) with
three moments to generate an estimator for the variance-covariance matrix
Ωˆ2step, which you then use to get the two-step estimator for the optimal
weighting matrix Wˆ 2step. Report your estimates as well as the criterion
function value at these estimates. Plot your estimated log-normal PDF against
the histogram from part (a) and the estimated PDF from part (d). Report and
compare your three data moments against your three model moments at the
estimated parameter values.'''

print('\nPart e')
print('------\n')


error_vec_1e = err_vec_1d(incomes, mu_GMM_1d, sig_GMM_1d, cutoff, False)

vcv_1e = np.dot(error_vec_1e,error_vec_1e.T) / len(incomes)

w_hat_1e = np.linalg.pinv(vcv_1e)

params_1e = np.array([mu_GMM_1b, sig_GMM_1b])
args_1e = (incomes, cutoff, w_hat_1e)
results_1e = opt.minimize(criterion_1d, params_1e, args=(args_1e),\
                          method='L-BFGS-B', bounds=bds)

mu_GMM_1e, sig_GMM_1e = results_1e.x
print('mu_GMM_1e  = {:.3f}'.format(mu_GMM_1e))
print('sig_GMM_1e =  {:.3f}'.format(sig_GMM_1e))

crit_GMM_1e = results_1e.fun.sum()
print('Value of GMM criterion function with those values:', crit_GMM_1e)

low_earners_mdl_1e, avg_earners_mdl_1e, high_earners_mdl_1e = model_moms_1d(\
                   mu_GMM_1e, sig_GMM_1e, cutoff)

print('\nPercent of actual earners below $75,000: {:.9f}'.format(\
      low_earners_data))
print('Percent of model earners below $75,000:  {:.9f}'.format(low_earners_mdl_1e))
print('\nPercent of actual earners between $75,000 and $100,000: {:.9f}'.format(\
      avg_earners_data))
print('Percent of model earners between $75,000 and $100,000:  {:.9f}'.format(\
      avg_earners_mdl_1e))
print('\nPercent of actual earners above $100,000: {:.9f}'.format(\
      high_earners_data))
print('Percent of model earners above $100,000:  {:.9f}\n'.format(\
      high_earners_mdl_1e))







fig_name_1e = 'Fig_1e'

pdf_1e = log_norm_pdf(dist_pts, mu_GMM_1e, sig_GMM_1e, cutoff)

count, bins, ignored = plt.hist(incomes, 30, normed = True)
plt.title('MACSS Graduates\' Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Proportion of Incomes')

plt.plot(dist_pts, pdf_1e, linewidth=2, color='purple', \
         label='$GMM_1e: \mu = {:.3f}$, $\sigma ={:.3f}$'\
         .format(mu_GMM_1e,sig_GMM_1e))
plt.legend(loc='upper left')
plt.tight_layout()
#plt.show()
output_path = os.path.join(output_dir, fig_name_1e)
plt.savefig(output_path)
plt.close()
print('Saved {}.\n'.format(fig_name_1e))




'''_________________________________________________________________________'''
'''_________________________________________________________________________'''
'''_________________________________________________________________________'''

print('-----PROBLEM 2-----')
'''2
Linear regression and GMM (3 points). Ordinary least squares (OLS) is the most
common estimator for a linear regression. However, in this exercise, you will
implement the more general GMM estimator of the linear regression. Assume the
following linear regression model for determining what effects the number of
weeks that an individual i is sick during the year (sicki).

sicki = β0 + β1agei + β2childreni + β3temp winteri + εi

The parameters (β0, β1, β2, β3, ) are the parameters of the model that we want
to estimate. Note that we don not have to make any assumptions about the
distribution of the error terms εi. The variable agei gives the age of
individual i at the end of 2016 (including fractions of a year). The variable
childreni states how many children individual i had at the end of 2016. And the
variable temp winteri is the average temperature during the months of January,
February, and December 2016 for individual i. The data for this model are in
the file sick.txt, which contains comma-separated values of 200 individuals for
four variables (sicki, agei, childreni, temp winteri) with variable labels in
the first row.
'''


print('\nPart a')
print('------\n')

'''2a
Estimate the parameters of the model (β0, β1, β2, β3) by GMM by solving the
minimization problem of the GMM criterion function. Use the identity matrix as
the estimator for the optimal weighting matrix. Treat each of the 200 values of
the variable sicki as your data moments m(xi) (200 data moments). Treat the
predicted or expected sick values from your model as your model moments (200
model moments),
m(xi|β0, β1, β2, β3) = β0 + β1agei + β2childreni + β3temp winteri
where xi is short hand for the data. Let the error function of the moments
be the simple difference (not percent difference) of the data moments from
the model moments. These are equal to the error terms of the linear regression
equation
e(xi|β0, β1, β2, β3) = sicki − β0 − β1agei − β2childreni − β3temp winteri = εi
Use these error functions in your criterion function to estimate the model
parameters (β0, β1, β2, β3) by GMM. This is a more general version of what OLS
does. It minimizes the distance between the model moments and the data moments.
It minimizes the sum of squared error terms. Report your estimates and report
the value of your GMM criterion function. In this case, the GMM criterion
function value evaluated at the optimal parameter values is simply the sum of
squared errors.
'''

sicko = 'sick.txt'
cols = ['Sick_Weeks_Taken', 'Age', 'Children', 'Mean_Winter_Temp_F']

sick_2016 = pd.read_csv(sicko, names = cols, skiprows = 1)

params_sicko = np.array([0,0,0,0])

w_identity_2a = np.eye(len(sick_2016))
cutoff_sicko = 52 # weeks in a year
args_sicko = (sick_2016, cutoff_sicko, w_identity_2a)

results_sicko = opt.minimize(crit_sicko, params_sicko, args=(args_sicko), \
                method = 'L-BFGS-B')#, bounds = bds)
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results_sicko.x
crit_sicko = results_sicko.fun.sum()


print('The parameter estimates are:')
print('β_0 =  {:.3f}'.format(beta0_GMM))
print('β_1 =  {:.3f}'.format(beta1_GMM))
print('β_2 =  {:.3f}'.format(beta2_GMM))
print('β_3 = {:.3f}'.format(beta3_GMM))
print('\nValue of GMM criterion function with those values:', crit_sicko)
