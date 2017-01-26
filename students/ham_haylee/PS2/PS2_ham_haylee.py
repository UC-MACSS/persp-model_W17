# Haylee Ham


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
import pandas as pd

def save_image(name):
    '''
    --------------------------------------------------------------------
    The function takes the name of a file and creates a path and directory
    if necessary to save the file. It then saves the file and closes it.
    --------------------------------------------------------------------
    INPUTS:
    name = string, the desiredname of the file

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    FILES CREATED BY THIS FUNCTION: directory called 'ham_images' and
    image file called name

    RETURNS: None
    --------------------------------------------------------------------
    '''
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'ham_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path)
    plt.close()

# a
def plot_hist(inc_data):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and standard
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
    inc_count = len(inc_data)
    inc_weights = (100/inc_count) * np.ones(inc_count)
    plt.hist(inc_data, 30, weights = inc_weights, normed = False)
    plt.title('MACSS Graduate Incomes: 2018-2020', fontsize=20)
    plt.xlabel('Income')
    plt.ylabel('Percentage of Incomes')

    save_image('Fig_1a')

def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and standard
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

    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * (np.exp( - \
    (np.log(xvals) - mu)**2 / (2 * sigma**2)))) / prob_notcut

    return pdf_vals


def log_lik_norm(xvals, mu, sigma, cutoff):
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()

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
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
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
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

def log_like_2(data, b0, b1, b2, b3, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters betas and sigma.
    --------------------------------------------------------------------
    INPUTS:
    data   = dataframe
    b0, b1, b2, b3, sigma = scalars, inital estimates for these parameters

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for parameters and sigma
                  corresponding to data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for parameters and sigma corresponding to data
    log_lik_val = scalar, value of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    error = data['sick']- (b0+ b1*data['age'] + b2*data['children']+ b3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma, 'None')
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

def crit_2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params  = array, initial estimates of the parameters
    args    = dataframe

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_like_2()

    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3, sigma = params
    sick_data = args[0]
    log_lik_val = log_like_2(sick_data, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

def norm_pdf(xvals, mu, sigma, cutoff):
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

    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)

    return pdf_vals

if __name__ == '__main__':
    # Question 1
    # a
    print("Question 1")
    print("#1 Part A, see Fig_1a")
    inc_data = np.loadtxt("incomes.txt")
    plot_hist(inc_data)

    # b
    print("\n\n#1 Part B, see Fig_1b")
    dist_inc = np.linspace(0.0000000001, 150000, 500)
    mu_1 = 9
    sig_1 = 0.3
    cutoff = 'None'

    plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_1, sig_1, cutoff),
         linewidth=2, color='r', label='1: $\mu$={:d},$\sigma$={:.2f}'.format(mu_1, sig_1))
    plt.legend(loc='upper right')
    plt.ylabel('f(x|mu, sigma)')
    plt.title('Lognormal CDF, mu of 9 and sigma of 0.3', fontsize=15)
    save_image('Fig_1b')

    # Reportig the log likelihood value for values given
    log_lik_income = -1 * log_lik_norm(inc_data, mu_1, sig_1, cutoff)
    print("The log likelihood value for mu = 9.0 and sigma = 0.3 is", log_lik_income)

    # c
    print("\n\n#1 Part C, see Fig_1c")
    # estimate parameters by maximum likelihood
    mu_init = 11
    sig_init = 0.3

    params_init = np.array([mu_init, sig_init])
    mle_args = (inc_data, 'None')
    results = opt.minimize(crit, params_init, args=(mle_args))
    mu_MLE, sig_MLE = results.x
    print("The maximum likelihood estimate for mu is", mu_MLE, "and the estimate for sigma is", sig_MLE)

    # report the value of the likelihood function
    log_lik_income = -1 * log_lik_norm(inc_data, mu_MLE, sig_MLE, cutoff)
    print("The log likelihood value for the ML estimates is", log_lik_income)

    # plot the original histogram
    count, bins, ignored = plt.hist(inc_data, 30, normed=True)
    plt.title('Incomes of MACSS Graduates with lognormal CDFs', fontsize=15)
    plt.xlabel('Income')
    plt.ylabel('f(x|mu, sigma)')

    # Plot the first terrible distribution
    plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_1, sig_1, "None"),
             linewidth=2, color='r', label='1: $\mu$={:d},$\sigma$={:.2f}'.format(mu_1, sig_1))
    plt.legend(loc='upper right')

    # Plot the MLE estimated distribution
    plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_MLE, sig_MLE, "None"),
             linewidth=2, color='g', label='2: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_MLE, sig_MLE))
    plt.legend(loc='upper right')
    save_image("Fig_1c")

    # print the variance/covariance matrix
    vcv_mle = results.hess_inv
    stderr_mu_mle = np.sqrt(vcv_mle[0,0])
    stderr_sig_mle = np.sqrt(vcv_mle[1,1])
    print("The variance/covariance matrix for the ML estimates is", vcv_mle)

    # d
    print("\n\n#1 Part D")
    log_lik_h0 = log_lik_norm(inc_data, mu_1, sig_1, 'None')
    log_lik_mle = log_lik_norm(inc_data, mu_MLE, sig_MLE, 'None')
    LR_val = 2 * (log_lik_mle - log_lik_h0)
    pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
    print('Chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
    print("It is unlikely that the incomes data came from this distribution")

    # e
    print("\n\n#1 Part E")

    cdf_100000 = 100 * (1 - sts.lognorm.cdf(100000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE)))
    print("The probability that I will earn more than $100,000 is", round(cdf_100000, 2), "%")

    cdf_75000 = 100* sts.lognorm.cdf(75000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
    print("The probability that I will earn less than $75,000 is", round(cdf_75000, 2), "%\n\n")

    #Question 2
    # a
    print("\n\nQuestion 2")
    print("\n\n#2 Part A")
    sick_data = pd.read_csv('sick.txt', header = 0)

    # set initial estimated values
    b0_0 = 1
    b1_0 = 0
    b2_0 = 0
    b3_0 = 0
    sigma_0 = .1

    params_init = np.array([b0_0, b1_0, b2_0, b3_0, sigma_0])

    # First use the SLSQP method, which gives Success while L-BFGS-B never seems to,
    # and then plugging those results into the L-BFGS-B method in order to get the Hessian
    results = opt.minimize(crit_2, params_init, args=sick_data, method = "SLSQP")
    b0, b1, b2, b3, sigma = results.x

    results = opt.minimize(crit_2,  results.x, args=sick_data, method = "L-BFGS-B")
    b0, b1, b2, b3, sigma = results.x

    # print the final MLE estimators
    print('Beta0 =', b0, ' Beta1 =', b1, 'Beta2 =', b2, 'Beta3 =', b3, 'Sigma =', sigma)

    # The variance-covariance matrix from the Hessian
    hessian = results.hess_inv
    print('The variance-covariance matrix is: ', hessian.todense())

    # print log likelihood value
    print('Log-likelihood: ', log_like_2(sick_data, *results.x))

    # b
    print("\n\n#2 Part B")
    # identify the hypothesis that the data comes from the supposed values of the parameters
    log_lik_h0 = log_like_2(sick_data, b0_0, b1_0, b2_0, b3_0, sigma_0)
    # identify the log likelihood value with the ML estimates
    log_lik_mle = log_like_2(sick_data, *results.x)
    # find the difference
    LR_val = 2 * (log_lik_mle - log_lik_h0)
    # find the p value find the CDF of the difference
    pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
    print('chi squared of H0 with 5 degrees of freedom p-value = {:f}'.format(pval_h0))
