import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt

#Question 1
#loading dataset
income = np.loadtxt('incomes.txt')
income

#part (a)
#setting up weights for density histogram
weights = (1 / income.shape[0]) * np.ones_like(income)
#plotting histogram
count, bins, patches = plt.hist(income, 30, weights = weights)
#labelling histogram
plt.title('MACS Graduate Annual Incomes: 2018-2020', fontsize=20)
plt.xlabel('Annual Income')
plt.ylabel('Proportion of Graduates')

#part (b)
#defining function that generates values of a log normal pdf
def log_norm_pdf(xvals, mu, sigma, cutoff):
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

    pdf_vals    = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)

    return pdf_vals

#creating 200 draws from (0, 150000)
dist_inc = np.linspace(0, 150000, 200)
#plotting lognormal pdf for mu = 9 and sig = 0.3
mu_1 = 9
sig_1 = 0.3
plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_1, sig_1, 150000),
         linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper left')

#defining log likelihood function for the lognormal distribution
def log_lik_norm(xvals, mu, sigma, cutoff):
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

#calculating the log likelihood value of mu=9 and sigma=0.3
print('Log-likelihood value: ', log_lik_norm(income, mu_1, sig_1, 150000))

#part (c)
# setting up the criterion function in the minimisation problem
def crit(params, *args):
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

#the initial guesses parameters
mu_init = 11.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
mle_args = (income, 150000)
results = opt.minimize(crit, params_init, args=(mle_args))
mu_MLE, sig_MLE = results.x
#reporting mu_MLE and sig_MLE
print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
#value of likelihood function
print('Lognormal likelihood value: ', log_lik_norm(income, mu_MLE, sig_MLE, 150000))

# plot part (a), (b), (c) together
dist_inc = np.linspace(0, 150000, 200)
#normed histogram from part (a)
count, bins, patches = plt.hist(income, 30, normed=True)
#lognorm pdf of ML estimated parameters
plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_MLE, sig_MLE, 150000),
         linewidth=2, color='y', label='1: $\mu$=11.331,$\sigma$=0.212')
plt.legend(loc='upper left')
#lognormal pdf from part (b)
mu_1 = 9
sig_1 = 0.3
plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_1, sig_1, 150000),
         linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper left')

#variance-covariance matrix
results
vcv_mle = results.hess_inv
print('VCV(MLE) = ', vcv_mle)

#part (d)
#likelihood ratio test
log_lik_h0 = log_lik_norm(income, mu_1, sig_1, 150000)
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, 150000)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)


#part (e)
#using MLE log normal c.d.f
#probability that the graduate earns less than $75,000
print('Probability a graduate earning less than $75,000:', sts.lognorm.cdf(75000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE)))
#probability that the graduate earns more than $100,000
print('Probability a graduate earning more than $100,000:', 1 - sts.lognorm.cdf(100000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE)))

#Question 2
#import data
sick = np.loadtxt("sick.txt", delimiter = ",", skiprows = 1)

#finding MLE
#defining function that generates values of a normal pdf
def norm_pdf(xvals, mu, sigma, cutoff):
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)
    return pdf_vals

#defining log likelihood function for the lognormal distribution
def log_lik_norm(xvals, mu, sigma, cutoff):
    pdf_vals = norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

#criterion function for minimisation problem
def crit2(params, *args):
    beta_0, beta_1, beta_2, beta_3, sig = params
    xvals, cutoff = args
    xvals_y = xvals[:,0] - (beta_0 + beta_1*xvals[:,1] + beta_2*xvals[:,2] + beta_3*xvals[:,3])
    log_lik_val = log_lik_norm(xvals_y, 0, sig, cutoff)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

#initial guesses of parameters
beta_0_init = 0.01
beta_1_init = 0.001
beta_2_init = 0.4
beta_3_init = -0.009
sig_init = 0.05
params_init = np.array([beta_0_init, beta_1_init, beta_2_init, beta_3_init, sig_init])
mle_args = (sick, float('inf'))
bounds =((None, None), (None, None), (None, None), (None, None), (0.001, None))
#getting MLE
#using SLSQP for bounded minimization
#using SLSQP for initial guesses then plugging values to L-BFGS-B minimisation
results = opt.minimize(crit, params_init, args=(mle_args), bounds=bounds, method = 'SLSQP')
results2 = opt.minimize(crit, results.x, args=(mle_args), method = 'L-BFGS-B', bounds = bounds)
#ML Estimated parameters
beta_0 = results2.x[0]
beta_1 = results2.x[1]
beta_2 = results2.x[2]
beta_3 = results2.x[3]
sig = results2.x[4]
#variance-covariance matrix
vcv_mle = results2.hess_inv
print('VCV(MLE) = ', vcv_mle.todense())
y = (sick[:,0] - (beta_0 + beta_1*sick[:,1] + beta_2*sick[:,2] + beta_3*sick[:,3]))
log_lik_value_2a = log_lik_norm(y, 0, 0.00301768, float('inf'))
print('Loglikelihood value=', log_lik_value_2a)

#part (b)
#likelihood ratio test
h0 = sick_i - 1.0
log_lik_h0 = log_lik_norm(h0, 0, 0.01, float('inf'))
log_lik_mle = log_lik_norm(y, 0, 0.00301768, float('inf'))
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
