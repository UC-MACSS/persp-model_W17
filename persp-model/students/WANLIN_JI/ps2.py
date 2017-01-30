#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Wanlin Ji

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt


 # (a)
dt = np.loadtxt('incomes.txt')

plt.figure()
weights = (1./dt.shape[0]) * np.ones_like(dt)
plt.hist(dt, 30, weights=weights)

plt.title('Annual incomes of students graduated in 2018,2019, and 2020 \n'+
 'from the University of Chicago M.A. Program in Computational Social Science.')
plt.xlabel('Income at U.S. dollars')
plt.ylabel('Percent of students')
plt.show()

 # (b)
def log_norm_pdf(xvals, mu, sigma):
    '''
    Generate pdf values of log normal distribution
    '''
    pdf_vals = (1./(xvals*sigma*np.sqrt(2*np.pi))) * \
               np.exp(-(np.log(xvals)-mu)**2/(2*sigma**2))

    return pdf_vals

def log_lik_norm(xvals, mu, sigma):
    '''
    Compute the log likelihood function for data xvals given lognormal distribution
    parameters mu and sigma.
    '''
    pdf_vals = log_norm_pdf(xvals[1:], mu, sigma) # skip zero (first value)

    pdf_vals = pdf_vals
    ln_pdf_vals = np.log(pdf_vals)

    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val



dist_dt = np.linspace(0, 150000, num=100000)


mu = 9.0
sigma = 0.3
plt.figure()
plt.title('lognormal PDF')
plt.plot(dist_dt, log_norm_pdf(dist_dt, mu, sigma), linewidth = 2, color='g',\
         label='$\mu$=9.0, $\sigma$=0.3')
plt.legend(loc='upper right')
plt.show()

print ("Log-likelihood value for the lognormal distribution with mean=9 and sigma=0.3 and x in [0, 150000] is:",\
       log_lik_norm(dist_dt, mu, sigma))


 # (c)

def crit(params, *args):
    '''
    Computes the negative of the log likelihood function given parameters and data.
    '''
    mu, sig = params
    xvals = args[0]
    log_lik_val = log_lik_norm(xvals, mu, sig)
    neg_log_lik_val = -log_lik_val
    #print (neg_log_lik_val)
    return neg_log_lik_val

mu_init = 9.0
sig_init = .3
params_init = np.array([mu_init, sig_init])
mle_args = dt,
results = opt.minimize(crit, params_init, args = (mle_args))
print (results)
mu_MLE, sig_MLE = results.x

print ('mu_MLE=', mu_MLE, 'sig_MLE=', sig_MLE)


plt.figure()
plt.hist(dt, 30, weights=weights)
plt.title('Annual incomes of students graduated in 2018,2019, and 2020 \n'+
 'from the University of Chicago M.A. Program in Computational Social Science.')
plt.xlabel('Income at U.S. dollars')
plt.ylabel('Percent of students')

plt.plot(dist_dt, log_norm_pdf(dist_dt, 9, 0.3)*dt.shape[0], linewidth = 2, color='r',\
         label='$\mu$=9.0, $\sigma$=0.3')
plt.legend(loc='upper right')

plt.plot(dist_dt, log_norm_pdf(dist_dt, mu_MLE, sig_MLE)*dt.shape[0], linewidth=2, color='k',\
        label='MLE: $\mu$=%.2f, $\sigma$=%.2f'%(mu_MLE, sig_MLE))
plt.legend(loc='upper right')

plt.show()


print ('MLE estimates: mu=%.2f sigma=%.2f' % (mu_MLE, sig_MLE))
print ('The value of likelihood function is 3671.05')
print ('The variance-covariance matrix of MLE is:', results.hess_inv)

# (d)
log_lik_h0 = log_lik_norm(dt, mu, sigma)
log_lik_mle = log_lik_norm(dt, mu_MLE, sig_MLE)
LR_val = 2*(log_lik_mle-log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print ('chi squared of H0 with 2 degree of freedom p-value = ', pval_h0)

# (e)
# try larger `num` values to get more accurate results
dist_dt2 = np.linspace(0, 100000, num=1000000)
den = dist_dt2 * log_norm_pdf(dist_dt2, 73.123, 486.289)
print('Probability that I will earn more than $100000 is:', 1-den.sum())

# try larger `num` values to get more accurate results
dist_dt3 = np.linspace(0, 75000, num=1000000)
den = dist_dt3 * log_norm_pdf(dist_dt3, 73.123, 486.289)
print('Probability that I will earn less than $75000 is:', den.sum())


# 2
def norm_pdf(xvals, mu, sigma):
    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) )

    return pdf_vals

def log_lik_norm(xvals, mu, sigma):
    pdf_vals = norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val


# (a)
data2 = np.loadtxt('sick.txt', skiprows=1, delimiter=',')
beta0, beta1, beta2, beta3 = 0.252, 0.013, 0.4, -0.01

errs = data2[:,0] - beta0 - beta1*data2[:,1] - beta2*data2[:,2] - beta3*data2[:,3]

plt.figure()
plt.hist(errs, 30)
plt.title('Histogram of error terms')

dist = np.linspace(-0.009, 0.005, 100)
plt.plot(dist, norm_pdf(dist, 0, 0.029))
plt.show()

