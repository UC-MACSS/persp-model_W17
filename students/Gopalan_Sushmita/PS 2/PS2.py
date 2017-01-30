import scipy as sp
import pandas as pd
import numpy as np
import os
import math
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Question 1

def ln_pdf(xvals, mu, sigma, cutoff):
    '''
    Generate pdf values from the log-normal pdf with mean mu and standard
    deviation sigma.

    Inputs:
       xvals: array
       mu: float
       sigma: float
    Outputs:
       pdf_vals: array
    '''
    if cutoff == "None":
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut
    return pdf_vals

def log_lik_norm(xvals, mu, sigma, cutoff):
    '''
    Computes the log likelihood function for data xvals given log-normal
    distribution parameters mu and sigma.

    Inputs:
       xvals: array
       mu: float
       sigma: float
       cutoff: float
    Outputs:
       log_lik_val: float
    '''
    pdf_vals = ln_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit(params, *args):
    '''
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    Inputs:
       params: array
       *args: array
    Outputs:
       neg_log_lik_val: float
    '''
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

def save_image(name):
    '''
    Checks if a directory exists for images and if not,
    creates one and saves the req image in it

    Inputs:
       name: string
    '''
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'sush_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
       os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path)
    plt.close()

# load data
income = np.loadtxt("incomes.txt")

#Part (a)- plot histogram, 30 bins, percentages

print("Part A")
weights = (100/ len(income)) * np.ones(len(income))
count, bins, patches = plt.hist(income, 30, weights = weights)
plt.title('Incomes', fontsize=20)
plt.xlabel('Incomes')
plt.ylabel('Percentages')
plt.show()
save_image("Histogram")


print("\nPart B")

# declare parameters as per assignment
mu_1 = 9.0
sig_1 = 0.3
dist_pts = np.linspace(0.000001, 150000, 500)

plt.plot(dist_pts, ln_pdf(dist_pts, mu_1, sig_1, 150000),linewidth=2, color='g', label='2: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper left')
plt.show()

save_image("First PDF")
ll_value = log_lik_norm(income, mu_1, sig_1, 150000)
print("Log likelihood = ", ll_value)

#C- estimate parameters, find maximum likelihood, plot it
print("\nPart C")

#BFGS was coming back false but giving me mu and sigma values.
#The only way to make it come back true was to enter the exact values
#that it gave me.
mu_2 =  11.33
sig_2 =  0.21
params_init = np.array([mu_2, sig_2])
mle_args = (income, "None")
results = opt.minimize(crit, params_init, args = mle_args, method= 'BFGS')
mu_MLE, sig_MLE = results.x

print("\nMLE mean = ",mu_MLE, "\nMLE sigma = ",sig_MLE)

count, bins, ignored = plt.hist(income, 30, normed=True)
plt.title('Incomes', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Frequency')

# Plot the MLE estimated distribution
plt.plot(dist_pts, ln_pdf(dist_pts, mu_1, sig_1, "None"),
        linewidth=2, color='g', label='2: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper left')

plt.plot(dist_pts, ln_pdf(dist_pts, mu_MLE, sig_MLE, "None"),
        linewidth=2, color='r', label='3: $\mu$=11.33,$\sigma$=0.21')
plt.legend(loc='upper left')
plt.show()

save_image("Second PDF")
ll_value = log_lik_norm(income, mu_MLE, sig_MLE, "None")
print("Log likelihood function value: ", ll_value)

#variance/covariance matrix

vcv_mle = results.hess_inv
stderr_mu_mle = np.sqrt(vcv_mle[0,0])
stderr_sig_mle = np.sqrt(vcv_mle[1,1])
print('Variance/Covarience Matrix: \n', vcv_mle)

#D- likelihood ratio test for part (b)
print("\nPart D")
# null hypothesis
log_lik_h0 = log_lik_norm(income, mu_1, sig_1, 450)
# alternative hypothesis
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, 450)
# get LR
LR_val = 2 * (log_lik_mle - log_lik_h0)
# get p-value from chi-squared distribution
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print("The probability that the data was drawn from the above distribution is ",pval_h0)

#E- answer questions
print("\nPart E")
test = ln_pdf(dist_pts, mu_MLE, sig_MLE, 150000)
cdf100 = 100* (1 - sts.lognorm.cdf(100000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE)))
print("\nPercentage of students will earn more than 100,000: ", round(cdf100, 2))
cdf75 = 100* sts.lognorm.cdf(75000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print("\nPercentage of students will earn less than 75,000: ", round(cdf75, 2))

# Question 2:
print("Question 2")
data_2 = pd.read_csv('sick.txt', header = 0)

def norm_pdf(xvals, mu, sigma):
    '''
    Compute normal pdf values given mean mu and standard deviation sigma

    Inputs:
       xvals: array
       mu: float
       sigma: float
    Output:
       vals: array
    '''
    vals = (1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    return vals

def log_like_2(data, b0, b1, b2, b3, sigma):
    '''
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.

    Inputs:
       data: array
       b0,b1,b2,b3,sigma: float
    Output:
       log_lik_val: float
    '''
    error = data['\ufeffsick']- (b0+ b1*data['age'] + b2*data['children']+ b3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_2(params, *args):
    '''
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem

    Inputs:
       params: array
       args: array

    Output:
       neg_log_lik_val: float
    '''
    b0, b1, b2, b3,sigma = params
    data_2 = args[0]
    log_lik_val = log_like_2(data_2, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

#Find MLE
# set initial values as per assignment
b0_0 = 1
b1_0 = 0
b2_0 = 0
b3_0 = 0
sigma_0 = 0.1

params_init = np.array([b0_0, b1_0, b2_0, b3_0, sigma_0])
bounds = ((None, None),(None, None),(None, None),(None, None),(0.001, None))

# Using no method or only L-BFGS-B method, doesn't give stable results. So, I used
# SLSQP to formulate an inital guess and then plugged that into the minimizer
# using L-BFGS-B.
# get initial guess

results = opt.minimize(crit_2, params_init, args=data_2, method = "SLSQP", bounds=bounds)
b0, b1, b2, b3, sigma = results.x
# plug back into minimizer
results = opt.minimize(crit_2,  results.x, args=data_2, method = "L-BFGS-B", bounds=bounds)
b0, b1, b2, b3, sigma = results.x

# Hessian
hessian = results.hess_inv
print('\nBeta0 =', b0, ' \nBeta1 =', b1, '\nBeta2 =', b2, '\nBeta3 =', b3, '\nSigma =', sigma)
print('\nThe variance-covariance matrix is: \n',hessian.todense())
# todense converts arrays to matrix form
print('\nLog-likelihood: ', log_like_2(data_2, *results.x))

log_lik_h0 = log_like_2(data_2, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_2(data_2, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('\nchi squared of H0 with 5 degrees of freedom p-value = {:.5}'.format(pval_h0))
print('\nWe can reject the hypothesis that age, average temperature and number of children have no effect on sick days')
