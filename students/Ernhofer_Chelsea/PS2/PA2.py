#persp-mode PA2


import numpy as np
import scipy.stats as sts 
import scipy as sp
import scipy.optimize as opt
import os 
import matplotlib.pyplot as plt
import pandas as pd 
import math 


def ln_pdf(xvals, mu, sigma, cutoff):
	if cutoff == "None":
		prob_notcut = 1.0
	else:
		prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

	pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut

	return pdf_vals


def log_lik_norm(xvals, mu, sigma, cutoff):
	pdf_vals = ln_pdf(xvals, mu, sigma, cutoff)
	ln_pdf_vals = np.log(pdf_vals)
	log_lik_val = ln_pdf_vals.sum()

	return log_lik_val


def crit(params, *args):
	mu, sigma = params
	xvals, cutoff = args
	log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
	neg_log_lik_val = -log_lik_val

	return neg_log_lik_val

def save_image(name):
    '''
    '''
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'ernhofer_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path)
    plt.close()


def norm_pdf(xvals, mu, sigma):
    vals = (1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    return vals

def log_like_2(data, b0, b1, b2, b3, sigma):
    error = data['\ufeffsick'] - (b0+ b1*data['age'] + b2*data['children']+ b3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

def crit_2(params, *args):
    b0, b1, b2, b3,sigma = params
    data_2 = args[0]
    log_lik_val = log_like_2(data_2, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

#problem 1

income = np.loadtxt("incomes.txt")

#A- plot histogram, 30 bins, percentages
print("\n**********\nQuestion 1\n**********\n")
print("Part A")

weights = (100/ len(income)) * np.ones(len(income))

count, bins, patches = plt.hist(income, 30, weights = weights)
plt.title('Incomes', fontsize=20)
plt.xlabel('Incomes')
plt.ylabel('Percentages')

save_image("Histogram")

#B- lognormal PDF with specific values- it's terrible!

print("\nPart B")

mu_1 = 9.0
sig_1 = 0.3

dist_pts = np.linspace(0.000001, 150000, 500)

plt.plot(dist_pts, ln_pdf(dist_pts, mu_1, sig_1, 150000),
         linewidth=2, color='g', label='2: $\mu$=9,$\sigma$=0.3')
plt.title("Plot of Lognormal PDF")
plt.legend(loc='upper left')
plt.xlabel('Income')

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

print(mu_MLE, sig_MLE)

count, bins, ignored = plt.hist(income, 30, normed=True)
plt.xlabel('Income')
plt.title("Plot of Lognormal PDF")

# Plot the MLE estimated distribution
plt.plot(dist_pts, ln_pdf(dist_pts, mu_1, sig_1, "None"),
         linewidth=2, color='g', label='2: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper left')

plt.plot(dist_pts, ln_pdf(dist_pts, mu_MLE, sig_MLE, "None"),
         linewidth=2, color='r', label='3: $\mu$=11.33,$\sigma$=0.21')
plt.legend(loc='upper left')

save_image("Second PDF")

ll_value = log_lik_norm(income, mu_MLE, sig_MLE, "None")


print("Log-likelihood: ", ll_value)

#variance/covariance matrix

vcv_mle = results.hess_inv
stderr_mu_mle = np.sqrt(vcv_mle[0,0])
stderr_sig_mle = np.sqrt(vcv_mle[1,1])
print('Variance/Covarience Matrix: ', vcv_mle)

#D- likelihood ratio test for part (b) 
print("\nPart D")
log_lik_h0 = log_lik_norm(income, mu_1, sig_1, 450)
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, 450)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print("It is unlikely that the incomes data came from this distribution")


#E- answer questions 
print("\nPart E")

test = ln_pdf(dist_pts, mu_MLE, sig_MLE, 150000)

cdf100 = 100* (1 - sts.lognorm.cdf(100000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE)))

print("Percentage of students will earn more than 100,000: ", round(cdf100, 2))

cdf75 = 100* sts.lognorm.cdf(75000, s= sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))

print("Percentage of students will earn less than 75,000: ", round(cdf75, 2))

#plt.plot(dist_pts, cdf, linewidth=2, color='r', label='3: $\mu$=11.33,$\sigma$=0.21')


#QUESTION 2
print("\n**********\nQuestion 2\n**********\n")
data_2 = pd.read_csv('sick.txt', header = 0)

#Find MLE
# set initial values as per assignment
b0_0 = 1
b1_0 = 0
b2_0 = 0
b3_0 = 0
sigma_0 = .1

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
print("Part A")
print('Beta0 =', b0, '\nBeta1 =', b1, '\nBeta2 =', b2, '\nBeta3 =', b3, '\nSigma =', sigma)
print('Variance-Covariance matrix: \n',hessian.todense())
# todense converts arrays to matrix form
print('Log-likelihood: ', log_like_2(data_2, *results.x))

#Part B
log_lik_h0 = log_like_2(data_2, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_2(data_2, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print("\nPart B")
print('chi squared of H0 with 5 degrees of freedom p-value: ', pval_h0)
print("It is unlikely that are, number of children, and average winter temp have no effect on sick days")




