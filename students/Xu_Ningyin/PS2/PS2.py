# Perspectives on Computational Modeling
# Problem Set 2
# Name: Ningyin Xu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt


### Problem 1a:
def problem1a(data, num_bins):
	'''
	--------------------------------------------------------------------
	A function that plots the annual incomes of students who graduated in
	2018, 2019, and 2020 from the University of Chicago MACSS program.
	--------------------------------------------------------------------

	INPUTS:
		data: income data
		num_bins: number of bins along x-axis

	RETURNS:
		None.
	--------------------------------------------------------------------
	'''
	graph = True
	if graph:
		weights = (1 / data.shape[0]) * np.ones_like(data)
		count, bins, ignored = plt.hist(data, num_bins, weights = weights)
		plt.title('MACSS student income: 2018-2020', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('Percent')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1a')
		plt.savefig(output_path)
		plt.close()


### Problem 1b:
def lognorm_pdf(xvals, mu, sigma):
	'''
	--------------------------------------------------------------------
	Generate pdf values from the lognormal pdf with mean mu and standard
	deviation sigma. This function does the samething as 
	sts.lognorm.pdf(x, s = sig, scale = np.exp(mu)).
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, values of the log-normally distributed random
			 variable
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
	
	OBJECTS CREATED WITHIN FUNCTION:
	pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
			   corresponding to xvals data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: pdf_vals
	--------------------------------------------------------------------
	'''
	pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))    
	return pdf_vals


def log_lik(pdf_vals):
	'''
	--------------------------------------------------------------------
	Compute the log likelihood function given function that generates
	values of a lognormal pdf.
	--------------------------------------------------------------------
	INPUTS:
	pdf_vals  = values of a lognormal pdf
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		lognorm_pdf()
	
	OBJECTS CREATED WITHIN FUNCTION:
	log_lik_val = scalar, value of the log likelihood function
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: log_lik_val
	--------------------------------------------------------------------
	'''
	ln_pdf_vals = np.log(pdf_vals)
	log_lik_val = ln_pdf_vals.sum()
	return log_lik_val


def problem1b(data, mu, sig, dist_pts):
	'''
	--------------------------------------------------------------------
	A function that plots the lognormal PDF f(x|mu, sigma) of the given data.
	--------------------------------------------------------------------

	INPUTS:
		data     = income data
		mu       = scalar, mean of the normally distributed random variable
		sigma    = scalar > 0, standard deviation of the normally distributed
				   random variable
		dist_pts = given lower bounds, higher bounds, and number of points
				   generates a set of points.

	RETURNS:
		None.
	--------------------------------------------------------------------
	'''
	pdf_vals = sts.lognorm.pdf(x = dist_pts, s = sig, scale = np.exp(mu))
	# Same outcome can be attained by using the following function.
	# pdf_vals = lognorm_pdf(dist_pts, mu, sig)
	
	pdf_vals1 = sts.lognorm.pdf(x = data, s = sig, scale = np.exp(mu))
	# Same outcome can be attained by using the following function.
	# pdf_vals1 = lognorm_pdf(data, mu, sig)
	

	graph = True
	if graph:
		plt.plot(dist_pts, pdf_vals, linewidth=2, color='r',
			 label=r'$\mu$ = 9, $\sigma$ = 0.3')
		plt.legend(loc='upper right')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1b')
		plt.savefig(output_path)
		plt.close()

	return log_lik(pdf_vals1)


### Problem 1c:
def crit(params, args):
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
	args   = xvals
	xvals  = (N,) vector, values of the normally distributed random
			 variable
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		log_lik()
		lognorm_pdf()
	
	OBJECTS CREATED WITHIN FUNCTION:
	log_lik_val = scalar, value of the log likelihood function
	neg_log_lik_val = scalar, negative of log_lik_val
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: neg_log_lik_val
	--------------------------------------------------------------------
	'''
	mu, sigma = params
	xvals = args
	log_lik_val = log_lik(lognorm_pdf(xvals, mu, sigma))
	neg_log_lik_val = - log_lik_val

	return neg_log_lik_val


def mle_dist(mu_init, sig_init, data):
	'''
	--------------------------------------------------------------------
	This function returns the optimization results for problem 1.
	--------------------------------------------------------------------

	INPUTS:
		data     = income data
		mu_init  = scalar, mean of the normally distributed random variable
		sig_init = scalar > 0, standard deviation of the normally distributed
				   random variable
		
	RETURNS:
		None.
	--------------------------------------------------------------------
	'''
	params_init = np.array([mu_init, sig_init])
	mle_args = data
	bounds = ((None, None), (0.000000001, None))
	results = opt.minimize(crit, params_init, mle_args,
						   method = 'TNC',
						   bounds = bounds)
	# TNC methed doesn't return an inverse Hessian Matrix so I continue
	# to use L-BFGS-B method.
	mu_best, sig_best = results.x
	params_best = np.array([mu_best, sig_best])
	results = opt.minimize(crit, params_best, mle_args,
						   method = 'L-BFGS-B',
						   bounds = bounds)
	return results


def plot_pdf(mu, sig, data, dist_pts, num_bins):
	'''
	--------------------------------------------------------------------
	A function that plots the lognormal distribution by maximum likelihood,
	by initial parameters, and histogram.
	--------------------------------------------------------------------

	INPUTS:
		data     = income data
		mu       = scalar, mean of the normally distributed random variable
		sigma    = scalar > 0, standard deviation of the normally distributed
				   random variable
		dist_pts = given lower bounds, higher bounds, and number of points
				   generates a set of points.
		num_bins = number of bins along x-axis

	RETURNS:
		None.
	--------------------------------------------------------------------
	'''
	graph = True
	if graph:
		count, bins, ignored = plt.hist(data, num_bins, normed = True)
		plt.title('MACSS student income: 2018-2020', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('Frequency')

		pdf_vals_b = sts.lognorm.pdf(x = dist_pts, s = sig, scale = np.exp(mu))

		mu_MLE, sig_MLE = mle_dist(mu, sig, data).x
		pdf_vals_c = sts.lognorm.pdf(x = dist_pts, s = sig_MLE, 
									 scale = np.exp(mu_MLE))

		plt.plot(dist_pts, pdf_vals_b, linewidth=2, color='r',
			 label=r'$\mu$ = 9, $\sigma$ = 0.3')
		plt.plot(dist_pts, pdf_vals_c, linewidth=2, color='g',
			 label='$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu_MLE, sig_MLE))

		plt.legend(loc='upper right')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1c')
		plt.savefig(output_path)
		plt.close()


### Problem1d:
def problem1d(log_lik_mle, log_lik_h0):
	'''
	--------------------------------------------------------------------
	A function that does likelihood ratio test.
	--------------------------------------------------------------------

	INPUTS:
		log_like_mle = MLE estimate
		log_like_h0  = likelihood given by hypothesized values of parameters

	RETURNS:
		p value.
	--------------------------------------------------------------------
	'''
	LR_val = 2 * (log_lik_mle - log_lik_h0)
	pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
	return pval_h0


### Problem2a:
def norm_lik(df_sick, params):
	'''
	--------------------------------------------------------------------
	Compute the log likelihood function given function of sickdata from
	problem prompt.
	--------------------------------------------------------------------
	INPUTS:
	df_sick  = sick data in the form of datafrom
	params   = parameters, sigma, beta0, beta1, beta2, beta3
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		sts.norm()
	
	OBJECTS CREATED WITHIN FUNCTION:
	log_lik_val = scalar, value of the log likelihood function
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: log_lik_val
	--------------------------------------------------------------------
	'''
	sigma, beta_0, beta_1, beta_2, beta_3 = params
	errorterm = df_sick['sick'] - beta_0 - beta_1 * df_sick['age']\
				- beta_2 * df_sick['children']\
				- beta_3 * df_sick['avgtemp_winter']
	norm_dist = sts.norm(scale = sigma)
	norm_pdf = norm_dist.pdf(errorterm)
	ln_pdf_vals = np.log(norm_pdf)
	log_lik_val = ln_pdf_vals.sum()
	return log_lik_val


def crit2(params, args):
	'''
	--------------------------------------------------------------------
	This function computes the negative of the log likelihood function
	given parameters and data. This is the minimization problem version
	of the maximum likelihood optimization problem
	--------------------------------------------------------------------
	INPUTS:
	params = (5,) vector, ([sigma, beta_0, beta_1, beta_2, beta_3])
	args   = xvals
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		norm_lik()
	
	OBJECTS CREATED WITHIN FUNCTION:
	log_lik_val = scalar, value of the log likelihood function
	neg_log_lik_val = scalar, negative of log_lik_val
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: neg_log_lik_val
	--------------------------------------------------------------------
	'''
	sigma, beta_0, beta_1, beta_2, beta_3 = params
	xvals = args
	log_lik_val = norm_lik(xvals, params)
	neg_log_lik_val = - log_lik_val
	return neg_log_lik_val


def est_params2(params, data):
	'''
	--------------------------------------------------------------------
	This function returns the optimization results for problem 2.
	--------------------------------------------------------------------

	INPUTS:
		data     = sick data
		params   = parameters (sigma, beta0, beta1, beta2, beta3)
		
	RETURNS:
		results
	--------------------------------------------------------------------
	'''
	params_init = np.array(list(params))
	mle_args = data
	bnds = ((1e-10, None), (None, None), (None, None), (None, None), (None, None))
	results = opt.minimize(crit2, params_init, args = mle_args,
						   bounds = bnds, method = 'SLSQP')
	return results


def est2_hessinv(params, data):
	'''
	--------------------------------------------------------------------
	This function returns the inverse Hessian matrix for problem 2,
	since only 'L-BFGS-B' could return this outcome.
	--------------------------------------------------------------------

	INPUTS:
		data     = sick data
		params   = parameters (sigma, beta0, beta1, beta2, beta3)
		
	RETURNS:
		results.hess_inv
	--------------------------------------------------------------------
	'''
	mle_args = data
	bnds = ((1e-10, None), (None, None), (None, None), (None, None), (None, None))
	results = opt.minimize(crit2, params, args = mle_args,
						   bounds = bnds, method = 'L-BFGS-B')
	return results.hess_inv


### Problem 2b:
def problem2b(log_lik_mle, log_lik_h0):
	'''
	--------------------------------------------------------------------
	A function that does likelihood ratio test.
	--------------------------------------------------------------------

	INPUTS:
		log_like_mle = MLE estimate
		log_like_h0  = likelihood given by hypothesized values of parameters

	RETURNS:
		p value.
	--------------------------------------------------------------------
	'''
	LR_val = 2 * (log_lik_mle - log_lik_h0)
	pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
	return pval_h0


if __name__ == "__main__":
	np.seterr(all='ignore')
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)

	data = np.loadtxt("incomes.txt")
	num_bins = 30
	problem1a(data, num_bins)

	mu = 9.0
	sigma = 0.3
	lowerbnd = 0
	upperbnd = 150000
	num_pts = 10000
	dist_pts = np.linspace(lowerbnd, upperbnd, num_pts)
	print()
	log_lik_h0 = problem1b(data, mu, sigma, dist_pts)
	print("1b. The value of the log likelihood value is : {:.2f}".
		  format(log_lik_h0))

	plot_pdf(mu, sigma, data, dist_pts, num_bins)
	results = mle_dist(mu, sigma, data)
	mu_MLE, sig_MLE = results.x
	log_lik_mle = - results.fun
	vcv_mle = results.hess_inv

	print()
	print('1. The MLE for problem1 gives us this message:', results.success,
		  ',which means we got an optimum outcome.')
	print()
	print("1c. The ML estimates for mu is: {:.3f}, for sigma is: {:.3f}."
		  .format(mu_MLE, sig_MLE))
	print()
	print("1c. The maximum value of likelihood function is: {:.2f}."
		   .format(log_lik_mle))
	print()
	print('1c. The variance covariance matrix of the estimates is: {:}.'
		  .format(vcv_mle.todense()))

	print()
	pval = problem1d(log_lik_mle, log_lik_h0)
	print("1d. The Chi square of H0 with 2 degrees of freedom p-value = {:.8f}."
		  .format(pval), "Since this probability is small, the data is unlikely",
		  "coming from the distributioon in part(b)")

	finaldist = sts.lognorm(s = sig_MLE, scale = np.exp(mu_MLE))
	prob_highinc = (1 - finaldist.cdf(100000)) * 100
	prob_lowinc = finaldist.cdf(75000) * 100
	print()
	print("1e. The probability that a student would earn more than $100,000 is: {:.2f}%."
		  .format(prob_highinc), "The probability of a student earn less than",
		  "$75,000 is: {:.2f}%".format(prob_lowinc))

	sickdata = pd.read_csv("sick.txt", encoding = 'utf-8-sig')
	results2 = est_params2((1,0,0,0,0), sickdata)
	sigma, beta_0, beta_1, beta_2, beta_3 = results2.x
	params = np.array(list(results2.x))
	value = - results2.fun
	hess = est2_hessinv(params, sickdata)
	print()
	print('2. The MLE for problem2 gives us this message:', results2.success,
		  ',which means we got an optimum outcome.')
	print()
	print('2a. The estimates for beta_0, beta_1, beta_2, beta_3, and sigma^2 are:'
		  '{:.3f}, {:.3f}, {:.3f}, {:.6f}, and {:.8f}.\n'
		  .format(beta_0, beta_1, beta_2, beta_3, sigma**2))
	print('2a. The value of the log likelihood function is: {:.2f}\n'.format(
			value ))
	print('2a. The estimated variance covariance matrix of the estimates is: {}\n'
		   .format(hess.todense()))

	log_lik_mle = value
	log_lik_h0 = norm_lik(sickdata, (np.sqrt(.01), 1, 0, 0, 0))
	p_val2 = problem2b(log_lik_mle, log_lik_h0)
	print()
	print('Likelihood Ratio Test p-value is: {:.8f}\n'
		 .format(p_val2))
	print('This number is really low (< .05), so it is unlikely that age, '
		  'number of children, and average winter temperature have effect on'
		  'the number of sick days.\n')












