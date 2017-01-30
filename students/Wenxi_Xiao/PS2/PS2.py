import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
import pandas as pd


def f1a():
	'''
	Problem 1a
	--------------------------------------------------------------------
	This function plots a histogram of percentages of the incomes.txt data
	with 30 bins. The bins are weighted such that the height of each bin
	represents the percent of the income observations in that bin.
	--------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	pts = np.loadtxt('incomes.txt') #load txt file
	# plot frequency histogram: 
	fig, ax = plt.subplots(figsize = (8,6))
	weights = (1.0 / len(pts)) * np.ones_like(pts)
	h = plt.hist(pts, bins = 30, weights = weights)
	#count, bins, ignored = plt.hist(pts, 30, normed=True)
	plt.title('Histogram of percentages of Incomes', fontsize=20)
	plt.xlabel('Total points')
	plt.ylabel('Percent of incomes')
	#plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted


def norm_pdf(xvals, mu, sigma, cutoff):
	'''
	--------------------------------------------------------------------
	This function generates pdf values from the lognormal pdf with mean mu 
	and standard deviation sigma. If the cutoff is given, then the PDF 
	values are inflated upward to reflect the zero probability on values
	above the cutoff. If there is no cutoff given, this function does 
	the same thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
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
			
	pdf_vals   = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut)
	return pdf_vals


def f1b():
	'''
	Problem 1b
	--------------------------------------------------------------------
	This function plots a lognormal PDF.
	--------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	pts = np.loadtxt('incomes.txt') #load txt file
	dist_pts = np.linspace(0, 150000, 200) # 0-150000 is range, 200 is the number of the total data points
	mu = 9.0
	sig = 0.3
	plt.plot(dist_pts, norm_pdf(dist_pts, mu, sig, 'None'), linewidth=2, color='r', label='1b: $\mu$=9.0,$\sigma$=0.3')
	plt.title('1b. Lognormal PDF', fontsize=20)
	plt.xlabel('Incomes')
	plt.ylabel('Percent of students')
	plt.legend(loc='upper right')


def log_lik_norm(xvals, mu, sigma, cutoff):
	'''
	--------------------------------------------------------------------
	This function computes the log likelihood function for data xvals given normal
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


pts = np.loadtxt('incomes.txt') #load txt file
mu = 9.0
sig = 0.3
print('Log-likelihood 1b: ', log_lik_norm(pts, mu, sig, 'None'))
# print('This value is really low, which suggests that the data does not fit the model. This is a bad model.')


def crit(params, *args):
	'''
	--------------------------------------------------------------------
	This function computes the negative of the log likelihood function
	given parameters and data. This is the minimization problem version
	of the maximum likelihood optimization problem.
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


def f1c():
	'''
	Problem 1c:
	--------------------------------------------------------------------
	This function computes the estimated PDF by maximum likeli-
	hood, and plots its against the PDF from part (b) and the histogram 
	from part (a).
	--------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	pts = np.loadtxt('incomes.txt') #load txt file
	mu_init = 9.0  
	sig_init = 0.3 
	params_init = np.array([mu_init, sig_init])
	mle_args = (pts, 150000.0)
	results = opt.minimize(crit, params_init, args=(mle_args), bounds=((None, None), (1e-10, None)), method='TNC')
	mu_MLE, sig_MLE = results.x
	params_init2 = np.array([mu_MLE, sig_MLE]) 
	results2 = opt.minimize(crit, params_init2, args=(mle_args), bounds=((None, None), (1e-10, None)), method='L-BFGS-B')
	mu_MLE2, sig_MLE2= results2.x

	# Plot the hostogram of the data (part a):
	count, bins, ignored = plt.hist(pts, 30, normed=True)
	plt.title('Incomes of MACSS Graduates of 2018-2020', fontsize=20)
	plt.xlabel('Incomes')
	plt.ylabel('Percent of students')
	plt.xlim([0, 150000])  # This gives the xmin and xmax to be plotted

	# Plot the the lognormal PDF from part b):
	pts = np.loadtxt('incomes.txt')
	dist_pts = np.linspace(0, 150000, 200) # 0-150000 is range, 200 is the number of the total data points
	plt.plot(dist_pts, norm_pdf(dist_pts, mu_init, sig_init, 'None'), linewidth=2, color='r', label='1b: $\mu$=9.0,$\sigma$=0.3')
	plt.legend(loc='upper right')

	# Plot the MLE estimated distribution
	plt.plot(dist_pts, norm_pdf(dist_pts, mu_MLE2, sig_MLE2, 'None'), linewidth=2, color='k', label='1c: $\mu$=11.33,$\sigma$=0.21')
	plt.legend(loc='upper right')

	print('1c. mu_MLE=', mu_MLE2, '1c. sig_MLE=', sig_MLE2)
	print('Log-likelihood 1c: ', log_lik_norm(pts, mu_MLE2, sig_MLE2, 'None'))
	# Compute the hessian matrix:
	vcv_mle = results2.hess_inv
	print('Variance-covariance matrix = ', vcv_mle.todense())


	'''
	Problem 1d:
	--------------------------------------------------------------------
	The following code segment performs a likelihood ratio test (Chi-square test)
	to determine the probability that the data in incomes.txt came from 
	the distribution in part (b).
	--------------------------------------------------------------------
	'''
	log_lik_1b = log_lik_norm(pts, 9.0, 0.3, 'None')
	log_lik_mle = log_lik_norm(pts, mu_MLE, sig_MLE, 'None')
	LR_val = 2 * (log_lik_mle - log_lik_1b)
	chi2_1b = sts.chi2.pdf(LR_val, 2)
	print('1d. The probability that the data in incomes.txt came from the distribution in part (b) is', chi2_1b)


	'''
	Problem 1e:
	--------------------------------------------------------------------
	The following code segment uses the estimated model from part (c) to 
	calculate 1) the probability that I will earn more than $100,000, 2)
	the probability that I will earn less than $75,000.
	--------------------------------------------------------------------
	'''
	dist_1c = sts.lognorm(s = sig_MLE2, loc = mu_MLE2, scale = np.exp(mu_MLE2)) 
	p_more_100000 = 1 - dist_1c.cdf (100000)
	p_less_75000 = dist_1c.cdf (75000) 
	print('1e. The probability that I will earn more than 100,000 is ', p_more_100000)
	print('1e. The probability that I will earn less than 75,000 is ', p_less_75000)


#--------------------------------------------------------------------
#Problem 2
#--------------------------------------------------------------------
mydf = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter']) #load txt file as a pd df
#print(df)


def lik_linear_regression(mydf, b0, b1, b2, b3, sigma, cutoff):
	'''
	--------------------------------------------------------------------
	This function computes the log likelihood function for data xvals 
	given normal distribution parameters b0, b1, b2, b3, and sigma.
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
	myerror = mydf['sick'] - b0 - b1 * mydf['age'] - b2 * mydf['children'] - b3 * mydf['avgtemp_winter']
	norm_dist = sts.norm(scale = sigma) # forcing betas to normally distributed with sigma = sigma
	norm_pdf = norm_dist.pdf(myerror)
	ln_pdf_vals = np.log(norm_pdf)
	log_lik_val = ln_pdf_vals.sum()
	return log_lik_val


def crit_problem2(params, args):
	'''
	--------------------------------------------------------------------
	This function computes the negative of the log likelihood function
	given parameters and data. This is the minimization problem version
	of the maximum likelihood optimization problem.
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
	b0, b1, b2, b3, sigma = params
	mydf = args
	log_lik_val = lik_linear_regression(mydf, b0, b1, b2, b3, sigma, 'None')
	neg_log_lik_val = -log_lik_val
	
	return neg_log_lik_val


def f2a():
	'''
	------------------------------------------------------------------------
	Problem 2a: estimate b0, b1, b2, b3, sigma^2 to maximize the likelihood
	of seeing the data in sick.txt. Report your estimates, the value of the 
	log likelihood function, and the estimated variance covariance matrix of
	the estimates.
	------------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	df = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter'])
	b0_init = 0.0  
	b1_init = 0.0
	b2_init = 0.0  
	b3_init = 0.0
	sig2_init = 1.0
	params_init = np.array([b0_init, b1_init, b2_init, b3_init, sig2_init])
	mle_args = df
	results = opt.minimize(crit_problem2, params_init, args=mle_args, bounds=((None, None), (None, None), (None, None), (None, None), (1e-10, None)), method='SLSQP')
	b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig2_MLE = results.x
	params_init2 = np.array([b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig2_MLE]) 
	results2 = opt.minimize(crit_problem2, params_init2, args=mle_args, bounds=((None, None), (None, None), (None, None), (None, None), (1e-10, None)), method='L-BFGS-B')
	b0_MLE2, b1_MLE2, b2_MLE2, b3_MLE2, sig2_MLE2= results2.x
	print('2a. b0: ', b0_MLE)
	print('2a. b1: ', b1_MLE)
	print('2a. b2: ', b2_MLE)
	print('2a. b3: ', b3_MLE)
	print('2a. sigma^2: ', sig2_MLE * sig2_MLE)
	print('Log-likelihood 2a: ', lik_linear_regression(df, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig2_MLE, 'None'))
	# Compute the hessian matrix:
	vcv_mle2 = results2.hess_inv
	print('2a. Variance-covariance matrix = ', vcv_mle2.todense())


	'''
	Problem 2b:
	--------------------------------------------------------------------
	The following code segment performs a likelihood ratio test (Chi-square test)
	to determine the probability that sigma^2 = 0.01 and b1, b2, b3 = 0.
	--------------------------------------------------------------------
	'''
	log_lik_2b = lik_linear_regression(mydf, 0, 0, 0, 0, 0.1, 'None')
	log_lik_mle = lik_linear_regression(mydf, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig2_MLE, 'None')
	LR_val = 2 * (log_lik_mle - log_lik_2b)
	chi2_2b = sts.chi2.pdf(LR_val, 5)
	print('2b. The likelihood that age, number of children, and average winter temperature have no effect on the number of sick days is', chi2_2b)



def save_figure(filename):
	'''
	--------------------------------------------------------------------
	This function saves figures into the images/ directory. 
	--------------------------------------------------------------------
	INPUT: a string represents the filename of a figure to be saved
	RETURN: none
	--------------------------------------------------------------------
	'''
	fn = os.path.split(filename)[1]
	if not os.path.exists('images'):
		os.mkdir('images')
	fn = os.path.join('images', fn)
	plt.savefig(fn)
	print(fn + ' saved.')


#--------------------------------------------------------------------
#The followig code segment saves the plots produced in problem 1a, 1b, 
#1c in the images folder.
#--------------------------------------------------------------------
if '__main__'==__name__:
	# plt.ion()
	f1a(); save_figure('Fig_1a');  plt.close()
	f1b(); save_figure('Fig_1b');  plt.close()
	f1c(); save_figure('Fig_1c');  plt.close()
	f2a();
	pass