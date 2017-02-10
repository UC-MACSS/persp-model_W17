# Perspectives on Computational Modeling
# Problem Set 4
# Name: Ningyin Xu

import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import os


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
		fig = plt.figure(figsize = (10,10))
		# weights = (1 / data.shape[0]) * np.ones_like(data)
		count, bins, ignored = plt.hist(data, num_bins, normed = True, 
										color = 'powderblue')
		plt.title('MACSS student income: 2018-2020', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF Value of Incomes')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1a')
		plt.savefig(output_path)
		plt.close()


### Problem 1b:
def LN_pdf(xvals, mu, sigma):
	'''
	--------------------------------------------------------------------
	Generate pdf values from the lognormal pdf with mean mu and standard
	deviation sigma. This function does the samething as 
	sts.lognorm.pdf(x, s = sig, scale = np.exp(mu)).
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, values of the log-normally distributed random
			 variable or (N, S) matrix of simulated data
	mu     = scalar, mean of the lognormally distributed random variable
	sigma  = scalar > 0, standard deviation of the lognormally distributed
			 random variable
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
	np.apply_along_axis
	
	OBJECTS CREATED WITHIN FUNCTION:
	pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
			   corresponding to xvals data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: pdf_vals
	--------------------------------------------------------------------
	'''
	if xvals.ndim == 1:
		pdf_vals  = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
	elif xvals.ndim == 2:
		pdf_val = lambda x : (1/(x * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2))
		pdf_vals = np.apply_along_axis(pdf_val, axis = 1, arr = xvals)

	return pdf_vals


### Problem 1c:
def lognorm_draws(N, S, seed, mu, sigma):
	'''
	--------------------------------------------------------------------
	Draw (N x S) matrix of random draws from a log normal
	distribution with mean mu and standard deviation sigma. These draws
	correspond to an (N x S) matrix of randomly generated draws from a
	normal distribution U(0,1).
	--------------------------------------------------------------------
	INPUTS:
	N, S   = number of rows and columns of the matrix
	seed   = seed for generating random number according to certain order
	mu     = scalar, mean of the lognormally distributed random variable
	sigma  = scalar > 0, standard deviation of the lognormally distributed
			 random variable

	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		np.random.normal()
	
	OBJECTS CREATED WITHIN FUNCTION:
	draws
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: draws
	--------------------------------------------------------------------
	'''
	np.random.seed(seed)
	norm_dist = rnd.normal(mu,sigma, size = (N, S))
	draws = np.exp(norm_dist)
	
	return draws


def data_moments(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the two data moments for SMM
	(mean(data), std(data)) from both the actual data and from the
	simulated data.
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
			income data, either real world or simulated. Real world
			data will come in the form (N,). Simulated data comes in the
			form (N,) or (N, S).
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
	
	OBJECTS CREATED WITHIN FUNCTION:
	mean_data = scalar or (S,) vector, mean value of income data
	std_data  = scalar > 0 or (S,) vector, std of income data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: mean_data, std_data
	--------------------------------------------------------------------
	'''
	if xvals.ndim == 1:
		mean_data = xvals.mean()
		std_data = xvals.std()
	elif xvals.ndim == 2:
		mean_data = xvals.mean(axis=0)
		std_data = xvals.std(axis=0)
	
	return mean_data, std_data


def err_vec(data_vals, sim_vals, mu, sigma, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for SMM.
	--------------------------------------------------------------------
	INPUTS:
	data_vals = (N,) vector, income data
	sim_vals  = (N, S) matrix, S simulations of income data
	mu        = scalar, mean of the nontruncated normal distribution
				from which the truncated normal is derived
	sigma     = scalar > 0, standard deviation of the nontruncated
				normal distribution from which the truncated normal is
				derived
	simple    = boolean, =True if errors are simple difference, =False
				if errors are percent deviation from data moments
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		data_moments()
		
	OBJECTS CREATED WITHIN FUNCTION:
	mean_data  = scalar, mean value of data
	std_data   = scalar > 0, standard deviation of data
	moms_data  = (2, 1) matrix, column vector of two data moments
	mean_model = scalar, estimated mean value from model
	std_model  = scalar > 0, estimated standard deviation from model
	moms_model = (2, 1) matrix, column vector of two model moments
	err_vec    = (2, 1) matrix, column vector of two moment error
				 functions
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
	mean_data, std_data = data_moments(data_vals)
	moms_data = np.array([[mean_data], [std_data]])
	mean_sim, std_sim = data_moments(sim_vals)
	mean_model = mean_sim.mean()
	std_model = std_sim.mean()
	moms_model = np.array([[mean_model], [std_model]])
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = (moms_model - moms_data) / moms_data

	return err_vec


def criterion(params, *args):
	'''
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
				(xvals, W_hat, N, S, seed)
	xvals     = (N,) vector, values of the lognormally
				distributed random variable
	draws     = (N, S) matrix
	W_hat     = (R, R) matrix, estimate of optimal weighting matrix
	N, S   	  = number of rows and columns of the matrix
	seed   	  = seed for generating random number according to certain order
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		norm_pdf()
	
	OBJECTS CREATED WITHIN FUNCTION:
	err        = (2, 1) matrix, column vector of two moment error
				 functions
	crit_val   = scalar > 0, SMM criterion function value
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: crit_val
	--------------------------------------------------------------------
	'''
	mu, sigma = params
	xvals, W_hat, N, S, seed = args
	sim_vals = lognorm_draws(N, S, seed, mu, sigma)
	err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W_hat), err) 
	
	return crit_val


def problem1c(pts, N, S, seed, mu, sigma):
	'''
	--------------------------------------------------------------------
	Main function for problem 1c, including executing optimaztion.minimize
	function (SMM part), printng all the expected outcome, and plotting.
	--------------------------------------------------------------------
	INPUTS:
	pts       = (N,) vector, income data
	mu        = scalar, initial mean
	sigma     = scalar > 0, initial standard deviation
	N, S   	  = number of rows and columns of the matrix
	seed   	  = seed for generating random number according to certain order
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		opt.minimize;
		lognorm_draws;
		data_moments
	
	OBJECTS CREATED WITHIN FUNCTION:
		mu_SMM1
		sig_SMM1
		sim_vals
	
	FILES CREATED BY THIS FUNCTION: 
		Fig_1c.png
	
	RETURNS: mu_SMM1, sig_SMM1, sim_vals
	--------------------------------------------------------------------
	'''
	params_init = np.array([mu, sigma])
	W_hat = np.eye(2)
	args = (pts, W_hat, N, S, seed)
	results = opt.minimize(criterion, params_init, args = (args),
						   method = 'L-BFGS-B',
						   bounds = ((None, None), (1e-10, None)))
	mu_SMM1, sig_SMM1 = results.x
	params_SMM = np.array([mu_SMM1, sig_SMM1])
	value = criterion(params_SMM, *args)[0][0]
	sim_vals = lognorm_draws(N, S, seed, mu_SMM1, sig_SMM1)
	mean_data, std_data = data_moments(pts)
	mean_sim, std_sim = data_moments(sim_vals)
	mean_model = mean_sim.mean()
	std_model = std_sim.mean()
	print()
	print('1c.')
	print('mu_SMM1={:.4f}'.format(mu_SMM1), 'sig_SMM1={:.4f}'.format(sig_SMM1))
	print('The value of SMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: mu: {:.4f}, std: {:.4f}'.format(mean_data, std_data))
	print('Model moments are: mu: {:.4f}, std: {:.4f}'.format(mean_model, std_model))

	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
		 linewidth=2, color='k', label=
		 '$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_SMM1, sig_SMM1))
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1c')
		plt.savefig(output_path)
		plt.close()

	return (mu_SMM1, sig_SMM1, sim_vals)


### Problem 1d:
def problem1d(pts, mu_SMM1, sig_SMM1, sim_vals, mu_init, std_init,
			  N, S, seed):
	'''
	--------------------------------------------------------------------
	Main function for problem 1d, including executing optimaztion.minimize
	function (SMM part), printng all the expected outcome, and plotting.
	--------------------------------------------------------------------
	INPUTS:
	pts       = (N,) vector, income data
	mu_init   = scalar, initial mean
	std_init  = scalar > 0, initial std
	mu_SMM1   = scalar, mean of the 1c SMM estimation
	mig_SMM1  = scalar, standard deviation of the 1c SMM estimation
	N, S   	  = number of rows and columns of the matrix
	seed   	  = seed for generating random number according to certain order
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		err_vec
		opt.minimize;
		lognorm_draws;
		data_moments
	
	OBJECTS CREATED WITHIN FUNCTION:
		mu_SMM2
		sig_SMM2
	
	FILES CREATED BY THIS FUNCTION: 
		Fig_1d.png
	
	RETURNS: mu_SMM2, sig_SMM2
	--------------------------------------------------------------------
	'''
	err1 = err_vec(pts, sim_vals, mu_SMM1, sig_SMM1, False)
	VCV2 = np.dot(err1, err1.T) / pts.shape[0]
	W_hat2 = lin.pinv(VCV2)
	params_init = np.array([mu_init, std_init])
	args = (pts, W_hat2, N, S, seed)
	results = opt.minimize(criterion, params_init, args=(args),
						   method='TNC', bounds=((None, None), (1e-10, None)))
	mu_SMM2, sig_SMM2 = results.x
	params_SMM = np.array([mu_SMM2, sig_SMM2])
	value = criterion(params_SMM, *args)[0][0]
	sim_vals = lognorm_draws(N, S, seed, mu_SMM2, sig_SMM2)
	mean_data, std_data = data_moments(pts)
	mean_sim, std_sim = data_moments(sim_vals)
	mean_model = mean_sim.mean()
	std_model = std_sim.mean()
	print()
	print('1d.')
	print('mu_SMM2={:.4f}'.format(mu_SMM2), 'sig_SMM2={:.4f}'.format(sig_SMM2))
	print('The value of SMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: mu: {:.4f}, std: {:.4f}'.format(mean_data, std_data))
	print('Model moments are: mu: {:.4f}, std: {:.4f}'.format(mean_model, std_model))

	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF Values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1, sig_SMM1),
		 linewidth=2, color='k', label=
		 '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_SMM1, sig_SMM1))
		plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2, sig_SMM2),
		 linewidth=2, color='red', label=
		 '2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_SMM2, sig_SMM2))
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1d')
		plt.savefig(output_path)
		plt.close()

	return (mu_SMM2, sig_SMM2)

if __name__ == "__main__":
	np.seterr(all='ignore')
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)


	incomepts = np.loadtxt('incomes.txt')
	num_bins = 30
	problem1a(incomepts, num_bins)

	testxvals = np.array([[200.0, 270.0], [180.0, 195.5]])
	testmu = 5.0
	testsig = 1.0
	testoc = LN_pdf(testxvals, testmu, testsig)
	if testoc.shape == testxvals.shape:
		print()
		print('1b.')
		print(testoc)
		print('The lnpdf function is constructed successfully.')

	S = 300
	N = 200
	seed = 1234
	mu_init = 9
	sigma_init = 0.3
	
	mu_SMM1, sig_SMM1, sim_vals = problem1c(incomepts, N, S, seed, mu_init, sigma_init)

	mu_init = 11
	sigma_init = 0.2
	mu_SMM2, sig_SMM2 = problem1d(incomepts, mu_SMM1, sig_SMM1, sim_vals,
								  mu_init, sigma_init, N, S, seed)



