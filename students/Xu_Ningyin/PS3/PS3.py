# Perspectives on Computational Modeling
# Problem Set 3
# Name: Ningyin Xu

import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd


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


def data_moments(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the two data moments for GMM
	(mean(data), std(data)).
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N,) vector
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
	
	OBJECTS CREATED WITHIN FUNCTION:
	mean_data = scalar, mean value of data
	std_data  = scalar > 0, standard deviation of data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: mean_data, std_data
	--------------------------------------------------------------------
	'''
	mean_data = xvals.mean()
	std_data = xvals.std()    
	return mean_data, std_data


def model_moments(mu, sigma):
	'''
	--------------------------------------------------------------------
	This function computes the two model moments for GMM
	(mean(model data), standard-deviation(model data)).
	--------------------------------------------------------------------
	INPUTS:
	mu     = scalar, mean of the log-normally distributed random variable
	sigma  = scalar > 0, standard deviation of the log-normally distributed
			 random variable
	
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		lognorm_pdf()
		xfx()
		x2fx()
	
	OBJECTS CREATED WITHIN FUNCTION:
	mean_model = scalar, mean value of data from model
	m_m_err    = scalar > 0, estimated error in the computation of the
				 integral for the mean of the distribution
	var_model  = scalar > 0, variance of data from model
	v_m_err    = scalar > 0, estimated error in the computation of the
				 integral for the variance of the distribution
	std_model  = scalar, standard deviation of data from model
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: mean_model, std_model
	--------------------------------------------------------------------
	'''
	xfx = lambda x: x * lognorm_pdf(x, mu, sigma)
	(mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
	x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma) 
	(var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
	std_model = np.sqrt(var_model)
	return mean_model, std_model


def err_vec(xvals, mu, sigma, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, data
	mu     = scalar, mean of the lognormally distributed random variable
	sigma  = scalar > 0, standard deviation of the lognormally distributed
			 random variable
	simple = boolean, =True if errors are simple difference, =False if
			 errors are percent deviation from data moments
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		data_moments()
		model_moments()
	
	OBJECTS CREATED WITHIN FUNCTION:
	mean_data  = scalar, mean value of data
	std_data   = scalar > 0, standard deviation of data
	moms_data  = (2, 1) matrix, column vector of two data moments
	mean_model = scalar, mean value from model
	std_model  = scalar > 0, standard deviation from model
	moms_model = (2, 1) matrix, column vector of two model moments
	err_vec    = (2, 1) matrix, column vector of two moment error
				 functions
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
	mean_data, std_data = data_moments(xvals)
	moms_data = np.array([[mean_data], [std_data]])
	mean_model, std_model = model_moments(mu, sigma)
	moms_model = np.array([[mean_model], [std_model]])
	if simple:
		err_vec = moms_model - moms_data # in case moms_data is too close to zero
	else:
		err_vec = (moms_model - moms_data) / moms_data
	
	return err_vec


def criterion(params, *args):
	'''
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
	args   = length 2 tuple, (xvals, W_hat)
	xvals  = (N,) vector, values of the truncated normally distributed
			 random variable
	W_hat  = (R, R) matrix, estimate of optimal weighting matrix
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		lognorm_pdf()
	
	OBJECTS CREATED WITHIN FUNCTION:
	err        = (2, 1) matrix, column vector of two moment error
				 functions
	crit_val   = scalar > 0, GMM criterion function value
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: crit_val
	--------------------------------------------------------------------
	'''
	mu, sigma = params
	xvals, W = args
	err = err_vec(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)     
	return crit_val


def problem1b(pts, mu_init, std_init):
	params_init = np.array([mu_init, std_init])
	W_hat = np.eye(2)
	gmm_args = (pts, W_hat)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM1, sig_GMM1 = results.x
	params_GMM = np.array([mu_GMM1, sig_GMM1])
	value = criterion(params_GMM, *gmm_args)[0][0]
	mu_data, std_data = data_moments(pts)
	mu_model, std_model = model_moments(mu_GMM1, sig_GMM1)
	print()
	print('1b.')
	print('mu_GMM1={:.4f}'.format(mu_GMM1), 'sig_GMM1={:.4f}'.format(sig_GMM1))
	print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: mu: {:.4f}, std: {:.4f}'.format(mu_data, std_data))
	print('Model moments are: mu: {:.4f}, std: {:.4f}'.format(mu_model, std_model))
	
	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1),
		 linewidth=2, color='k', label=
		 '$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM1, sig_GMM1))
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1b')
		plt.savefig(output_path)
		plt.close()

	return (mu_GMM1, sig_GMM1)

### Problem 1c:
def problem1c(pts, mu_GMM1, sig_GMM1, mu_init, std_init):
	err1 = err_vec(pts, mu_GMM1, sig_GMM1, False)
	VCV2 = np.dot(err1, err1.T) / pts.shape[0]
	W_hat2 = lin.pinv(VCV2)
	params_init = np.array([mu_init, std_init])
	gmm_args = (pts, W_hat2)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM2, sig_GMM2 = results.x
	params_GMM = np.array([mu_GMM2, sig_GMM2])
	value = criterion(params_GMM, *gmm_args)[0][0]
	mu_data, std_data = data_moments(pts)
	mu_model2, std_model2 = model_moments(mu_GMM2, sig_GMM2)
	print()
	# print(results)
	print('1c.')
	print('mu_GMM2={:.4f}'.format(mu_GMM2), 'sig_GMM2={:.4f}'.format(sig_GMM2))
	print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: mu: {:.4f}, std: {:.4f}'.format(mu_data, std_data))
	print('2-step Model moments are: mu: {:.4f}, std: {:.4f}'.format(mu_model2, std_model2))
	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF Values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1),
		 linewidth=2, color='k', label=
		 '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM1, sig_GMM1))
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM2, sig_GMM2),
		 linewidth=2, color='red', label=
		 '2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM2, sig_GMM2))
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1c')
		plt.savefig(output_path)
		plt.close()

	return (mu_GMM2, sig_GMM2)


### Problem 1d:
def data_moments3(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the three data moments for GMM
	(binpct_1, binpct_2, binpct_3).
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N,) vector, income data

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

	OBJECTS CREATED WITHIN FUNCTION:
	bpct_1_dat = scalar in [0, 1], percent of observations
				 0 <= x < 75000
	bpct_2_dat = scalar in [0, 1], percent of observations
				 75000 <= x < 100000
	bpct_3_dat = scalar in [0, 1], percent of observations
				 100000 <= x

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: bpct_1, bpct_2, bpct_3
	--------------------------------------------------------------------
	'''
	bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
	bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
				  xvals.shape[0])
	bpct_3_dat = (xvals[xvals >=100000].shape[0] /
				  xvals.shape[0])
	return (bpct_1_dat, bpct_2_dat, bpct_3_dat)


def model_moments3(mu, sigma):
	'''
	--------------------------------------------------------------------
	This function computes the three model moments for GMM
	(binpct_1, binpct_2, binpct_3).
	--------------------------------------------------------------------
	INPUTS:
	mu     = scalar, mean of the lognormally distributed random variable
	sigma  = scalar > 0, standard deviation of the lognormally distributed
			 random variable

	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		lognorm_pdf()
		xfx()
	
	OBJECTS CREATED WITHIN FUNCTION:
	bpct_1_mod = scalar in [0, 1], percent of model observations in
				 bin 1
	bp_1_err   = scalar > 0, estimated error in the computation of the
				 integral for bpct_1_mod
	bpct_2_mod = scalar in [0, 1], percent of model observations in
				 bin 2
	bp_2_err   = scalar > 0, estimated error in the computation of the
				 integral for bpct_2_mod
	bpct_3_mod = scalar in [0, 1], percent of model observations in
				 bin 3
	bp_3_err   = scalar > 0, estimated error in the computation of the
				 integral for bpct_3_mod
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod
	--------------------------------------------------------------------
	'''
	xfx = lambda x: lognorm_pdf(x, mu, sigma)
	(bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
	(bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
	(bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)    
	return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, income data
	mu     = scalar, mean of the lognormally distributed random variable
	sigma  = scalar > 0, standard deviation of the lognormally distributed
			 random variable
	simple = boolean, =True if errors are simple difference, =False if
			 errors are percent deviation from data moments
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		data_moments3()
		model_moments3()
	
	OBJECTS CREATED WITHIN FUNCTION:
	moms_data  = (3, 1) matrix, column vector of three data moments
	moms_model = (3, 1) matrix, column vector of three model moments
	err_vec    = (3, 1) matrix, column vector of three moment error
				 functions
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
	moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
	moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = 100 * ((moms_model - moms_data) / moms_data)
	
	return err_vec


def criterion3(params, *args):
	'''
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
	args   = length 2 tuple, (xvals, W_hat)
	xvals  = (N,) vector, values of the truncated normally distributed
			 random variable
	W_hat  = (R, R) matrix, estimate of optimal weighting matrix
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		err_vec3()
	
	OBJECTS CREATED WITHIN FUNCTION:
	err        = (3, 1) matrix, column vector of three moment error
				 functions
	crit_val   = scalar > 0, GMM criterion function value
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: crit_val
	--------------------------------------------------------------------
	'''
	mu, sigma = params
	xvals, W = args
	err = err_vec3(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val


def problem1d(pts, mu_init, std_init):
	params_init = np.array([mu_init, std_init])
	W_hat = np.eye(3)
	gmm_args = (pts, W_hat)
	results = opt.minimize(criterion3, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM3, sig_GMM3 = results.x
	params_GMM = np.array([mu_GMM3, sig_GMM3])
	value = criterion3(params_GMM, *gmm_args)[0][0]
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(pts)
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM3, sig_GMM3)
	print()
	print('1d.')
	print('mu_GMM3={:.4f}'.format(mu_GMM3), 'sig_GMM3={:.4f}'.format(sig_GMM3))
	print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: {}, {}, {}'
		  .format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
	print('Model moments are: {}, {}, {}'
		  .format(bpct_1_mod, bpct_2_mod, bpct_3_mod))
	
	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF Values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM3, sig_GMM3),
		 linewidth=2, color='k', label=
		 '$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3))
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1d')
		plt.savefig(output_path)
		plt.close()
	return (mu_GMM3, sig_GMM3)


### Problem 1e:
def problem1e(pts, mu_GMM3, sig_GMM3, mu_init, std_init):
	err3 = err_vec3(pts, mu_GMM3, sig_GMM3, False)
	VCV2 = np.dot(err3, err3.T) / pts.shape[0]
	W_hat2 = lin.pinv(VCV2)
	params_init = np.array([mu_init, std_init])
	gmm_args = (pts, W_hat2)
	results = opt.minimize(criterion3, params_init, args=(gmm_args),
						   method='TNC', bounds=((None, None), (1e-10, None)))
	mu_GMM4, sig_GMM4 = results.x
	params_GMM = np.array([mu_GMM4, sig_GMM4])
	value = criterion3(params_GMM, *gmm_args)[0][0]
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(pts)
	bpct_1_mod2, bpct_2_mod2, bpct_3_mod2 = model_moments3(mu_GMM4, sig_GMM4)
	print()
	print('1e.')
	print('mu_GMM4={:.4f}'.format(mu_GMM4), 'sig_GMM4={:.4f}'.format(sig_GMM4))
	print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value))
	print('Data moments are: {}, {}, {}'
		  .format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
	print('2-step Model moments are: {:.4f}, {:.4f}, {:.4f}'
		  .format(bpct_1_mod2, bpct_2_mod2, bpct_3_mod2))
	graph = True
	if graph:
		fig = plt.figure(figsize = (10,10))
		count, bins, ignored = plt.hist(pts, 30, normed = True, color = 'powderblue')
		plt.title('PDF Values for MACSS student annual income', fontsize=20)
		plt.xlabel('Incomes')
		plt.ylabel('PDF Values')

		dist_pts = np.linspace(0, 150000, 10000)
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM3, sig_GMM3),
		 linewidth=2, color='black', label=
		 '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3),
		 alpha = 0.7)
		plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM4, sig_GMM4),
		 linewidth=2, color='red', label=
		 '2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM4, sig_GMM4),
		 linestyle = '--')
		plt.legend(loc='upper left')
		# plt.show()
		output_path = os.path.join(output_dir, 'Fig_1e')
		plt.savefig(output_path)
		plt.close()

	return (mu_GMM2, sig_GMM2)


### Problem 2:
def err_vec_sick(data, b0, b1, b2, b3):
	moms_data = np.array([[i] for i in data['sick']])
	moms_model = b0 + b1 * data['age'] + \
				 b2 * data['children'] + b3 * data['avgtemp_winter']
	moms_model = np.array([[j] for j in moms_model])
	err_vec = moms_model - moms_data
	return err_vec


def criterion_sick(params, *args):
	data, W = args
	if len(params) == 1: # somehow at the end the optimization function would 
						 # pack parameters into an array, so I have to unpack
		b0, b1, b2, b3 = params[0]
	else:
		b0, b1, b2, b3 = params
	err = err_vec_sick(data, b0, b1, b2, b3)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val


def problem2(data, params):
	b0, b1, b2, b3 = params
	params_init = np.array([b0, b1, b2, b3])
	W = np.eye(200)
	gmm_args = (data, W)
	results_sick = opt.minimize(criterion_sick, params_init, args = (gmm_args),
								method = 'L-BFGS-B')
	b0_GMM, b1_GMM, b2_GMM, b3_GMM = results_sick.x
	params_GMM = np.array([b0_GMM, b1_GMM, b2_GMM, b3_GMM])
	value = criterion_sick(params_GMM, *gmm_args)[0][0]
	print()
	print('2a.')
	print('beta_0 = {:.4f}, '.format(b0_GMM), 'beta_1 = {:.4f}, '.format(b1_GMM),
		  'beta_2 = {:.4f}, '.format(b2_GMM), 'beta_3 = {:.4f}.'.format(b3_GMM))
	print('The value of GMM criterion function at the estimated parameter values is: {:.6f}'
		  .format(value))


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

	mu_init = 9.0
	std_init = 0.3
	mu_GMM1, sig_GMM1 = problem1b(incomepts, mu_init, std_init)

	mu_GMM2, sig_GMM2 = problem1c(incomepts, mu_GMM1, sig_GMM1, mu_init, std_init)

	mu_init = 11.0
	std_init = 0.2
	mu_GMM3, sig_GMM3 = problem1d(incomepts, mu_init, std_init)

	mu_GMM4, sig_GMM4 = problem1e(incomepts, mu_GMM3, sig_GMM3, mu_init, std_init)


	sickdata = pd.read_csv("sick.txt", encoding = 'utf-8-sig')
	params_init = np.array([0,0,0,0])
	problem2(sickdata, params_init)
	

	






