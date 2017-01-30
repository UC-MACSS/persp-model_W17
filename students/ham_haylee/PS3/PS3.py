import numpy as np
import numpy.linalg as lin
import os
import pandas as pd
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def save_image(name):
	'''
	--------------------------------------------------------------------
	The function takes the name of a file and creates a path and directory
	if necessary to save the file. It then saves the file and closes it.
	--------------------------------------------------------------------
	INPUTS:
	name = string, the desired name of the file

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

# 1
def plot_hist(inc_data):
	'''
	--------------------------------------------------------------------
	Creates a histogram with provided data.
	--------------------------------------------------------------------
	INPUTS:
	inc_data  = incomes data

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: save_image

	OBJECTS CREATED WITHIN FUNCTION:
	inc_count = int
	inc_weights = matrix of weights
	plt = histogram of data

	FILES CREATED BY THIS FUNCTION: Fig_1a

	RETURNS: None
	--------------------------------------------------------------------
	'''
	inc_count = len(inc_data)
	inc_weights = (100/inc_count) * np.ones(inc_count)
	plt.hist(inc_data, 30, weights = inc_weights, normed = True)
	plt.title('MACSS Graduate Incomes: 2018-2020', fontsize=20)
	plt.xlabel('Income ($)')
	plt.ylabel('Percentage of Incomes')

	save_image('Fig_1a')

# 1b

def data_moments(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the two data moments for GMM
	(mean(data), variance(data)).
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N,) vector, incomes data

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

	OBJECTS CREATED WITHIN FUNCTION:
	mean_data = scalar, mean value of incomes data
	var_data  = scalar > 0, variance of incomes data

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: mean_data, var_data
	--------------------------------------------------------------------
	'''
	mean_data = xvals.mean()
	var_data = xvals.var()

	return mean_data, var_data


def model_moments(mu, sigma):
	'''
	--------------------------------------------------------------------
	This function computes the two model moments for GMM
	(mean(model data), variance(model data)).
	--------------------------------------------------------------------
	INPUTS:
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		log_norm_pdf()
		xfx()
		x2fx()

	OBJECTS CREATED WITHIN FUNCTION:
	mean_model = scalar, mean value of incomes from model
	m_m_err    = scalar > 0, estimated error in the computation of the
				 integral for the mean of the distribution
	var_model  = scalar > 0, variance of incomes from model
	v_m_err    = scalar > 0, estimated error in the computation of the
				 integral for the variance of the distribution

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: mean_model, var_model
	--------------------------------------------------------------------
	'''
	xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
	(mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
	x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma)
	(var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)

	return mean_model, var_model

def err_vec1(xvals, mu, sigma, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, incomes data
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable
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
	mean_data, var_data = data_moments(xvals)
	moms_data = np.array([[mean_data], [var_data]])
	mean_model, var_model = model_moments(mu, sigma)
	moms_model = np.array([[mean_model], [var_model]])
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = (moms_model - moms_data) / moms_data

	return err_vec

def criterion1(params, *args):
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
	xvals  = (N,) vector, values of the log normally distributed
			 random variable
	W_hat  = (R, R) matrix, estimate of optimal weighting matrix

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		err_vec1

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
	err = err_vec1(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)

	return crit_val

def log_norm_pdf(xvals, mu, sigma):
	'''
	--------------------------------------------------------------------
	Generate pdf values from the log normal pdf with mean mu and standard
	deviation sigma. This function does the same thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, values of the normally distributed random
			 variable
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

	OBJECTS CREATED WITHIN FUNCTION:
	prob_notcut = scalar
	pdf_vals = (N,) vector, normal PDF values for mu and sigma
			   corresponding to xvals data

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: pdf_vals
	--------------------------------------------------------------------
	'''
	pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))

	return pdf_vals

def data_momentsD(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the three data moments for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N,) vector, incomes data

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

	OBJECTS CREATED WITHIN FUNCTION:
	bpct_1_dat = scalar, value of incomes data from those who earn less
		than $75,000
	bpct_2_dat = scalar, value of incomes data from those who earn between
		$75,000 and $100,000
	bpct_3_dat = scalar, value of incomes data from those who earn more
		than $100,000

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: bpct_1_dat, bpct_2_dat, bpct_3_dat
	--------------------------------------------------------------------
	'''
	bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
	bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
				  xvals.shape[0])
	bpct_3_dat = (xvals[xvals >=100000].shape[0] /
				  xvals.shape[0])
	return (bpct_1_dat, bpct_2_dat, bpct_3_dat)

def model_momentsD(mu, sigma):
	'''
	--------------------------------------------------------------------
	This function computes the three model moments for GMM.
	--------------------------------------------------------------------
	INPUTS:
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		log_norm_pdf()
		xfx()
		x2fx()

	OBJECTS CREATED WITHIN FUNCTION:
	xfx = pdf values
	bpct_1_mod, bpct_2_mod, bpct_3_mod = values integrated over specified
		bounds
	bpct_1_err, bpct_2_err, bpct_3_err = the errors returned from intgr.quad


	FILES CREATED BY THIS FUNCTION: None

	RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod
	--------------------------------------------------------------------
	'''
	xfx = lambda x: log_norm_pdf(x, mu, sigma)
	(bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
	(bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
	(bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)

	return bpct_1_mod, bpct_2_mod, bpct_3_mod

def err_vecD(x, mu, sigma, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, incomes data
	mu     = scalar, mean of the normally distributed random variable
	sigma  = scalar > 0, standard deviation of the normally distributed
			 random variable
	simple = boolean, =True if errors are simple difference, =False if
			 errors are percent deviation from data moments

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		data_momentsD()
		model_momentsD()

	OBJECTS CREATED WITHIN FUNCTION:
	moms_data  = matrix, column vector of three data moments
	moms_model = matrix, column vector of three model moments
	err_vec    = matrix, column vector of three moment error
				 functions

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_momentsD(x)
	moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_momentsD(mu, sigma)
	moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = 100 * ((moms_model - moms_data) / moms_data)

	return err_vec

def criterionD(params, *args):
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
	xvals  = (N,) vector, values of the log normally distributed
			 random variable
	W_hat  = (R, R) matrix, estimate of optimal weighting matrix

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		err_vecD

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
	err = err_vecD(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val

# 2
def err_vec2(xvals, params, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for GMM.
	--------------------------------------------------------------------
	INPUTS:
	xvals  = data points
	params = a vector of floats, estimated betas
	simple = boolean, =True if errors are simple difference, =False if
			 errors are percent deviation from data moments

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		None

	OBJECTS CREATED WITHIN FUNCTION:
	mean_data  = scalar, mean value of data
	moms_data  = (200, 1) matrix, column vector of 200 data moments
	moms_model = (200, 1) matrix, column vector of 200 model moments
	err_vec    = (200, 1) matrix, column vector of 200 moment error
				 functions

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
	beta0, beta1, beta2, beta3 = params
	moms_data = np.array(xvals['sick'])
	moms_model = beta0 + beta1 * xvals['age'] + beta2 * xvals['children'] + beta3 * xvals['avgtemp_winter']

	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = (moms_model - moms_data) / moms_data

	return err_vec

def criterion2(params, xvals, W):
	'''
	--------------------------------------------------------------------
	This function computes the GMM weighted sum of squared moment errors
	criterion function value given parameter values and an estimate of
	the weighting matrix.
	--------------------------------------------------------------------
	INPUTS:
	params = a vector of floats, estimated betas
	xvals  = data points
	W_hat  = (R, R) matrix, estimate of optimal weighting matrix

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		err_vec

	OBJECTS CREATED WITHIN FUNCTION:
	err        = (200, 1)matrix, column vector of 200 moment error
				 functions
	crit_val   = scalar > 0, GMM criterion function value

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: crit_val
	--------------------------------------------------------------------
	'''
	err = err_vec2(xvals, params, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)

	return crit_val


if __name__ == '__main__':
	# Question 1
	# a
	print("\n\n-------------------------")
	print("Question 1")
	print("-------------------------")
	print("\n\n#1 Part A, see plot in images folder")
	income_data = np.loadtxt('incomes.txt')
	plot_hist(income_data)

	# b
	print("\n\n#1 Part B, see plot in images folder")
	mu_init = 11
	sig_init = 0.3
	params_init = np.array([mu_init, sig_init])
	W_hat1 = np.eye(2)
	gmm_args = (income_data, W_hat1)
	results = opt.minimize(criterion1, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMM1, sig_GMM1 = results.x
	print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)

	mean_data, var_data = data_moments(income_data)
	mean_model, var_model = model_moments(mu_GMM1, sig_GMM1)
	err1 = err_vec1(income_data, mu_GMM1, sig_GMM1, False).reshape(2,)
	print('Mean of data =', mean_data, ', Variance of data =', var_data)
	print('Mean of model =', mean_model, ', Variance of model =', var_model)

	GMM_value1 = criterion1(results.x, income_data, W_hat1)[0][0]
	print("The value of the GMM:", GMM_value1)

	# plot the initial histogram
	count, bins, ignored = plt.hist(income_data, 30, normed=True)
	plt.title('Incomes of MACSS Graduates with lognormal CDFs', fontsize=15)
	plt.xlabel('Income')
	plt.ylabel('f(x|mu, sigma)')

	# add the new estimated PDF to it
	dist_inc = np.linspace(0.0000001, 150000, 500)
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMM1, sig_GMM1),
		 linewidth=2, color='r', label='B: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMM1, sig_GMM1))
	plt.legend(loc='upper right')
	plt.title('PDF Values for Income Data', fontsize=15)
	save_image('Fig_1b')

	# c
	print("\n#1 Part C, see plot in images folder")
	errc = err_vec1(income_data, mu_GMM1, sig_GMM1, False)
	VCV2 = np.dot(errc, errc.T) / income_data.shape[0]
	W_hatc = lin.pinv(VCV2)

	params_GMM1 = np.array([mu_GMM1, sig_GMM1])
	gmm_argsc = (income_data, W_hatc)

	resultsc = opt.minimize(criterion1, params_GMM1, args=(gmm_argsc),
					   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMMC, sig_GMMC = resultsc.x
	print('mu_GMMC=', mu_GMMC, ' sig_GMMC=', sig_GMMC)

	mean_modelC, var_modelC = model_moments(mu_GMMC, sig_GMMC)
	print('Mean of data =', mean_data, ', Variance of data =', var_data)
	print('Mean of model =', mean_model, ', Variance of model =', var_model)

	GMM_value1c = criterion1(resultsc.x, income_data, W_hatc)[0][0]
	print("The value of the GMM:", GMM_value1c)


	# plot the initial histogram
	count, bins, ignored = plt.hist(income_data, 30, normed=True)
	plt.title('Incomes of MACSS Graduates with lognormal CDFs', fontsize=15)
	plt.xlabel('Income')
	plt.ylabel('f(x|mu, sigma)')

	# add the estimated PDF from partb to it
	dist_inc = np.linspace(0.0000001, 150000, 500)
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMM1, sig_GMM1),
		 linewidth=2, color='r', label='B: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMM1, sig_GMM1))

	# add the newest estimated two-step PDF
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMMC, sig_GMMC),
		linewidth=2, color='g', label='C: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMMC, sig_GMMC))
	plt.legend(loc='upper right')
	plt.title('PDF Values for Income Data', fontsize=15)
	save_image('Fig_1c')

	# d
	print("\n#1 Part D, see plot in images folder")
	W_hatd = np.eye(3)
	gmm_argsd = (income_data, W_hatd)
	resultsd = opt.minimize(criterionD, params_init, args=(gmm_argsd),
					   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMMD, sig_GMMD = resultsd.x
	params_GMMD = np.array([mu_GMMD, sig_GMMD])
	print('mu_GMMD=', mu_GMMD, ' sig_GMMD=', sig_GMMD)

	data1, data2, data3 = data_momentsD(income_data)
	model1, model2, model3 = model_momentsD(mu_GMMD, sig_GMMD)
	print("Data moments:", data1, data2, data3)
	print("Model moments:", model1, model2, model3)

	GMM_value1d = criterionD(params_GMMD, *gmm_argsd)[0][0]
	print("The value of the GMM:", GMM_value1d)

	# plot the initial histogram
	count, bins, ignored = plt.hist(income_data, 30, normed=True)
	plt.title('Incomes of MACSS Graduates with lognormal CDFs', fontsize=15)
	plt.xlabel('Income')
	plt.ylabel('f(x|mu, sigma)')

	# plot the new 3 moment PDF
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMMD, sig_GMMD),
		linewidth=2, color='y', label='D: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMMD, sig_GMMD))
	plt.legend(loc='upper right')
	plt.title('PDF Values for Income Data', fontsize=15)
	save_image('Fig_1d')

	# e
	print("\n#1 Part E, see plot in images folder")
	errE = err_vecD(income_data, mu_GMMD, sig_GMMD, False)
	VCVE = np.dot(errE, errE.T) / income_data.shape[0]
	W_hate = lin.pinv(VCVE)

	gmm_argse = (income_data, W_hate)
	results = opt.minimize(criterionD, params_GMMD, args=(gmm_argse),
						   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMME, sig_GMME = results.x
	params_GMME = np.array([mu_GMME, sig_GMME])
	print('mu_GMME=', mu_GMME, ' sig_GMME=', sig_GMME)

	data1e, data2e, data3e = data_momentsD(income_data)
	model1e, model2e, model3e = model_momentsD(mu_GMME, sig_GMME)
	print("Data moments:", data1e, data2e, data3e)
	print("Model moments:", model1e, model2e, model3e)

	GMM_value1e = criterionD(params_GMME, *gmm_argse)[0][0]
	print("The value of the GMM:", GMM_value1e)


	# plot the initial histogram
	count, bins, ignored = plt.hist(income_data, 30, normed=True)
	plt.title('Incomes of MACSS Graduates with lognormal CDFs', fontsize=15)
	plt.xlabel('Income')
	plt.ylabel('f(x|mu, sigma)')

	# plot the 3 moment PDF from d
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMMD, sig_GMMD),
		linewidth=2, color='y', label='D: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMMD, sig_GMMD))

	# plot the new 3 moment PDF with two-step
	plt.plot(dist_inc, log_norm_pdf(dist_inc, mu_GMME, sig_GMME),
		linewidth=2, color='k', label='E: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_GMME, sig_GMME))
	plt.legend(loc='upper right')
	plt.title('PDF Values for Income Data', fontsize=15)
	save_image('Fig_1e')

	# Question 2
	# a
	print("\n\n-------------------------")
	print("Question 2")
	print("-------------------------")
	print("\n#2 Part A")
	sick_data = pd.read_csv('sick.txt', header = 0)

	params_init_sick = np.array([0,0,0,0])

	W_hat2 = np.eye(200)
	gmm_args = (sick_data, W_hat2)
	results = opt.minimize(criterion2, params_init_sick, args=(gmm_args),
					   method='L-BFGS-B')
	beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results.x
	print("Beta vector estimated by GMM:", beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM)

	params_GMM2 = np.array([beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM])
	GMM_value2 = criterion2(params_GMM2, sick_data, W_hat2)
	print("The value of the GMM:", GMM_value2)
