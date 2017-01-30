# Import packages and load the data
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import seaborn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


'''
lognorm(s =sig_1, scale =np.exp(mu_1)).pdf(dist_pts)
'''

#Define function that generates values of a truncated normal pdf
def trunc_lognorm_pdf(xvals, mu, sigma, cutoff):
	if cutoff == None:
		prob_notcut = 1.0
	else:
		prob_notcut = sts.lognorm.cdf(cutoff, sigma, loc= mu, scale = np.exp(mu))
	pdf_vals    = sts.lognorm(s =sigma, scale =np.exp(mu)).pdf(xvals)/prob_notcut
	
	return pdf_vals

def data_moments(xvals, num_moments):
	if num_moments == 2:
		mean_data = xvals.mean()
		var_data = xvals.var()
		return mean_data, var_data
	else:
		a = 75000
		b = 100000
		bpct_1_dat = xvals[xvals < a].shape[0] / xvals.shape[0]
		bpct_2_dat = (xvals[(xvals >=a) & (xvals < b)].shape[0] /
					  xvals.shape[0])
		bpct_3_dat = xvals[xvals >= b].shape[0] / xvals.shape[0]

		return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_moments(mu, sigma, cutoff, num):
	#using 2 moments: part b
	if num == 2:
		xfx = lambda x: x * trunc_lognorm_pdf(x, mu, sigma, cutoff)
		(mean_model, m_m_err) = intgr.quad(xfx, 0, cutoff)
		x2fx = lambda x: ((x - mean_model) ** 2) * trunc_lognorm_pdf(x, mu, sigma, cutoff)
		(var_model, v_m_err) = intgr.quad(x2fx, 0, cutoff)

		return mean_model, var_model
	else:
		a = 75000
		b = 100000

		fx = lambda x: trunc_lognorm_pdf(x, mu, sigma, cutoff)
		(bpct_1_mod, bp_1_err) = intgr.quad(fx, -np.inf, a)
		(bpct_2_mod, bp_2_err) = intgr.quad(fx, a, b)
		(bpct_3_mod, bp_3_err) = intgr.quad(fx, b, cutoff)
		return bpct_1_mod, bpct_2_mod, bpct_3_mod

		

def err_vec(xvals, mu, sigma, cutoff, simple, num):
	if num == 2:
		mean_data, var_data = data_moments(xvals, 2)
		moms_data = np.array([[mean_data], [var_data]])
		mean_model, var_model = model_moments(mu, sigma, cutoff, 2)
		moms_model = np.array([[mean_model], [var_model]])

	else:
		bpct_1_dat, bpct_2_dat, bpct_3_dat = \
			data_moments(xvals, num)
		moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
		bpct_1_mod, bpct_2_mod, bpct_3_mod = \
			model_moments(mu, sigma, cutoff, 3)
		moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
							   
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = ((moms_model - moms_data) / moms_data)
	return err_vec


def criterion(params, *args):
	mu, sigma = params
	xvals, cutoff, W, num= args
	err = err_vec(xvals, mu, sigma, cutoff, False, num)
	crit_val = np.dot(np.dot(err.T, W), err) 
	return crit_val

def plot(question):
	#Create directory if images directory does not already exist
	cur_path = os.path.split(os.path.abspath(__file__))[0]
	output_fldr = 'images'
	output_dir = os.path.join(cur_path, output_fldr)
	if not os.access(output_dir, os.F_OK):
		os.makedirs(output_dir)

	fig = plt.figure()
	dist_pts = np.linspace(0, 150000, 500)

	plt.hist(data, bins = 30, normed = True)


	if question == 'b':
		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1, data.max()),\
		linewidth=2, color='r', \
		label='1: $\mu$ and $\sigma$ estimated by GMM with identity as the weight matrix')
	
	elif question == 'c':
		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1, data.max()),\
		linewidth=2, color='r', \
		label='1: $\mu$ and $\sigma$ estimated by GMM with identity as the weight matrix')

		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM2, sig_GMM2, data.max()),\
		linewidth=2, color='g', \
		label='2: $\mu$ and $\sigma$ estimated by the two-step GMM estimator')

	elif question == 'd':
		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM4, sig_GMM4, data.max()),\
		linewidth=2, color='orange', \
		label='1: $\mu$ and $\sigma$ estimated by GMM with percentages as moments')

	elif question == 'e':
		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM4, sig_GMM4, data.max()),\
		linewidth=2, color='orange', \
		label='1: $\mu$ and $\sigma$ estimated by GMM with percentages as moments and identity')
		
		plt.plot(dist_pts, trunc_lognorm_pdf(dist_pts, mu_GMM5, sig_GMM5, data.max()),\
		linewidth=2, color='purple', \
		label='1: $\mu$ and $\sigma$ estimated by GMM with percentages as moments and identity')

		
	plt.title('Histogram of incomes of MACSS graduates', fontsize=15)
	plt.xlabel('Annual Income($)')
	plt.ylabel('Percent of observations in bin')
	plt.legend(loc='upper right')
	plt.xlim(0, 150000)

	output_path = os.path.join(output_dir, 'Fig_1{}'.format(question))
	plt.savefig(output_path)

	plt.close()



'''
--------------------------------------------------------------------
Question 1: a
--------------------------------------------------------------------
'''
data= np.loadtxt('incomes.txt')
q_a = True
if q_a:
	plot('a')

'''
--------------------------------------------------------------------
Question 1: b
GMM estimation with identity matrix as our estimate for the optimal weighting matrix 
--------------------------------------------------------------------
'''
q_b = True
if q_b:
	print("\nQuestion 1b:")
	mu_init = 9
	sig_init = 0.5
	params_init = np.array([mu_init, sig_init])
	W_hat = np.eye(2)
	gmm_args = (data, data.max(), W_hat, 2)
	results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMM1, sig_GMM1 = results.x
	print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)

	#Report the value of your GMM criterion function at the estimated parameter values.
	print("Value of GMM criterion function at the estimated parameter values:", \
		criterion(results.x, *(data, data.max(), W_hat, 2)))
	#Report and compare your two data moments against your two model moments at the estimated parameter values.
	mean_data, var_data = data_moments(data, 2)
	mean_model, var_model = model_moments(mu_GMM1, sig_GMM1, data.max(), 2)
	err1 = err_vec(data, mu_GMM1, sig_GMM1, data.max(), False, 2).reshape(2,)
	print('Mean of points =', mean_data, ', Variance of points =', var_data)
	print('Mean of model =', mean_model, ', Variance of model =', var_model)
	print('Error vector=', err1)

	plot('b')

'''
--------------------------------------------------------------------
Question 1: c
Perform the two-step GMM estimator by using your estimates from 
part (b) with two moments to generate an estimator for the variance covariance matrix
--------------------------------------------------------------------
'''
q_c = True
if q_c:
	print("\nQuestion 1c:")

	err1 = err_vec(data, mu_GMM1, sig_GMM1, data.max(), False, 2)
	VCV2 = np.dot(err1, err1.T) / data.shape[0]
	W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned


	params_init = np.array([mu_GMM1, sig_GMM1])
	gmm_args = (data, data.max(), W_hat2, 2)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMM2, sig_GMM2 = results.x
	print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)

	#criterion function value
	print("Value of GMM criterion function at the estimated parameter values:", \
		criterion(results.x, *(data, data.max(), W_hat2, 2)))

	#comparing model moments with data moments
	#Report and compare your two data moments against your two model moments at the estimated parameter values.
	#mean_data, var_data = data_moments(data)
	mean_model, var_model = model_moments(mu_GMM2, sig_GMM2, data.max(), 2)
	err2 = err_vec(data, mu_GMM2, sig_GMM2, data.max(), False, 2)
	err2_ = err2.reshape(2,)
	print('Mean of points =', mean_data, ', Variance of points =', var_data)
	print('Mean of model =', mean_model, ', Variance of model =', var_model)
	print('Error vector=', err2_)

	#plot
	plot('c')
'''
--------------------------------------------------------------------
Question 1: d
GMM estimator with different moments and identity matrix
--------------------------------------------------------------------
'''
q_d = True
if q_d:
	print("\nQuestion 1d:")
	mu_init = 10
	sig_init = 0.5
	params_init = np.array([mu_init, sig_init])
	W_hat = np.eye(3)
	gmm_args = (data, data.max(), W_hat, 3)
	results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMM4, sig_GMM4 = results.x
	print('mu_GMM4=', mu_GMM4, ' sig_GMM4=', sig_GMM4)

	#Report the value of your GMM criterion function at the estimated parameter values.
	print("Value of GMM criterion function at the estimated parameter values:", \
		criterion(results.x, *(data, data.max(), W_hat, 3)))

	#Report and compare your two data moments against your two model moments at the estimated parameter values.
	p1_data, p2_data, p3_data = data_moments(data, 3)
	p1_model, p2_model, p3_model = model_moments(mu_GMM4, sig_GMM4, data.max(), 3)
	err4 = err_vec(data, mu_GMM4, sig_GMM4, data.max(), False, 3).reshape(3,)
	print('P1 of points =', p1_data, ', P2 of points =', p2_data, ', P3 of points =', p3_data)
	print('P1 of model =', p1_model, ', P2 of model =', p2_model, ', P3 of model =', p3_model)
	print('Error vector=', err4)

	plot('d')

'''
--------------------------------------------------------------------
Question 1: e
GMM estimator with different moments and variance covariance matrix
--------------------------------------------------------------------
'''
q_e = True
if q_e:
	print("\nQuestion 1e:")

	err4 = err_vec(data, mu_GMM4, sig_GMM4, data.max(), False, 3)
	VCV4 = np.dot(err4, err4.T) / data.shape[0]
	W_hat4 = lin.pinv(VCV4)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

	params_init = np.array([mu_GMM4, sig_GMM4])
	# W_hat3 = np.array([[1. / VCV2[0, 0], 0.], [0., 1. / VCV2[1, 1]]])
	gmm_args = (data, data.max(), W_hat4, 3)
	results = opt.minimize(criterion, params_init, args=(gmm_args),\
	                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
	mu_GMM5, sig_GMM5 = results.x
	print('mu_GMM5=', mu_GMM5, ' sig_GMM5=', sig_GMM5)

	#Report the value of your GMM criterion function at the estimated parameter values.
	print("Value of GMM criterion function at the estimated parameter values:", \
		criterion(results.x, *(data, data.max(), W_hat4, 3)))
	#Report and compare your two data moments against your two model moments at the estimated parameter values.

	p1_data, p2_data, p3_data = data_moments(data, 3)
	p1_model, p2_model, p3_model = model_moments(mu_GMM5, sig_GMM5, data.max(), 3)
	err5 = err_vec(data, mu_GMM5, sig_GMM5, data.max(), False, 3).reshape(3,)
	print('P1 of points =', p1_data, ', P2 of points =', p2_data, ', P3 of points =', p3_data)
	print('P1 of model =', p1_model, ', P2 of model =', p2_model, ', P3 of model =', p3_model)
	print('Error vector=', err5)

	plot('e')
'''
--------------------------------------------------------------------
Question 2:
GMM estimator with different moments
--------------------------------------------------------------------
'''
def err_vec_sick(data, *args):
    b0, b1, b2, b3 = args
    sick_data = np.array(data['sick'])
    sick_model = b0 + b1 * data['age'] + b2 * data['children'] + b3 * data['avgtemp_winter']
    err_vec = sick_data - sick_model
    return err_vec

def criterion_sick(params, *args):
    dat = args[0]
    W = np.eye(len(dat))
    err = err_vec_sick(dat, *params)
    crit_val = np.dot(np.dot(err.T, W), err) 
    return crit_val

q_2 = True
if q_2:
	print("\nQuestion 2:")

	#load data
	sick_data= pd.read_csv('sick.txt', header = 0)

	#estimate
	beta0_init = 1
	beta1_init = 0
	beta2_init = 0
	beta3_init = 0
	params_init = np.array([beta0_init, beta1_init, beta2_init, beta3_init])
	bounds = ((None, None),(None, None),(None, None),(None, None))

	results = opt.minimize(criterion_sick, params_init, args=sick_data, method = 'L-BFGS-B')
	b0, b1, b2, b3 = results.x
	print('Beta0 =', b0, ' Beta1 =', b1, 'Beta2 =', b2, 'Beta3 =', b3)

	#evaluation
	res = criterion_sick(np.array(results.x), *(sick_data,2))
	print('Criterion function value with GMM estimates = {}\n'.format(res))

