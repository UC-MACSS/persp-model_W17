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



def lognormal_draws(obs, sims, mu, sigma):

	np.random.seed(1234)

	norm_dist = sts.norm(mu, sigma)
	norm_draws = norm_dist.rvs((obs, sims))

	return np.exp(norm_draws)



def moments(xvals):
	'''
	--------------------------------------------------------------------
	This function computes the two data moments for SMM
	(mean(data), variance(data)) from both the actual data and from the
	simulated data.
	--------------------------------------------------------------------
	INPUTS:
	xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
			test scores data, either real world or simulated. Real world
			data will come in the form (N,). Simulated data comes in the
			form (N,) or (N, S).
	
	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
	
	OBJECTS CREATED WITHIN FUNCTION:
	mean_data = scalar or (S,) vector, mean value of test scores data
	var_data  = scalar > 0 or (S,) vector, variance of test scores data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: mean_data, var_data
	--------------------------------------------------------------------
	'''
	if xvals.ndim == 1:
		mean_data = xvals.mean()
		var_data = xvals.var()
	elif xvals.ndim == 2:
		mean_data = xvals.mean(axis=0).mean()
		var_data = xvals.var(axis=0).mean()
	
	return mean_data, var_data

def err_vec(data_vals, sim_vals, simple):
	'''
	--------------------------------------------------------------------
	This function computes the vector of moment errors (in percent
	deviation from the data moment vector) for SMM.
	--------------------------------------------------------------------
	'''
	mean_data, var_data = moments(data_vals)
	moms_data = np.array([[mean_data], [var_data]])

	mean_sim, var_sim = moments(sim_vals)

	moms_model = np.array([[mean_sim], [var_sim]])
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
	'''
	mu, sigma = params
	xvals, W_hat, sims = args
	sim_vals = lognormal_draws(len(xvals), sims, mu, sigma)

	err = err_vec(xvals, sim_vals, simple=False)
	crit_val = np.dot(np.dot(err.T, W_hat), err) 
	
	return crit_val

#Define function that generates values of a truncated normal pdf
def LN_pdf(x, mu, sigma):
	return np.exp(- np.square((np.log(x) - mu) / (np.sqrt(2) * sigma))) / (x * sigma * np.sqrt(2 * np.pi))


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


	if question == 'c':
		plt.plot(dist_pts, LN_pdf(dist_pts, mu_smm1, sig_smm1),\
		linewidth=2, color='r', \
		label='1: $\mu$ and $\sigma$ estimated by SMM with identity as the weight matrix')
	
	elif question == 'd':
		plt.plot(dist_pts, LN_pdf(dist_pts, mu_smm1, sig_smm1),\
		linewidth=2, color='r', \
		label='1: $\mu$ and $\sigma$ estimated by SMM with identity as the weight matrix')

		plt.plot(dist_pts, LN_pdf(dist_pts, mu_smm2, sig_smm2),\
			linewidth=2, color='g', \
			label='2: $\mu$ and $\sigma$ estimated by two-step SMM ]')


		
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
Write a lognormal function
--------------------------------------------------------------------
'''
xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
mu = 5.0
sigma = 1.0
print("Result from 1b:")
print(LN_pdf(xvals, mu, sigma))
'''
--------------------------------------------------------------------
Question 1: c
Create S = 300 simulations, each with N = 200 observations on income from the lognormal distribution above
moments: average and sd
weight matrix: identity matrix
--------------------------------------------------------------------
'''
print('\nResults from 1c:')
np.random.seed(seed=1234)
sim = 300
obs = 200
mu_init = 9
sigma_init = 0.3
params_init = np.array([mu_init, sigma_init])
W = np.eye(2)
args = (data, W, sim)
results1 = opt.minimize(criterion, params_init, args=(args),
						  method='L-BFGS-B',
						  bounds=((None, None), (1e-10, None)))

mu_smm1, sig_smm1 = results1.x

sim_vals_smm1 = lognormal_draws(len(data), sim, mu_smm1, sig_smm1)
data_moments = moments(data)

print('mu_smm1=', mu_smm1, ' sig_smm1=', sig_smm1)
print('Criterion value is:',criterion((mu_smm1, sig_smm1),*args))

model_moments1 = moments(sim_vals_smm1)
print('Data moment mu = {}, variance = {}.\nModel Moments mu = {}, variance = {}'
	  .format(data_moments[0], data_moments[1], model_moments1[0], model_moments1[1]))
plot("c")

'''
--------------------------------------------------------------------
Question 1: d
two-step- we'll just use our result from part c as our initial values
--------------------------------------------------------------------
'''
print('\nResults from 1d:')
np.random.seed(seed=1234)
params_init = np.array([mu_smm1, sig_smm1])

# VCV2_1 = np.dot(err2_1, err2_1.T) / pts.shape[0]
# print(VCV2_1)
# W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
# print(W_hat2_1)


errvec = err_vec(data, sim_vals_smm1, False)
W_twostep = lin.pinv(np.dot(errvec, errvec.T) / data.shape[0])
args_twostep = (data, W_twostep, sim)
results2 = opt.minimize(criterion, params_init, args=(args_twostep),
						  method='L-BFGS-B',
						  bounds=((None, None), (1e-10, None)))

mu_smm2, sig_smm2 = results2.x

sim_vals_smm2 = lognormal_draws(len(data), sim, mu_smm2, sig_smm2)
print('mu_smm2=', mu_smm2, ' sig_smm2=', sig_smm2)
print('Criterion value is:',criterion((mu_smm2, sig_smm2),*args_twostep))

model_moments2 = moments(sim_vals_smm2)
print('Data moment mu = {}, variance = {}.\nModel Moments mu = {}, variance = {}'
	  .format(data_moments[0], data_moments[1], model_moments2[0], model_moments2[1]))
plot("d")
