import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt
import numpy.linalg as lin
np.seterr(all = 'ignore')


def LN_pdf(xvals, mu, sigma):
    '''
    Returns lognorm pdf values
    '''
    ln_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    return ln_dist.pdf(xvals)

def criterion(params, *args):
    '''
    Criterion function for lognormal distribution, mu, and sigma.
    '''
    mu, sigma = params
    xvals, W_hat, num_sims = args
    sim_vals = LN_draws(len(xvals), num_sims, mu, sigma)
    err = err_vec(xvals, sim_vals)
    return np.dot(np.dot(err.T, W_hat), err) 


def moments(vals):
	'''
	Returns mean and standard deviation moments
	'''
	if vals.ndim == 2:
		mean_vec = vals.mean(axis = 0)
		std_vec = vals.std(axis = 0)
		mean = mean_vec.mean()
		std = std_vec.mean()
	elif vals.ndim == 1:
		mean = vals.mean()
		std = vals.std()

	return (mean, std)


def err_vec(data_vals, sim_vals):
	'''
	Calculates error vector between model and simulation.
	'''
	mean_data, var_data = moments(data_vals)
	moms_data = np.array([[mean_data], [var_data]])

	mean_model, var_model = moments(sim_vals)
	moms_model = np.array([[mean_model], [var_model]])

	err_vec = (moms_model - moms_data) / moms_data
	
	return err_vec


def LN_draws(num_obs, num_sims, mu, sigma):
	'''
	Draws a two dimensional arrays for the lognormal distribution.
	'''
	np.random.seed(1234)
	norm_dist  = sts.norm(mu, sigma)
	norm_draws = norm_dist.rvs((num_obs, num_sims))
	ln_draws   = np.exp(norm_draws)
	return ln_draws

if __name__ == '__main__':
	
	data = np.loadtxt('incomes.txt')

	# PART A
	plt.hist(data, bins=30, normed=True)
	plt.title("Incomes of MACSS Grads")
	plt.xlabel("Income")
	plt.ylabel("PDF Value")
	plt.savefig('images/1a')

	## PART B
	print("PART B")
	print(LN_pdf(np.array([[200.0, 270.0], [180.0, 195.5]]), 5.0, 1.0))
	print("\n")

	## PART C
	print("PART C")
	bounds = ((None, None), (1e-10, None))
	num_sims = 300
	mu_init = 9.0
	sig_init = .3

	params_init1 = np.array([mu_init, sig_init])

	weights = np.eye(2)

	smm_args = (data, weights, num_sims)
	results1 = opt.minimize(criterion, params_init1, args=(smm_args), method='L-BFGS-B', bounds=bounds)

	mu_SMM1, sig_SMM1 = results1.x

	sim_vals_SMM1 = LN_draws(len(data), num_sims, mu_SMM1, sig_SMM1)
	data_moments = moments(data)

	print('mu_SMM1=', mu_SMM1, ' sig_SMM1=', sig_SMM1)
	print('Criterion value is: {}'.format(criterion((mu_SMM1, sig_SMM1), *(data, weights, num_sims))))

	model_moments1 = moments(sim_vals_SMM1)
	print('Data moment mu = {}, sigma = {}.\nModel Moments mu = {}, sigma = {}'.format(data_moments[0], data_moments[1], model_moments1[0], model_moments1[1]))
	print("\n")

	x = np.linspace(0, 150000, 500)
	y = plt.plot(x, LN_pdf(x, mu_SMM1, sig_SMM1), label='$\mu=11.33 \sigma=0.21$')
	plt.legend(loc='upper left')
	plt.savefig('images/1c')

	## PART D
	print("PART D")
	# Init with results from SMM
	params_init2 = np.array([11.3, .210])

	errvec = err_vec(data, sim_vals_SMM1)
	weights_twostep = lin.pinv(np.dot(errvec, errvec.T) / data.shape[0])

	smm_args2 = (data, weights_twostep, num_sims)
	results2 = opt.minimize(criterion, params_init2, args = (smm_args2), method = 'TNC', bounds = bounds)

	mu_SMM2, sig_SMM2 = results2.x
	print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)
	print('Criterion value is: {}'.format(criterion((mu_SMM2, sig_SMM2), *(data, weights_twostep, num_sims))))

	sim_vals_SMM2 = LN_draws(len(data), num_sims, mu_SMM2, sig_SMM2)
	model_moments2 = moments(sim_vals_SMM2)
	print('Data moment mu = {}, sigma = {}.\nModel Moments mu = {}, sigma = {}'.format(data_moments[0], data_moments[1], model_moments2[0], model_moments2[1]))
	x = np.linspace(0, 150000, 500)
	y = plt.plot(x, LN_pdf(x, mu_SMM2, sig_SMM2), color='red', label='$\mu=11.33 \sigma=0.22$')
	plt.legend(loc='upper left')
	plt.savefig('images/1d')