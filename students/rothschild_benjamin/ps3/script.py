import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

data = np.loadtxt('incomes.txt')

def hist(data):
    plt.title('Incomes for MACSS Students 2018-2020 Graduates', fontsize=20)
    plt.xlabel(r'$Incomes$')
    plt.ylabel(r'$PDF value of Incomes$')
    num_bins = 30
    weights = (100.0 / data.shape[0]) * np.ones_like(data)
    n, bin_cuts, patches = plt.hist(data, num_bins, weights=weights, normed=True)
    plt.savefig('images/1a')

def lognorm_pdf(xvals, mu, sigma):
    '''
    Returns the pdf vals for a log normal distribution with xvals, mu and aigma 
    '''
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))    
    return pdf_vals

def data_moments(data):
    '''
    Calculates the mean and std dev for a dataset
    '''
    mu_data    = data.mean()
    sigma_data = data.std()    
    return mu_data, sigma_data

def model_moments(mu, sigma):
    '''
    Calculates mean, std dev from mu, sigma
    '''
    xfx = lambda x: x * lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    std_model = np.sqrt(var_model)
    return mean_model, std_model

def err_vec(xvals, mu, sigma, percent):
    '''
    
    '''
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [std_model]])
    if percent:
        err_vec = moms_model - moms_data # in case moms_data is too close to zero
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec

def criterion(params, *args):
    '''
    
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, percent=False)
    crit_val = np.dot(np.dot(err.T, W), err)     
    return crit_val


def model_moments_1d(mu, sigma):
    xfx = lambda x: lognorm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod

def data_moments_1d(xvals):
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] / xvals.shape[0])
    bpct_3_dat = (xvals[xvals >=100000].shape[0] / xvals.shape[0])
    return (bpct_1_dat, bpct_2_dat, bpct_3_dat)

def err_vec_1d(x, mu, sigma, simple):
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments_1d(x)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments_1d(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    return err_vec

def criterion_1d(params, *args):
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_vec_1d(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val

def criterion_2(params, *args):
	data = args[0] # See 'WTF'
	W = np.eye(len(data))
	err = err_vec_2(data, *params)
	crit_val = np.dot(np.dot(err.T, W), err)

	return(crit_val)

# Exercise 2
def err_vec_2(data, *betas):
    b0, b1, b2, b3 = betas
    sick_data = np.array(data['sick'])
    sick_model = b0 + b1 * data['age'] + b2 * data['children'] + b3 * data['avgtemp_winter']

    err_vec = sick_data - sick_model
    return(err_vec)

if __name__ == '__main__':
	## 1A
	print("1A.")
	hist(data)
	print("\n")

	## 1B
	print("1B.")
	mu_init = 10.0
	sigma_init = 0.2
	params_init = np.array([mu_init, sigma_init])
	W_hat_1b = np.eye(2)
	gmm_args = (data, W_hat_1b)
	results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((None, None), (1e-10, None)))

	mu_1b, sig_1b = results['x']
	pts = np.linspace(0, 150000, 10000)
	plt.plot(pts, lognorm_pdf(pts, mu_1b, sig_1b), linewidth=2, color='red', label='$\mu_1b$ = {:.4f}, $\sigma_1b$ = {:.4f}'.format(mu_1b, sig_1b))
	plt.legend(loc='upper right')
	plt.savefig('images/1b')

	print("mu = {:.4f}, sigma = {:.4f}".format(mu_1b, sig_1b))
	print("value for GMM criterion func", results['fun'][0][0])
	mean_1b, std_1b = model_moments(mu_1b, sig_1b)
	print("model moments mu = {:.4f}, sigma = {:.4f}".format(mean_1b, std_1b))
	mu_data, sig_data = data_moments(data)
	print("data moments mu = {:.4f}, sigma = {:.4f}".format(mu_data, sig_data))
	print("\n")

	## 1C
	print("1C")
	mu_init = 9.0
	std_init = 0.2

	err1 = err_vec(data, mu_1b, sig_1b, False)
	VCV2 = np.dot(err1, err1.T) / data.shape[0]
	W_hat_1c = lin.pinv(VCV2)
	params_init = np.array([mu_init, std_init])
	gmm_args = (data, W_hat_1c)
	results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((None, None), (1e-10, None)))

	mu_1c, sig_1c = results['x']
	# print(results)
	params_GMM = np.array([mu_1c, sig_1c])
	value = criterion(params_GMM, *gmm_args)[0][0]
	mean_data, std_data = data_moments(pts)
	mean_1c, std_1c = model_moments(mu_1c, sig_1c)

	print("mu = {:.4f}, sigma = {:.4f}".format(mu_1c, sig_1c))
	print("value for GMM criterion func", results['fun'][0][0])
	mean_1c, std_1c = model_moments(mu_1c, sig_1c)
	print("model moments mu = {:.4f}, sigma = {:.4f}".format(mean_1c, std_1c))
	mu_data, sig_data = data_moments(data)
	print("data moments mu = {:.4f}, sigma = {:.4f}".format(mu_data, sig_data))

	data_1c = np.linspace(0, 150000, 10000)
	plt.plot(pts, lognorm_pdf(data_1c, mu_1c, sig_1c), linewidth=2, color='green', label='$\mu_1c$ = {:.4f}, $\sigma_1c$ = {:.4f}'.format(mu_1c, sig_1c))
	plt.legend(loc='upper right')
	plt.savefig('images/1c')
	plt.gcf().clear()
	print("\n")

	## 1D
	print("1D")
	mu_init = 11.0
	sigma_init = 0.3
	params_init = np.array([mu_init, sigma_init])
	W_hat_1d = np.eye(3)
	gmm_args = (data, 150000, W_hat_1d)
	results = opt.minimize(criterion_1d, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((None, None), (1e-10, None)))

	mu_1d, sig_1d = results['x']
	# print(results)
	params_GMM = np.array([mu_1d, sig_1d])

	print("mu = {:.3f}, sigma = {:.3f}".format(mu_1d, sig_1d))
	print("value for GMM criterion func", results['fun'][0][0])
	mom_1, mom_2, mom_3 = model_moments_1d(mu_1d, sig_1d)
	print("mom 1 = {:.3f}, mom 2 = {:.3f}, mom 3 = {:.3f}".format(mom_1, mom_2, mom_3))
	mom_1, mom_2, mom_3 = data_moments_1d(data)
	print("mom 1 = {:.3f}, mom 2 = {:.3f}, mom 3 = {:.3f}".format(mom_1, mom_2, mom_3))

	hist(data)
	data_1d = np.linspace(0, 150000, 10000)
	plt.plot(pts, lognorm_pdf(data_1d, mu_1d, sig_1d), linewidth=2, color='green', label='$\mu_1d$ = {:.4f}, $\sigma_1d$ = {:.4f}'.format(mu_1d, sig_1d))
	plt.legend(loc='upper right')
	plt.savefig('images/1d')
	print("\n")

	## 1E
	print("1E")
	## 1E
	err_1e = err_vec(data, mu_1d, sig_1d, True)  
	# print(err_1e)
	VCV_1e = np.dot(err_1e, err_1e.T) / data.shape[0]
	# print(VCV2)
	W_hat_1e = lin.pinv(VCV_1e)
	# print(W_hat_1e)
	params_init = np.array([mu_1d, sig_1d])

	gmm_args = (data, W_hat_1e)
	results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_1e, sig_1e = results.x
	params_1e = np.array([mu_1e, sig_1e])
	GMMvalue = criterion(params_1e, *gmm_args)[0][0]

	print('Mu:', mu_1e, 'Sigma:', sig_1e)

	print("Value of GMM criterion function:", GMMvalue)


	print("mu = {:.3f}, sigma = {:.3f}".format(mu_1e, sig_1e))
	print("value for GMM criterion func", results['fun'][0][0])
	mom_1, mom_2, mom_3 = model_moments_1d(mu_1e, sig_1e)
	print("mom 1 = {:.3f}, mom 2 = {:.3f}, mom 3 = {:.3f}".format(mom_1, mom_2, mom_3))
	mom_1, mom_2, mom_3 = data_moments_1d(data)
	print("mom 1 = {:.3f}, mom 2 = {:.3f}, mom 3 = {:.3f}".format(mom_1, mom_2, mom_3))

	plt.plot(pts, lognorm_pdf(data_1d, mu_1e, sig_1e), linewidth=2, color='red', label='$\mu_1e$ = {:.4f}, $\sigma_1e$ = {:.4f}'.format(mu_1e, sig_1e))
	plt.legend(loc='upper right')
	plt.savefig('images/1e')
	print("\n")

	## 2
	print("2")
	sick_dat = pd.read_csv('sick.txt', encoding='utf-8-sig')
	params_init2a = np.array([0, 0, 0, 0])
	results2a = opt.minimize(criterion_2, params_init2a, args=sick_dat, method = 'L-BFGS-B')
	beta_vec = results2a.x
	print('Beta 0, 1, 2, and 3: {}'.format(beta_vec))
	print('Criterion function estimates = {}\n'.format(criterion_2(np.array(beta_vec), *(sick_dat, 0))))


