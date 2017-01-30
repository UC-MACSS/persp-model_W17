# Import packages and load the data
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os 


def save_image(name):
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'ernhofer_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path)
    plt.close()

def plot_hist(data):
	weights = (100/ len(data)) * np.ones(len(data))

	count, bins, patches = plt.hist(data, 30, weights = weights)
	plt.title('Future incomes of MACSS students', fontsize=20)
	plt.xlabel('Incomes')
	plt.ylabel('Percentages')

	#plt.show()

	save_image("1a")

def plot_plots(data, mu, sig, name, mu2=None, sig2=None, mult=False):
	count, bins, patches = plt.hist(data, 30, normed=True)
	plt.title('PDF Values for MACSS student annual income', fontsize=20)
	plt.xlabel('Incomes')
	plt.ylabel('PDF Values')

	rmu = round(mu, 5)
	rsig = round(sig, 5)
	dist_pts = np.linspace(0.000001, 150000, 500)
	plt.plot(dist_pts, lognormPDF(dist_pts, mu, sig),
         linewidth=2, color='k', label=('mu:', rmu, "sigma:", rsig) )
	plt.legend(loc='upper left')

	if mult:
		rmu2 = round(mu2, 5)
		rsig2 = round(sig2, 5)
		dist_pts = np.linspace(0.000001, 150000, 500)
		plt.plot(dist_pts, lognormPDF(dist_pts, mu2, sig2),
         linewidth=2, color='r', label=("mu:", rmu2, "sigma:", rsig2) )
		plt.legend(loc='upper left')


	#plt.show()

	save_image(name)

def lognormPDF(x, mu, sigma):
	pdf_vals    = (1/(x * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2))    
	return pdf_vals

def data_moments(xvals):
	mean_data = xvals.mean()
	std_data = xvals.std()

	return mean_data, std_data

def model_moments(mu, sigma):
	#function is creating the mean and variance of the model (of the PDF) 
	xfx = lambda x: x * lognormPDF(x, mu, sigma) # define function getting the mean 
	(mean_model, m_m_err) = intgr.quad(xfx, 0, 150000) #taking the integral of xfx from negative infinity to the cutoff
	x2fx = lambda x: ((x - mean_model) ** 2) * lognormPDF(x, mu, sigma) # define function getting the variance 
	(std_model, v_m_err) = intgr.quad(x2fx, 0, 150000) 

	return mean_model, np.sqrt(std_model)

def err_vec(xvals, mu, sigma, simple):
	mean_data, std_data = data_moments(xvals) #getting data moments (getting the mean and variance of the data)
	moms_data = np.array([[mean_data], [std_data]]) #arranging code into a 2x1 vector 
	mean_model, std_model = model_moments(mu, sigma)#model moments, new function- doesn't need the data 
	#takes my, sigma, cutoff -- uses trunkated normal 
	moms_model = np.array([[mean_model], [std_model]])
	if simple:
	    err_vec = moms_model - moms_data
	else:
	    err_vec = (moms_model - moms_data) / moms_data

	return err_vec

def err_vec4(xvals, params, simple):
	beta0, beta1, beta2, beta3 = params
	moms_data = np.array(xvals['sick'])
	moms_model = beta0 + beta1 * xvals['age'] + beta2 * xvals['children'] + beta3 * xvals['avgtemp_winter']

	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = (moms_model - moms_data) / moms_data

	return err_vec

def criterion(params, *args):
	mu, sigma = params
	xvals, cutoff, W = args
	err = err_vec(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err) 

	return crit_val

def data_moments3(xvals):

	bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
	bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
				  xvals.shape[0])
	bpct_3_dat = (xvals[xvals >=100000].shape[0] /
				  xvals.shape[0])
	return (bpct_1_dat, bpct_2_dat, bpct_3_dat)

def model_moments3(mu, sigma):
	xfx = lambda x: lognormPDF(x, mu, sigma)
	(bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
	(bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
	(bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)    
	return bpct_1_mod, bpct_2_mod, bpct_3_mod

def err_vec3(x, mu, sigma, simple):
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(x)
	moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
	moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
	if simple:
		err_vec = moms_model - moms_data
	else:
		err_vec = 100 * ((moms_model - moms_data) / moms_data)
	
	return err_vec

def criterion3(params, *args):
	mu, sigma = params
	xvals, cutoff, W = args
	err = err_vec3(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val

def criterion4(params, xvals, cutoff, W):
	err = err_vec4(xvals, params, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)

	return crit_val

#problem 1

if __name__ == "__main__":

	income = np.loadtxt("incomes.txt")


	print("\n**********\nQuestion 1\n**********\n")
	print("Part A")

	plot_hist(income)

	print("\nPart B")
	mu_init = 11
	sig_init = 0.2

	params_init = np.array([mu_init, sig_init]) #put guesses into array 
	W_hat = np.eye(2)
	gmm_args = (income, 150000, W_hat)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
	                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))  #constraining the values to be positive 
	mu_GMM1, sig_GMM1 = results.x

	mean_data, std_data = data_moments(income)
	mean_model, std_model = model_moments(mu_GMM1, sig_GMM1)
	err1 = err_vec(income, mu_GMM1, sig_GMM1, False).reshape(2,)
	params_GMM = np.array([mu_GMM1, sig_GMM1])
	GMMvalue = criterion(params_GMM, *gmm_args)[0][0]
	print("Value of GMM criterion:", GMMvalue)
	print('Mean of points:', mean_data, ', Standard Deviation of points:', std_data)
	print('Mean of model:', mean_model, ', Standard Deviation of model:', std_model)

	plot_plots(income, mu_GMM1, sig_GMM1, "1b")

	#Part C

	print("\nPart C")

	err1 = err_vec(income, mu_GMM1, sig_GMM1, False)  #take the GMM estimations of mu and sigma and use it to estimate the error function 
	VCV2 = np.dot(err1, err1.T) / income.shape[0]
	W_hat2 = lin.pinv(VCV2)  # inverse of the variance covariance matrix **use this one

	params_init = np.array([mu_GMM1, sig_GMM1])
	gmm_args = (income, 150000, W_hat2)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
	                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM2, sig_GMM2 = results.x
	params_GMM = np.array([mu_GMM2, sig_GMM2])
	GMMvalue = criterion(params_GMM, *gmm_args)[0][0]
	print('Mu:', mu_GMM2, 'Sigma:', sig_GMM2)
	print("Value of GMM criterion:", GMMvalue)

	plot_plots(income, mu_GMM1, sig_GMM1, "1c", mu_GMM2, sig_GMM2, True)

	print('Mean of points:', mean_data, ', Standard Deviation of points:', std_data)
	print('Mean of model:', mean_model, ', Standard Deviation of model:', std_model)

	print("\nPart D")

	params_init = np.array([mu_init, sig_init])
	W_hat = np.eye(3)
	gmm_args = (income, 150000, W_hat)
	results = opt.minimize(criterion3, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM3, sig_GMM3 = results.x
	params_GMM = np.array([mu_GMM3, sig_GMM3])
	GMMvalue = criterion3(params_GMM, *gmm_args)[0][0]
	data1, data2, data3 = data_moments3(income)
	model1, model2, model3 = model_moments3(mu_GMM3, sig_GMM3)

	print("Value of GMM criterion function:", GMMvalue)
	print("Data moments:", data1, data2, data3)
	print("Model moments:", model1, model2, model3)

	plot_plots(income, mu_GMM3, sig_GMM3, "1d")


	print("Part E")

	err3 = err_vec(income, mu_GMM3, sig_GMM3, False)  #take the GMM estimations of mu and sigma and use it to estimate the error function 
	VCV2 = np.dot(err3, err3.T) / income.shape[0]
	W_hat3 = lin.pinv(VCV2)  # inverse of the variance covariance matrix **use this one

	params_init = np.array([mu_GMM3, sig_GMM3])
	# W_hat3 = np.array([[1. / VCV2[0, 0], 0.], [0., 1. / VCV2[1, 1]]])
	gmm_args = (income, 150000, W_hat3)
	results = opt.minimize(criterion, params_init, args=(gmm_args),
	                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
	mu_GMM4, sig_GMM4 = results.x
	params_GMM4 = np.array([mu_GMM4, sig_GMM4])
	GMMvalue = criterion(params_GMM4, *gmm_args)[0][0]

	print('Mu:', mu_GMM2, 'Sigma:', sig_GMM2)

	print("Value of GMM criterion function:", GMMvalue)


	print("Data moments:", data1, data2, data3)
	print("Model moments:", model1, model2, model3)


	plot_plots(income, mu_GMM3, sig_GMM3, "1e", mu_GMM4, sig_GMM4, True)

	#QUESTION 2

	print("\n**********\nQuestion 2\n**********\n")
	print("\nPart A")
	sick_data = pd.read_csv('sick.txt', header = 0, names = ['sick', 'age', 'children', 'avgtemp_winter'])

	params_init_sick = np.array([0,0,0,0])

	W_hat = np.eye(200)
	gmm_args = (sick_data, 'None', W_hat)
	results = opt.minimize(criterion4, params_init_sick, args=((sick_data, np.inf, W_hat)),
	                   method='L-BFGS-B')
	beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results.x
	print("Beta0:", beta0_GMM, "\nBeta1:", beta1_GMM, "\nBeta2:", beta2_GMM, "\nBeta3:", beta3_GMM)

	GMM_value = criterion4(results.x, *gmm_args)
	print("Value of GMM criterion function:", GMMvalue)


