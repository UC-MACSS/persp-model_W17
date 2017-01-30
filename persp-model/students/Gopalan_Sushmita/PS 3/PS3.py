# PS 3
# Perspectives in Computational Modeling
# Sushmita V Gopalan


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
import pandas as pd

cmap1 = matplotlib.cm.get_cmap('summer')

# load the data
incomes = np.loadtxt('incomes.txt')


def save_image(name):
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'sush_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path)
    plt.close()

def plot_hist(data):
    weights = (100/ len(data)) * np.ones(len(data))
    count, bins, patches = plt.hist(data, 30, weights = weights)
    plt.title('Incomes of MACSS Graduates', fontsize=20)
    plt.xlabel('Incomes')
    plt.ylabel('Percentages')
    save_image("1a")

def plot_plots(data, mu, sig, name, mu2=None, sig2=None, mult=False):
	count, bins, patches = plt.hist(data, 30, normed=True)
	plt.title('Incomes of MACSS Graduates with PDF', fontsize=20)
	plt.xlabel('Incomes')
	plt.ylabel('PDF Values')


	rmu = round(mu, 5)
	rsig = round(sig, 5)
	dist_pts = np.linspace(0, 150000, 500)
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu, sig),
         linewidth=2, color='m', label=('mu: {:.5f}'.format(rmu), 'sigma: {:.5f}'.format(rsig)))
	plt.legend(loc='upper left')

	if mult:
		rmu2 = round(mu2, 5)
		rsig2 = round(sig2, 5)
		dist_pts = np.linspace(0, 150000, 500)
		plt.plot(dist_pts, log_norm_pdf(dist_pts, mu2, sig2),
         linewidth=2, color='c', label=("mu:", rmu2, "sigma:", rsig2) )
		plt.legend(loc='upper left')
	#plt.show()
	save_image(name
    )

# Define function that generates values of a log normal pdf
def log_norm_pdf(xvals, mu, sigma):
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
					np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
    return pdf_vals

def data_moments(xvals):
    mean_data = xvals.mean()
    std_data = xvals.std()
    return mean_data, std_data

def model_moments(mu, sigma):
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    std_model = np.sqrt(var_model)
    return mean_model, std_model

def err_vec(xvals, mu, sigma, simple):
    mean_data, var_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_model, var_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    return err_vec

def criterion(params, *args):
    mu, sigma = params
    xvals,  W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val

# Problem 1(a)
print("\n**********\nQuestion 1\n**********\n")
plot_hist(incomes)

# Problem 1(b)
mu_init = 11.3
std_init = 0.21
pts = incomes # just in case
# put initial guesses into an array
params_init = np.array([mu_init, std_init])
W_hat = np.eye(2) # weighting matrix is 2x2 identity
gmm_args = (incomes, W_hat)
# optimize
results = opt.minimize(criterion, params_init, args=(gmm_args),
					   method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
params_GMM = np.array([mu_GMM1, sig_GMM1])
value = criterion(params_GMM, *gmm_args)
mu_data, std_data = data_moments(incomes)
mu_model, std_model = model_moments(mu_GMM1, sig_GMM1)

#Report the value of your GMM criterion function at the estimated param- eter values.
criterion_val = criterion(params_GMM,*gmm_args)
print("The value of the criterion function is ",criterion_val[0][0])

#Report and compare your two data moments against your two model moments at the estimated parameter values.
mean_data, var_data = data_moments(incomes)
mean_model, var_model = model_moments(mu_GMM1, sig_GMM1)
err1 = err_vec(incomes, mu_GMM1, sig_GMM1, False).reshape(2,)
print('Estimates: \n mu = ', mu_GMM1, 'sigma = ', sig_GMM1)
print('Mean of Incomes =', mean_data, ', Standard Deviation of Incomes =', var_data)
print('Mean of Model =', mean_model, ', Standard Deviation of Model =', var_model)
print('Error vector=', err1)
#print(results)

# Plot histogram and draw pdf over it
plot_plots(incomes, mu_GMM1, sig_GMM1, "1b")

# Problem 1(c)

# (c) Perform the two-step GMM estimator by using your estimates from part (b)
# with two moments to generate an estimator for the variance covariance matrix Ωˆ2step,
# which you then use to get the two-step estimator for the optimal weighting matrix Wˆ 2step.
# Report your estimates as well as the criterion function value at these estimates.
print("\n Part (c)")
# get vcv matrix from estimates in part b
err1 = err_vec(incomes, mu_GMM1, sig_GMM1, False)
VCV = np.dot(err1, err1.T) / incomes.shape[0]
W_hat2 = lin.pinv(VCV)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
gmm2_args = (incomes, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm2_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x
params_GMM2 = np.array([mu_GMM2, sig_GMM2])
criterion_val = criterion(params_GMM2,*gmm_args)
mu_data, sig_data = data_moments(incomes)
mu_model2, sig_model2 = model_moments(mu_GMM2, sig_GMM2)
err2 = err_vec(incomes, mu_GMM2, sig_GMM2, False).reshape(2,)
print('Estimates: \n mu = ', mu_GMM2, 'sigma = ', sig_GMM2)
print("The value of the criterion function is ",criterion_val[0][0])
#print(results)
plot_plots(incomes, mu_GMM1, sig_GMM1, "1c", mu_GMM2, sig_GMM2, True)

#Report and compare your two data moments against your two model moments at the estimated parameter values.
print('Mean of Incomes =', mean_data, ', Standard Deviation of Incomes =', sig_data)
print('Mean of Model =', mean_model, ', Standard Deviation of Model =', sig_model2)
print('Error vector=', err2)
#print(results)

# Problem 1(d):

def data_moments3(xvals):
	bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
	bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
				  xvals.shape[0])
	bpct_3_dat = (xvals[xvals >=100000].shape[0] /
				  xvals.shape[0])
	return (bpct_1_dat, bpct_2_dat, bpct_3_dat)

def model_moments3(mu, sigma):
	xfx = lambda x: log_norm_pdf(x, mu, sigma)
	(bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
	(bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
	(bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)
	return bpct_1_mod, bpct_2_mod, bpct_3_mod

def err_vec3(xvals, mu, sigma, simple):
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
	mu, sigma = params
	xvals, W = args
	err = err_vec3(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val

params_init = np.array([11.3, 0.21])
W_hat = np.eye(3)
gmm_args = (incomes, W_hat)
results = opt.minimize(criterion3, params_init, args=(gmm_args),
						   method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results.x
params_GMM = np.array([mu_GMM3, sig_GMM3])
criterion_value = criterion3(params_GMM, *gmm_args)
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM3, sig_GMM3)
print('Part 1(d)')
print('\n Estimates: ','\n mu = ',mu_GMM3,' sig = ',sig_GMM3)
print('\n The value of GMM criterion function at the estimated parameter values = ',criterion_value[0][0])
print('Data moments: ')
print('Proportion of incomes less than $75000 = ', bpct_1_dat)
print('Proportion of incomes between $75000 and $100000 = ', bpct_2_dat)
print('Proportion of incomes between $100000 and $150000 = ', bpct_3_dat)
print('\n \n Model moments: ')
print('Proportion of incomes less than $75000 = ', bpct_1_mod)
print('Proportion of incomes between $75000 and $100000 = ', bpct_2_mod)
print('Proportion of incomes between $100000 and $150000 = ', bpct_3_mod)

# Problem 1(e)
# get vcv matrix from estimates in part d
err3 = err_vec(incomes, mu_GMM3, sig_GMM3, False)
VCV2 = np.dot(err3, err3.T) / incomes.shape[0]
W_hat3 = lin.pinv(VCV2)

params_init = np.array([mu_GMM3, sig_GMM3])
gmm_args = (incomes, W_hat3)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM4, sig_GMM4 = results.x
params_GMM4 = np.array([mu_GMM4, sig_GMM4])
criterion_value4 = criterion(params_GMM4, *gmm_args)

bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM4, sig_GMM4)
print('Part 1(e)')
print('\n Estimates: ','\n mu = ',mu_GMM4,' sig = ',sig_GMM4)
print('\n The value of GMM criterion function at the estimated parameter values = ',criterion_value4[0][0])
print('Data moments: ')
print('Proportion of incomes less than $75000 = ', bpct_1_dat)
print('Proportion of incomes between $75000 and $100000 = ', bpct_2_dat)
print('Proportion of incomes between $100000 and $150000 = ', bpct_3_dat)
print('\n \n Model moments: ')
print('Proportion of incomes less than $75000 = ', bpct_1_mod)
print('Proportion of incomes between $75000 and $100000 = ', bpct_2_mod)
print('Proportion of incomes between $100000 and $150000 = ', bpct_3_mod)

plot_plots(incomes, mu_GMM3, sig_GMM3, "1d")
plot_plots(incomes, mu_GMM3, sig_GMM3, "1e", mu_GMM4, sig_GMM4, True)

# Problem 2

def err_vec2(xvals, params, simple):
    beta0, beta1, beta2, beta3 = params
    moms_data = np.array(xvals['\ufeffsick'])
    moms_model = beta0 + beta1 * xvals['age'] + beta2 * xvals['children'] + beta3 * xvals['avgtemp_winter']

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec

def criterion2(params, xvals, W):
    err = err_vec2(xvals, params, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

# Question 2
print("\n**********\nQuestion 2\n**********\n")
sick_data = pd.read_csv('sick.txt', header = 0)

params_init_sick = np.array([0,0,0,0])

W_hat2 = np.eye(200)
# use every data point as a moment
gmm_args = (sick_data, W_hat2)
results = opt.minimize(criterion2, params_init_sick, args=(gmm_args),
                   method='L-BFGS-B')
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results.x

params_GMM = np.array([beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM])
criterion_value2 = criterion2(params_GMM, sick_data, W_hat2)
print('\n The value of GMM criterion function at the estimated parameter values = ',criterion_value2)

print("Beta0 = ", beta0_GMM)
print("Beta1 = ", beta1_GMM)
print("Beta2 = ", beta2_GMM)
print("Beta3 = ", beta3_GMM)
print("Beta vector estimated by GMM:", beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM)
