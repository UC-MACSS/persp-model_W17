'''
---------- ---------- ---------- ---------- ----------
MACS 30100 - Perspectives on Computational Modeling
HyungJin Cho
---------- ---------- ---------- ---------- ----------
'''

# Importing Pacakges
import os
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


'''
1.(a)
'''
income = np.loadtxt('incomes.txt')
num_bins = 30

# < Directory >
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# < Graph >
fig, ax = plt.subplots()
count, bins, ignored = plt.hist(income, num_bins, normed=True)
plt.title('Distribution of Annual Income')
plt.xlabel('Incomes ($)')
plt.ylabel('PDF Values')
output_path = os.path.join(output_dir, '1(a)')
plt.savefig(output_path, bbox_inches='tight')
# plt.show()
plt.close()



'''
1.(b)
'''
# < Function >
def lognorm_pdf(X_VALUE, mu, sigma):
	pdf_vals = (1/(X_VALUE * sigma * np.sqrt(2 * np.pi))) * np.exp( - (np.log(X_VALUE) - mu)**2 / (2 * sigma**2))
	return pdf_vals

def data_moments(X_VALUE):
	mean_data = X_VALUE.mean()
	std_data = X_VALUE.std()
	return mean_data, std_data

def model_moments(mu, sigma):
	(mean_model, mean_model_ERRORor) = intgr.quad(lambda x: x * lognorm_pdf(x, mu, sigma), 0, 150000)
	(var_model, var_model_ERRORor) = intgr.quad(lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma), 0, 150000)
	std_model = np.sqrt(var_model)
	return mean_model, std_model

def ERROR_vec(X_VALUE, mu, sigma, simple):
	mean_data, std_data = data_moments(X_VALUE)
	moms_data = np.array([[mean_data], [std_data]])
	mean_model, std_model = model_moments(mu, sigma)
	moms_model = np.array([[mean_model], [std_model]])
	if simple:
		ERROR_vec = moms_model - moms_data
	else:
		ERROR_vec = (moms_model - moms_data) / moms_data
	return ERROR_vec

def criterion(params, *args):
	mu, sigma = params
	X_VALUE, W = args
	ERROR = ERROR_vec(X_VALUE, mu, sigma, simple=False)
	CRITERION_VALUE = np.dot(np.dot(ERROR.T, W), ERROR)
	return CRITERION_VALUE

# < Parameters >
mu_init = 9.0
std_init = 0.3
params_init = np.array([mu_init, std_init])
EstimatedWeight = np.eye(2)
gmm_args = (income, EstimatedWeight)
results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B',
                           bounds=((None, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x

params_GMM = np.array([mu_GMM1, sig_GMM1])
mu_data, std_data = data_moments(income)
mu_model, std_model = model_moments(mu_GMM1, sig_GMM1)

print('1.(b)', 'mu_GMM1={:.4f}'.format(mu_GMM1), 'sig_GMM1={:.4f}'.format(sig_GMM1))
print('Value of GMM Criterion: {}'
	  .format(criterion(params_GMM, *gmm_args)[0][0]))
print('Data moments are: mu: {:.4f}, std: {:.4f}'.format(mu_data, std_data))
print('Model moments are: mu: {:.4f}, std: {:.4f}'.format(mu_model, std_model))

# < Graph >
fig, ax = plt.subplots()
count, bins, ignored = plt.hist(income, 30, normed=True)
plt.title('Estimated Lognormal PDF against the Histogram', fontsize=20)
plt.xlabel('Incomes ($)')
plt.ylabel('PDF values')

dist_income = np.linspace(0, 150000, 10000)
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM1, sig_GMM1), linewidth=3, color='r',
             label= '$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM1, sig_GMM1))
plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1))
output_path = os.path.join(output_dir, '1(b)')
plt.savefig(output_path,bbox_inches='tight')
# plt.show()
plt.close()

'''
1.(c)
'''

ERROR1 = ERROR_vec(income, mu_GMM1, sig_GMM1, False)
VCV2 = np.dot(ERROR1, ERROR1.T) / income.shape[0]
EstimatedWeight2 = lin.pinv(VCV2)
params_init = np.array([mu_init, std_init])
gmm_args = (income, EstimatedWeight2)
results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B',
                           bounds=((None, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x
params_GMM = np.array([mu_GMM2, sig_GMM2])
mu_data, std_data = data_moments(income)
mu_model2, std_model2 = model_moments(mu_GMM2, sig_GMM2)


print('1.(c)', 'mu_GMM2={:.4f}'.format(mu_GMM2), 'sig_GMM2={:.4f}'.format(sig_GMM2))
print('Value of GMM Criterion: {}'\
      .format(criterion(params_GMM, *gmm_args)[0][0]))
print('Data moments: mu: {:.4f}, std: {:.4f}'.format(mu_data, std_data))
print('2-step Model moments: mu: {:.4f}, std: {:.4f}'.format(mu_model2, std_model2))

# < Graph >
fig, ax = plt.subplots()
ax.grid()
count, bins, ignored = plt.hist(income, 30, normed=True)
plt.title('Estimated Lognormal PDF against the Histogram', fontsize=20)
plt.xlabel('Incomes ($)')
plt.ylabel('PDF Values')

dist_income = np.linspace(0, 150000, 10000)
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM1, sig_GMM1), linewidth=3, color='r',
             label=  '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM1, sig_GMM1))
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM2, sig_GMM2), linewidth=3, color='b',
                 label= '2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM2, sig_GMM2))
plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1))
output_path = os.path.join(output_dir, '1(c)')
plt.savefig(output_path, bbox_inches='tight')
# plt.show()
plt.close()

def data_moments3(X_VALUE):
	bpct_1_dat = X_VALUE[X_VALUE < 75000].shape[0] / X_VALUE.shape[0]
	bpct_2_dat = (X_VALUE[(X_VALUE >=75000) & (X_VALUE < 100000)].shape[0] /
				  X_VALUE.shape[0])
	bpct_3_dat = (X_VALUE[X_VALUE >=100000].shape[0] /
				  X_VALUE.shape[0])
	return (bpct_1_dat, bpct_2_dat, bpct_3_dat)

def model_moments3(mu, sigma):
	(bpct_1_mod, bp_1_ERROR) = intgr.quad(lambda x: lognorm_pdf(x, mu, sigma), 0, 75000)
	(bpct_2_mod, bp_2_ERROR) = intgr.quad(lambda x: lognorm_pdf(x, mu, sigma), 75000, 100000)
	(bpct_3_mod, bp_3_ERROR) = intgr.quad(lambda x: lognorm_pdf(x, mu, sigma), 100000, 1500000)
	return bpct_1_mod, bpct_2_mod, bpct_3_mod

def ERROR_vec3(X_VALUE, mu, sigma, simple):
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(X_VALUE)
	moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
	moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
	if simple:
		ERROR_vec = moms_model - moms_data
	else:
		ERROR_vec = 100 * ((moms_model - moms_data) / moms_data)
	return ERROR_vec

def criterion3(params, *args):
	mu, sigma = params
	X_VALUE, W = args
	ERROR = ERROR_vec3(X_VALUE, mu, sigma, simple=False)
	CRITERION_VALUE = np.dot(np.dot(ERROR.T, W), ERROR)
	return CRITERION_VALUE


'''
1.(d)
'''
# < Parameter >
mu_init2 = 11
std_init2 = 0.2

params_init2 = np.array([mu_init2, std_init2])
EstimatedWeight = np.eye(3)
gmm_args = (income, EstimatedWeight)
results = opt.minimize(criterion3, params_init2, args=(gmm_args),method='L-BFGS-B',
                           bounds=((None, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results.x
params_GMM = np.array([mu_GMM3, sig_GMM3])
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(income)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM3, sig_GMM3)

print('1.(d)', 'mu_GMM3={:.4f}'.format(mu_GMM3), 'sig_GMM3={:.4f}'.format(sig_GMM3))
print('Value of GMM criterion: {}'\
      .format(criterion3(params_GMM, *gmm_args)[0][0]))
print('Data moments: {}, {}, {}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('Model moments: {:.4f}, {:.4f}, {:.4f}'.format(bpct_1_mod, bpct_2_mod, bpct_3_mod))

# < Graph >
fig, ax = plt.subplots()
ax.grid()
count, bins, ignored = plt.hist(income, 30, normed=True)
plt.title('Estimated Lognormal PDF against the Histogram', fontsize=15)
plt.xlabel('Incomes ($)')
plt.ylabel('PDF Values')

dist_income = np.linspace(0, 150000, 10000)
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM3, sig_GMM3), linewidth=3, color='r',
             label= '$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3))
plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1))
output_path = os.path.join(output_dir, '1(d)')
plt.savefig(output_path, bbox_inches='tight')
# plt.show()
plt.close()

'''
1.(e)
'''

ERROR3 = ERROR_vec3(income, mu_GMM3, sig_GMM3, False)
VCV2 = np.dot(ERROR3, ERROR3.T) / income.shape[0]
EstimatedWeight2 = lin.pinv(VCV2)
gmm_args = (income, EstimatedWeight2)
results = opt.minimize(criterion3, params_init2, args=(gmm_args), method='TNC',
                           bounds=((None, None), (1e-10, None)))
mu_GMM4, sig_GMM4 = results.x
params_GMM = np.array([mu_GMM4, sig_GMM4])
bpct_1_mod2, bpct_2_mod2, bpct_3_mod2 = model_moments3(mu_GMM4, sig_GMM4)

print('1.(e)')
print('mu_GMM4={:.4f}'.format(mu_GMM4), 'sig_GMM4={:.4f}'.format(sig_GMM4))
print('Value of GMM criterion: {}'\
      .format(criterion3(params_GMM, *gmm_args)[0][0]))
print('Data moments: {}, {}, {}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('2-step Model moments: {:.4f}, {:.4f}, {:.4f}'.format(bpct_1_mod2, bpct_2_mod2, bpct_3_mod2))

# < Graph >
fig, ax = plt.subplots()
ax.grid()
count, bins, ignored = plt.hist(income, 30, normed = True)
plt.title('Estimated Lognormal PDF against the Histogram', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('PDF Values')

dist_income = np.linspace(0, 150000, 10000)
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM3, sig_GMM3), linewidth=3, color='r',
                 label= '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3))
plt.plot(dist_income, lognorm_pdf(dist_income, mu_GMM4, sig_GMM4), linewidth=3, color='b',
             label= '2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM4, sig_GMM4),
		 linestyle = '--')
plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1))
output_path = os.path.join(output_dir, '1(e)')
plt.savefig(output_path, bbox_inches='tight')
# plt.show()
plt.close()

'''
2.(a)
'''
# < Function >
sick = pd.read_csv('sick.txt', encoding = 'utf-8-sig')

def sick_err_vec(data, *params):
    b0, b1, b2, b3 = params
    moms_data = np.array(data.iloc[:,0])
    moms_model = b0 + b1 * data.iloc[:,1] + b2 * data.iloc[:,2] + b3 * data.iloc[:,3]
    err_vec = moms_model - moms_data
    return err_vec

def sick_criterion(params, *args):
    data, W = args
    b0, b1, b2, b3 = params
    err = sick_err_vec(data, b0, b1, b2, b3)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val


sick_b0, sick_b1, sick_b2, sick_b3 = 1.0, 0.0, 0.0, 0.01
sick_params = (sick_b0, sick_b1, sick_b2, sick_b3)
sick_WEIGHT = np.eye(len(sick))
sick_gmm_args = (sick, sick_WEIGHT)
sick_bounds = ((None,None),(None,None),(None,None),(None,None))

sick_results = opt.minimize(sick_criterion, sick_params, args = (sick_gmm_args),
                            method='L-BFGS-B', bounds=sick_bounds)

sick_b0_GMM, sick_b1_GMM, sick_b2_GMM, sick_b3_GMM = sick_results.x
sick_crit_val = sick_results.fun

print('2.(a)')
print('beta_0 = {:.3f}, '.format(sick_b0_GMM), 'beta_1 = {:.3f}, '.format(sick_b1_GMM),
		  'beta_2 = {:.3f}, '.format(sick_b2_GMM), 'beta_3 = {:.3f}.'.format(sick_b3_GMM))
print('The value of GMM criterion: {:.6f}'.format(sick_results.fun))
