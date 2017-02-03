# Import packages
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import PS3_1_Funcs_1 as FUNC
import PS3_1_Funcs_2 as FUNC2
import os

# Load the data
pts = np.loadtxt('incomes.txt')

# PART 1a: Plot and save historgram of data

plot = True

if plot:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0+0.12, box.width*0.85, box.height * 0.85])

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS3_1a')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1b: Use GMM to estimate parameters mu and sigma, using the lognorml pdf as the model
# The identity matrix is used as the Weighting matrix.

# Initial inputs for mu and sigma
mu_init = 11.3
sig_init = 0.68
# Assign inputs to the initial parameter array
params_init = np.array([mu_init, sig_init])
# Assign identity matrix as Weighting matrix
W_hat = np.eye(2)
# Assign inputs to the arguments
gmm_args = (pts, W_hat)
# Run optimization using GMM. criterion function from PS3_1_Funcs_1 script)
results = opt.minimize(FUNC.criterion, params_init, args=(gmm_args),
                       method='SLSQP', bounds=((1e-10, None), (1e-10, None)))
# Assign results to mu_GMM1b and sig_GMM1b
mu_GMM1b, sig_GMM1b = results.x
mean_data, std_data = FUNC.data_moments(pts)
mean_model, std_model = FUNC.model_moments(mu_GMM1b, sig_GMM1b)
err1b = FUNC.err_vec(pts, mu_GMM1b, sig_GMM1b, False).reshape(2,)
fun = results.fun
# Print results
print('mu_GMM1b=', mu_GMM1b, ' sig_GMM1b=', sig_GMM1b)
print('mu_init=', mu_init, 'sig_init=', sig_init )
print('Mean of data =', mean_data, ', Std dev of data =', std_data)
print('Mean of model =', mean_model, ', Std dev of model =', std_model)
print('Error vector=', err1b)
print('value of GMM criterion function=', fun)

# Plot results of GMM for Part 1b against the histogram from Part 1a

plot = True

if plot:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0+0.12, box.width*0.85, box.height * 0.85])

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    dist_pts = np.linspace(0, 150000, 500)
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1b, s=sig_GMM1b, scale=np.exp(mu_GMM1b)),
             linewidth=4, color='r', label='$\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1b, sig_GMM1b))
    plt.legend(loc='upper right')

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS3_1b')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1c: perform a second optimization attempt using the 2-step GMM estimatoro method.
# The variance-covariance matrix from Part 1b will be used to get the two-step estimator
# for the optimal weighting matrix.

# Get the inverse of the variance covariance matrix from Part 1b, assign to new Weighting Matrix W_hat2
err1b = FUNC.err_vec(pts, mu_GMM1b, sig_GMM1b, False)
VCV2 = np.dot(err1b, err1b.T) / pts.shape[0]
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print('Part 1c: 2-step Weighting Matrix', W_hat2)

# Assign initial inputs for mu and sigma as the estimated mu and sigma from Part 1b
# Assign inputs to the initial parameter array
params_init = np.array([mu_GMM1b, sig_GMM1b])
# Assign inputs to the arguments, using new 2-step weighting matrix
gmm_args = (pts, W_hat2)
# Run optimization using GMM. criterion function from PS3_1_Funcs_1 script)
results = opt.minimize(FUNC.criterion, params_init, args=(gmm_args),
                       method='SLSQP', bounds=((1e-10, None), (1e-10, None)))
# Assign results to mu_GMM1c and sig_GMM1c
mu_GMM1c, sig_GMM1c = results.x
mean_data, std_data = FUNC.data_moments(pts)
mean_model, std_model = FUNC.model_moments(mu_GMM1c, sig_GMM1c)
err1c = FUNC.err_vec(pts, mu_GMM1c, sig_GMM1c, False).reshape(2,)
fun = results.fun
# Print results
print('mu_GMM1c=', mu_GMM1c, ' sig_GMM1c=', sig_GMM1c)
print('mu_init=', mu_GMM1b, 'sig_init=', sig_GMM1b)
print('Mean of data =', mean_data, ', Std dev of data =', std_data)
print('Mean of model =', mean_model, ', Std dev of model =', std_model)
print('Error vector1c=', err1c)
print('value of GMM criterion function=', fun)

# Plot results of GMM for Part 1c against the histogram from Part 1a and 1b

plot = True

if plot:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0+0.12, box.width*0.85, box.height * 0.85])

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    dist_pts = np.linspace(0, 150000, 500)
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1b, s=sig_GMM1b, scale=np.exp(mu_GMM1b)),
             linewidth=4, color='r', label='Part 1b: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1b, sig_GMM1b))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1c, s=sig_GMM1c, scale=np.exp(mu_GMM1c)),
             linewidth=4, color='g', label='Part 1c: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1c, sig_GMM1c))
    plt.legend(loc='upper right')

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS3_1c')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1d: Redo the parameter estimation using alternate data moments. The data moments
# will correspond instead to the percentage of observations x that meet criteria x <$75,000,
# $75,00 <= x <= $100,000 and x > $100,000

# Initial inputs for mu and sigma
mu_init = 11.3
sig_init = 0.68
# Assign inputs to the initial parameter array
params_init = np.array([mu_init, sig_init])
# Assign identity matrix as Weighting matrix
W_hat = np.eye(3)
# Assign inputs to the arguments
gmm_args = (pts, W_hat)
# Run optimization using GMM criterion4 function from PS3_1_Funcs_2 script)
results = opt.minimize(FUNC2.criterion4, params_init, args=(gmm_args),
                       method='SLSQP', bounds=((1e-10, None), (1e-10, None)))
# Assign results to variables
mu_GMM1d, sig_GMM1d = results.x
fun = results.fun
bpct_1_dat, bpct_2_dat, bpct_3_dat = FUNC2.data_moments4(pts)
bpct_1_mod, bpct_2_mod, bpct_3_mod = FUNC2.model_moments4(mu_GMM1d, sig_GMM1d)
err1d= FUNC2.err_vec4(pts, mu_GMM1d, sig_GMM1d, False).reshape(3,)
# Print results
print('mu_GMM1d=', mu_GMM1d, ' sig_GMM1d=', sig_GMM1d)
print('mu_init=', mu_init, 'sig_init=', sig_init )
print('Dat_1 points =', bpct_1_dat, 'Dat_2 points =', bpct_2_dat, 'Dat_3 points =', bpct_3_dat)
print('Mod_1 points =', bpct_1_mod, 'Mod_2 points =', bpct_2_mod, 'Mod_3 points =', bpct_3_mod)
print('Error vector=', err1d)
print('value of GMM criterion function=', fun)

# Plot results of GMM for Part 1d against the histogram from Part 1a-c

plot = True

if plot:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0+0.12, box.width*0.85, box.height * 0.85])

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    dist_pts = np.linspace(0, 150000, 500)
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1b, s=sig_GMM1b, scale=np.exp(mu_GMM1b)),
             linewidth=4, color='r', label='Part 1b: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1b, sig_GMM1b))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1c, s=sig_GMM1c, scale=np.exp(mu_GMM1c)),
             linewidth=4, color='g', label='Part 1c: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1c, sig_GMM1c))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1d, s=sig_GMM1d, scale=np.exp(mu_GMM1d)),
             linewidth=6, color='black', label='Part 1d: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1d, sig_GMM1d))
    plt.legend(loc='upper right', fontsize = 6)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS3_1d')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1e: perform a second optimization attempt using the 2-step GMM estimatoro method.
# The variance-covariance matrix from Part 1d will be used to get the two-step estimator
# for the optimal weighting matrix.

# Get the inverse of the variance covariance matrix from Part 1d, assign to new Weighting Matrix W_hat2
err1d = FUNC2.err_vec4(pts, mu_GMM1d, sig_GMM1d, False)
VCV2 = np.dot(err1d, err1d.T) / pts.shape[0]
W_hat3 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print('Part 1d: 2-step Weighting Matrix', W_hat3)

# Assign inputs to the initial parameter array: use values from Part 1d as initial values
params_init = np.array([mu_GMM1d, sig_GMM1d])
# Assign inverse of variance-covariance matrix from part 1d as Weighting matrix
gmm_args = (pts, W_hat3)
# Run optimization using GMM criterion4 function from PS3_1_Funcs_2 script)
results = opt.minimize(FUNC2.criterion4, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
# Assign results to variables
mu_GMM1e, sig_GMM1e = results.x
fun = results.fun
bpct_1_dat_e, bpct_2_dat_e, bpct_3_dat_e = FUNC2.data_moments4(pts)
bpct_1_mod_e, bpct_2_mod_e, bpct_3_mod_e = FUNC2.model_moments4(mu_GMM1e, sig_GMM1e)
err1e = FUNC2.err_vec4(pts, mu_GMM1e, sig_GMM1e, False).reshape(3,)
# Print results
print('mu_GMM1e=', mu_GMM1e, ' sig_GMM1e=', sig_GMM1e)
print('mu_init=', mu_GMM1d, 'sig_init=', sig_GMM1d)
print('Dat_1 points =', bpct_1_dat_e, 'Dat_2 points =', bpct_2_dat_e, 'Dat_3 points =', bpct_3_dat_e)
print('Mod_1 points =', bpct_1_mod_e, 'Mod_2 points =', bpct_2_mod_e, 'Mod_3 points =', bpct_3_mod_e)
print('Error vector=', err1e)
print('value of GMM criterion function=', fun)

# Plot results of GMM for Part 1e against the histogram from Part 1a-d

plot = True

if plot:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.05, box.y0+0.12, box.width*0.85, box.height * 0.85])

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    dist_pts = np.linspace(0, 150000, 500)
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1b, s=sig_GMM1b, scale=np.exp(mu_GMM1b)),
             linewidth=4, color='r', label='Part 1b: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1b, sig_GMM1b))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1c, s=sig_GMM1c, scale=np.exp(mu_GMM1c)),
             linewidth=4, color='g', label='Part 1c: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1c, sig_GMM1c))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1d, s=sig_GMM1d, scale=np.exp(mu_GMM1d)),
             linewidth=6, color='black', label='Part 1d: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1d, sig_GMM1d))
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, loc=mu_GMM1e, s=sig_GMM1e, scale=np.exp(mu_GMM1e)),
             linewidth=2, color='m', label='Part 1e: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_GMM1e, sig_GMM1e))
    plt.legend(loc='upper right', fontsize = 6)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS3_1e')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
