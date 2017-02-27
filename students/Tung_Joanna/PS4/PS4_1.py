# Import packages and load the data
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import os
import PS4_1_Funcs as FUNC


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

    output_path = os.path.join(output_dir, 'PS4_1a')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

#PART 1b: Write function LN_pdf() and test with sample array

# The function LN_pdf()
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector or (N,S) array, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector or (N,S) array lognormal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''

    pdf_vals = (1/(sigma * xvals * np.sqrt(2 * np.pi)) *
                np.exp( -(np.log(xvals) - mu)**2 / (2 * sigma**2)))

    return pdf_vals

# Test function LN_pdf() with sample array = data
data = np.array([[200,270], [180, 195.5]])
mu_test = 5.0
sig_test = 1.0
pdf_vals = LN_pdf(data, mu_test, sig_test)
print('Part 1b: Test of sample array [[200, 270], [180, 195.5]] returns the following array \
of lognormal pdf values for mu = 5.0, sigma = 1.0: {}'.format(pdf_vals))

# PART 1c: Run simulated method of moments:
#  - 300 simulations
#  - 200 observations per simulation
#  - Use weighting matrix = identity matrix
#  - Use random seed 1234

# Perform SMM estimation of parameters mu and sigma

# Set initial conditions
mu_init_1 = 11
sig_init_1 = 0.68
params_init_1 = np.array([mu_init_1, sig_init_1])

# Set the identity matrix as the weighting matrix
W_hat1_1 = np.eye(2)

# Set the random seed to 1234 to ensure the same array draw every time
np.random.seed(1234)
# Create a (300,200) array of random values between 0 and 1, choosing from uniform distribution
unif_vals = sts.uniform.rvs(0, 1, size=(300,200))

# Assign arguments
smm_args1_1 = (pts, unif_vals, W_hat1_1)

# Conduct optimization procedure
results1_1 = opt.minimize(FUNC.criterion, params_init_1, args=(smm_args1_1),
                          method='L-BFGS-B', bounds = ((1e-10, None), (1e-10, None)))
                        #options={'eps': 0.2},

# Assign results to variables

mu_SMM1_1, sig_SMM1_1 = results1_1.x
fun = results1_1.fun

print('Part 1c: The estimated value of the parameters are mu_SMM1_1={} and sig_SMM1_1={}.'.format(mu_SMM1_1, sig_SMM1_1))
print('The value of the criterion function using method L-BFGS-B is {}.'.format(fun))

# Calculate the values of data moments and model moments
data_mean, data_std = FUNC.data_moments(pts)
sims_mean, sims_std = FUNC.data_moments(np.exp(FUNC.norm_draws(unif_vals, mu_SMM1_1, sig_SMM1_1)))
mod_mean, mod_std = sims_mean.mean(), sims_std.mean()

print('The data moments are {} for mean and {} for standard deviation.'.format(data_mean, data_std))
print('The model moments are {} for mean and {} for standard deviation.'.format(mod_mean, mod_std))

# Generate graph of the pdf

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

    plt.plot(dist_pts, LN_pdf(dist_pts, 11, 0.68),
         linewidth=2, color='green', label='initial: $\mu_init$,$\sigma_init$')
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=6, color='black', label='Part 1c: $\mu_{SMM1_1}$,$\sigma_{SMM1_1}$')

    plt.legend(loc='upper left')
    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS4_1c')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1d: Run SMM with the optimal Weighting Matrix on the same data and parameters
# from Part 1b

# Generate the error vector from Part 1c
sims = np.exp(FUNC.norm_draws(unif_vals, mu_SMM1_1, sig_SMM1_1))
err1_1 = FUNC.err_vec(pts, sims, mu_SMM1_1, sig_SMM1_1, False)
print('The error vector for Part 1c is: {}'.format(err1_1))
# Use the error vector to calculate the variance-covariance matrix
VCV2_1 = np.dot(err1_1, err1_1.T) / pts.shape[0]
print('The variance-covariance matrix from Part 1c is: {}'.format(VCV2_1))

# Calculate the inverse of the variance-covariance matrix
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print('The optimal weighting matrix for Part 1d is: {}'.format(W_hat2_1))

# Perform the SMM estimation

# Set initial conditions
params_init_2 = np.array([mu_SMM1_1, sig_SMM1_1])

# Set the random seed to 1234 to ensure the same array draw every time
np.random.seed(1234)
# Recreate a (300,200) array of random values between 0 and 1, choosing from uniform distribution
unif_vals = sts.uniform.rvs(0, 1, size=(300,200))

# Assign arguments, with optimal weighting matrix
smm_args2_1 = (pts, unif_vals, W_hat2_1)
results2_1 = opt.minimize(FUNC.criterion, params_init_2, args=(smm_args2_1),
                          method='L-BFGS-B', bounds = ((1e-10, None), (1e-10, None)))#, options={'eps': 0.2})
# Assign results to variables
mu_SMM2_1, sig_SMM2_1 = results2_1.x
fun = results2_1.fun

print('Part 1d: The estimated value of the parameters are mu_SMM2_1={} and sig_SMM2_1={}.'.format(mu_SMM2_1, sig_SMM2_1))
print('The value of the criterion function using method L-BFGS-B is {}.'.format(fun))

# Calculate the values of data moments and model moments
data_mean, data_std = FUNC.data_moments(pts)
sims_mean, sims_std = FUNC.data_moments(np.exp(FUNC.norm_draws(unif_vals, mu_SMM2_1, sig_SMM2_1)))
mod_mean, mod_std = sims_mean.mean(), sims_std.mean()

# Calculate the error vector
sims = np.exp(FUNC.norm_draws(unif_vals, mu_SMM2_1, sig_SMM2_1))
err2_1 = FUNC.err_vec(pts, sims, mu_SMM2_1, sig_SMM2_1, False)
print('The error vector for Part 1d is: {}'.format(err2_1))

print('The data moments are {} for mean and {} for standard deviation.'.format(data_mean, data_std))
print('The model moments are {} for mean and {} for standard deviation.'.format(mod_mean, mod_std))

# Generate graph of the pdf

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

    plt.plot(dist_pts, LN_pdf(dist_pts, 11, 0.68),
         linewidth=2, color='green', label='initial: $\mu_init$,$\sigma_init$')
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=6, color='black', label='Part 1c: $\mu_{SMM1_1}$,$\sigma_{SMM1_1}$')
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2_1, sig_SMM2_1),
         linewidth=2, color='magenta', label='Part 1d: $\mu_{SMM2_1}$,$\sigma_{SMM2_1}$')

    plt.legend(loc='upper left')
    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    output_path = os.path.join(output_dir, 'PS4_1d')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
