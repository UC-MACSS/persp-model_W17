import os
import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')

pts = np.loadtxt('incomes.txt') # load data for PS4

# part (a)
def export(image):
    '''
    This function exports a plotted graph and saves it in the images folder.
    If the images folder has not been created, it creates one.
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, image)
    a = plt.savefig(output_path)
    plt.show()
    #plt.close()

    return a

def plot_hist(incomes, num_bins, norm = True):
    # Plot percentages histogram with 30 bins
    if norm:
        hist_wgts = None
    else:
        hist_wgts = (1 / income.shape[0]) * np.ones_like(income)

    fig, ax = plt.subplots()
    plt.hist(incomes, num_bins, weights = hist_wgts, normed = norm)
    plt.title('MACS Graduate Annual Incomes Normed Histogram', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF values of Income')
    plt.tight_layout

    export("1a")

print('Part (a)')
plot_hist(pts, 30, norm = True)

#part(b)
def LN_pdf(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a lognormal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma. These draws correspond to an (N x S) 
    matrix of randomly generated draws from a uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    '''

    pdf_vals = sts.lognorm.pdf(unif_vals, s = sigma, scale = np.exp(mu), loc = 0)
    return(pdf_vals)


print('Part (b)')
xvals = np.array([[200.0, 270.0], [180.0,195.5]])
mu_b = 5.0
sigma_b = 1.0
print('Lognormal PDF values', LN_pdf(xvals, mu_b, sigma_b))

#part(c)
#lognormal distributions draws of NxS (300x200)
#use average income and standard deviation of income as two moments

np.random.seed(seed=1234) #set seed

def LN_draws(unif_vals, mu, sigma):
    lognorm_draws = sts.norm.ppf(unif_vals, loc = mu, scale = sigma) #loc = mean | scale = sd
    e_lognorm_draws = np.exp(lognorm_draws)

    return e_lognorm_draws

def data_moments(xvals):
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        sd_data = xvals.std()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        sd_data = xvals.std(axis=0)

    return mean_data, sd_data

def model_moments(sim_vals):
    mean_sim, sd_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    sd_model = sd_sim.mean()

    return mean_model, sd_model

def err_vec(data_vals, sim_vals, mu, sigma, simple):
    mean_data, sd_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [sd_data]])
    mean_sim, sd_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    sd_model = sd_sim.mean()
    moms_model = np.array([[mean_model], [sd_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec

def criterion(params, *args):
    mu, sigma = params
    xvals, unif_vals, What = args
    sim_vals = LN_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple = False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val

print('Part (c)')
N = 200
S = 300
mu_init_1 = 4
sig_init_1 = 0.6
params_init_1 = np.array([mu_init_1, sig_init_1])
W_hat = np.eye(2)
unif_vals_1 = sts.uniform.rvs(0, 1, size=(N,S))

smm_args1_1 = (pts, unif_vals_1, W_hat)
results1_1 = opt.minimize(criterion, params_init_1, args=(smm_args1_1),
                          method='SLSQP',
                          bounds=((1e-10, None), (1e-10, None)))   #types of constrained optimisation 
                                                                  # TNC, SLSQP, L-BFGS-B
mu_SMM1_1, sig_SMM1_1 = results1_1.x #successful termination 
print('mu_1c=', mu_SMM1_1, ' sig_1c=', sig_SMM1_1)

crit1c = criterion(np.array([mu_SMM1_1, sig_SMM1_1]), pts, unif_vals_1, W_hat)
print('The SMM criterion function at the estimated parameters is', crit1c)

print('The mean and standard deviation of data moments are {} and {}'.format(data_moments(pts)[0], data_moments(pts)[1]))
SMM1_vals = LN_draws(unif_vals_1, mu_SMM1_1, sig_SMM1_1)
print('The mean and standard deviation of modelled moments are {} and {}'.format(model_moments(SMM1_vals)[0], model_moments(SMM1_vals)[1]))

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACS Graduate Annual Incomes Normed Histogram', fontsize=12)
plt.xlabel('Annual Income')
plt.ylabel('PDF values of Income')

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 150000, 300)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=2, color='y', label='$\mu_{1c}=11.330$,$\sigma_{1c}=0.2092$')
plt.legend(loc='upper left')

export('1c')

# part (d)
print('Part (d)')
err1d = err_vec(pts, SMM1_vals, mu_SMM1_1, sig_SMM1_1, False) #sub SMM mu and sigma to get the draws from the LN distribution
VCV1d = np.dot(err1d, err1d.T) / pts.shape[0] 
W_hat1d = lin.pinv(VCV1d)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

smm_args1d = (pts, unif_vals_1, W_hat1d) #SMM args take originally generated uniform values
results1d = opt.minimize(criterion, params_init_1, args=(smm_args1d),
                  method='SLSQP',
                  bounds=((1e-10, None), (1e-10, None)))
mu1d, sig1d = results1d.x #successful termination 
print('mu_1d=', mu1d, ' sig_1d=', sig1d)

crit1d = criterion(np.array([mu1d, sig1d]), pts, unif_vals_1, W_hat1d)
print('The SMM criterion function at the estimated parameters is', crit1d)

print('The mean and standard deviation of data moments are {} and {}'.format(data_moments(pts)[0], data_moments(pts)[1]))
SMM1d_vals = LN_draws(unif_vals_1, mu1d, sig1d)
print('The mean and standard deviation of modelled moments are {} and {}'.format(model_moments(SMM1d_vals)[0], model_moments(SMM1d_vals)[1]))

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACS Graduate Annual Incomes Normed Histogram', fontsize=12)
plt.xlabel('Annual Income')
plt.ylabel('PDF values of Income')

# Plot the estimated SMM PDF 1(c)
dist_pts = np.linspace(0, 150000, 300)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1),
         linewidth=2, color='y', label='$\mu_{1c}=11.330$,$\sigma_{1c}=0.2092$')
plt.legend(loc='upper left')

# Plot the estimated SMM PDF 1 (d)
plt.plot(dist_pts, LN_pdf(dist_pts, mu1d, sig1d),
         linewidth=2, color='r', label='$\mu_{1d}=11.330$,$\sigma_{1d}=0.2092$')
plt.legend(loc=2,prop={'size':10})

export('1d')
