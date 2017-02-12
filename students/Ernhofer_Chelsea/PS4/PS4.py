# PS 4
# Perspectives in Computational Modeling
# Chelsea Ernhofer

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
import os
np.seterr(all='ignore')


def save_image(name):
    current_path = os.path.split(os.path.abspath(__file__))[0]
    output_folder = 'ernhofer_images'
    output_dir = os.path.join(current_path, output_folder)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, name)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_hist(data):
    weights = (100/ len(data)) * np.ones(len(data))
    fig, ax = plt.subplots(figsize=(8,6))
    count, bins, patches = plt.hist(data, 30, weights = weights)
    plt.title('Incomes of MACSS Graduates', fontsize=20)
    plt.xlabel('Incomes')
    plt.ylabel('Percentages')

    save_image("1a")

def plot_plots(data, mu, sig, mu_name, sig_name, name, mu2=None, sig2=None, mu2_name = None, sig2_name = None, mult=False):
    fig, ax = plt.subplots(figsize=(8,6))
    count, bins, patches = plt.hist(data, 30, normed=True)
    plt.title('Incomes of MACSS Graduates with PDF', fontsize=20)
    plt.xlabel('Incomes')
    plt.ylabel('Percent of Incomes')

    rmu = round(mu_name, 5)
    rsig = round(sig_name, 5)
    dist_pts = np.linspace(0, 150000, 500)
    plt.plot(dist_pts, sts.norm.pdf(dist_pts, mu, sig),
         linewidth=2, color='m', label=('mu: {:.5f}'.format(rmu), 'sigma: {:.5f}'.format(rsig)))
    plt.legend(loc='upper left')

    if mult:
        rmu2 = round(mu2_name, 5)
        rsig2 = round(sig2_name, 5)
        dist_pts = np.linspace(0, 150000, 500)
        plt.plot(dist_pts, sts.norm.pdf(dist_pts, mu2, sig2),
         linewidth=2, color='c', label=("mu:", rmu2, "sigma:", rsig2) )
        plt.legend(loc='upper left')

    save_image(name)

def LN_pdf(xvals,mu,sigma):
    pdf_vals = None
    if xvals.ndim == 1:
        pdf_vals = (1/(xvals*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(xvals)-sigma)**2/2*sigma**2)
    elif xvals.ndim == 2:
        foo = lambda x: (1/(x*sigma*np.sqrt(2*np.pi))) \
                *np.exp(-(np.log(x)-sigma)**2/2*sigma**2)
        pdf_vals = np.apply_along_axis(foo, axis = 1, arr = xvals)

    return pdf_vals

    #added some parenthesis to the equation but don't know if I needed to? :/

def log_norm_draws(mu, sig, n, s):
    norm_dist = rnd.normal(mu,sig, size = (n, s))
    log_draws = np.exp(norm_dist)

    return log_draws

def data_moments(data):
    if data.ndim ==1:
        mu = data.mean()
        sigma = data.std()
    elif data.ndim ==2:
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)

    return mu, sigma

def criterion(params, *args):
    mu, sigma = params
    xvals, unif_vals, s, n, W_hat = args
    sim_vals = log_norm_draws(mu, sigma, n, s)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val

def err_vec(data_vals, sim_vals, mu, sigma, simple):
    mean_data, var_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec



if __name__ == "__main__":

    np.random.seed(seed=1234)
    income = np.loadtxt('incomes.txt')

    # Problem 1(a)
    print("\n**********\nQuestion 1\n**********\n")
    plot_hist(income)

    # Problem 1(b)
    print("\n\n Part (b)")


    xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
    mu = 5.0
    sigma = 1.0
    unif_vals_1 = sts.uniform.rvs(0, 1, size=xvals.size)
    test = LN_pdf(xvals, mu, sigma)
    print(test)

    # Problem 1(c)
    print("\n \n Part (c)")

    s = 300
    n = 200
    W = np.eye(2)

    mu_init = 11.33
    sig_init = .21

    unif_vals_2 = sts.uniform.rvs(0, 1, size=(n, s))

    #getting NxS matrix of lognormal distributions
    log_draws = log_norm_draws(mu_init, sig_init, s, n)

    data_mu, data_sigma = data_moments(income)

    sim_mu, sim_sigma = data_moments(log_draws)

    params_init = np.array([mu_init, sig_init])

    err_vec(income, log_draws, mu_init, sig_init, simple=False)
    crit_test = criterion(params_init, income, unif_vals_2, s, n, W)

    mean_mod = sim_mu.mean()
    var_mod = sim_sigma.mean()

    W = np.eye(2)
    smm_args1 = (income, unif_vals_2, s, n, W)
    results1 = opt.minimize(criterion, params_init, args=(smm_args1),
                              method='L-BFGS-B',
                              bounds=((1e-10, None), (1e-10, None)))
    mu_SMM1, sig_SMM1 = results1.x
    print('mu_SMM1=', mu_SMM1, ' sig_SMM1=', sig_SMM1)

    params_smm = np.array([mu_SMM1, sig_SMM1])
    crit_value1 = criterion(params_smm, *smm_args1)[0][0]
    print("Value of criterion function:", crit_value1)

    # changed from init to SMM1
    sim_vals_1 = log_norm_draws(mu_SMM1, sig_SMM1, s, n)
    mean_sim_1, std_sim_1 = data_moments(sim_vals_1)
    mean_model_1 = mean_sim_1.mean()
    var_model_1 = std_sim_1.mean()

    print('Data mean =', data_mu, ', Data standard deviation =', data_sigma)
    print('Model mean =', mean_model_1, ', Model standard deviation =', var_model_1)

    plot_plots(income, mean_model_1, var_model_1, mu_SMM1, sig_SMM1, "1c")

    # Problem 1(d)
    print("\n \n Part (d)")
    err_2 = err_vec(income, sim_vals_1, mu_SMM1, sig_SMM1, False)
    VCV2_1 = np.dot(err_2, err_2.T) / income.shape[0]
    W_hat2 = lin.pinv(VCV2_1)

    smm_args2 = (income, unif_vals_2, s, n, W_hat2)
    results2 = opt.minimize(criterion, params_smm, args=(smm_args2),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
    mu_SMM2, sig_SMM2 = results2.x
    print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)

    params_smm2 = np.array([mu_SMM2, sig_SMM2])
    crit_value2 = criterion(params_smm2, *smm_args2)[0][0]
    print("Value of criterion function:", crit_value2)

    sim_vals_2 = log_norm_draws(mu_SMM2, sig_SMM2, s, n)
    mean_sim_2, std_sim_2 = data_moments(sim_vals_2)
    mean_model_2 = mean_sim_2.mean()
    var_model_2 = std_sim_2.mean()

    print('Data mean =', data_mu, ', Data standard deviation =', data_sigma)
    print('Model mean =', mean_model_2, ', Model standard deviation =', var_model_2)

    plot_plots(income,  mean_model_2, var_model_2, mu_SMM2, sig_SMM2, "1d", mean_model_1, var_model_1, mu_SMM1, sig_SMM1, True)
