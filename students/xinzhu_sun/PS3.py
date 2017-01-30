# Perspectives on Computational Modeling
# Problem Set 3
# Name: Xinzhu Sun

import pandas as pd
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import numpy.linalg as lin
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as intgr
from scipy.stats import lognorm

df = pd.read_table('incomes.txt', names = ['incomes'])
df_sick = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter'])

# Problem 1
mu_init = 11
sig_init = 0.2

def lognorm_pdf(xvals, mu, sigma, cutoff):
    if cutoff == np.inf:
        prob_notcut = 1.0
    else:
        prob_notcut = sts.lognorm.cdf(cutoff, s=sigma, loc=mu, scale=exp(mu))
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * exp( - (ln(xvals) - mu) ** 2 / (2 * sigma ** 2))) / prob_notcut
    return pdf_vals

if True:
    # (a) Plot a histogram 
    if True:
        num_obs = len(df['incomes'])
        fig, ax = plt.subplots()
        hist_wgts = (1 / num_obs) * np.ones(num_obs)
        num_bins = 30
        plt.hist(df['incomes'], num_bins, weights=hist_wgts, normed=True)
        plt.xlim(0, 160000)
        plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=20)
        plt.xlabel('Annual income')
        plt.ylabel('Percent of students')
        plt.savefig('1a')
        plt.close()

    # (b) Perform 1 step GMM 
    if True:
        def data_moments(xvals):
            mean_data = xvals.mean()
            var_data = xvals.var()
            return mean_data, var_data

        def model_moments(mu, sigma, cutoff):
            mean_model = exp(mu + ((sigma ** 2) / 2))
            var_model = (exp(sigma ** 2) - 1) * exp(2 * mu + sigma ** 2)
            return mean_model, var_model

        def err_vec(xvals, mu, sigma, cutoff, simple):
            mean_data, var_data = data_moments(xvals)
            moms_data = np.array([[mean_data], [var_data]])
            mean_model, var_model = model_moments(mu, sigma, cutoff)
            moms_model = np.array([[mean_model], [var_model]])
            if simple:
                err_vec = moms_model - moms_data
            else:
                err_vec = (moms_model - moms_data) / moms_data
            return err_vec

        def criterion(params, *args):
            mu, sigma = params
            xvals, cutoff, W = args
            err = err_vec(xvals, mu, sigma, cutoff, simple=False)
            crit_val = np.dot(np.dot(err.T, W), err)
            return crit_val

        params_init = np.array([mu_init, sig_init])
        W_hat = np.eye(2)
        gmm_args = (df['incomes'], np.inf, W_hat)
        results = opt.minimize(criterion, params_init, args=(gmm_args), method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
        mu_GMM1, sig_GMM1 = results.x
        print('1.(b)')
        print('mu_GMM1 =', mu_GMM1, ' sig_GMM1 =', sig_GMM1)
        print('Criterion value_GMM1 =', criterion((mu_GMM1, sig_GMM1), df['incomes'], np.inf, W_hat))
        mean_data, var_data = data_moments(df['incomes'])
        mean_model, var_model = model_moments(mu_GMM1, sig_GMM1, np.inf)
        err1 = err_vec(df['incomes'], mu_GMM1, sig_GMM1, np.inf, False).reshape(2,)
        print('Mean of data =', mean_data, ', Variance of data =', var_data)
        print('Mean of model =', mean_model, ', Variance of model =', var_model)
        print('Difference in mean =', mean_data - mean_model, 'Difference in variance =', var_data - var_model)

        # Plot the histogram of the data
        count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
        plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=20)
        plt.xlabel('Annual income')
        plt.ylabel('Percent of students')
        # Plot the estimated GMM PDF
        dist_pts = np.linspace(0, 160000, 500)
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1, np.inf),
                 linewidth=2, color='k', label='1: $\mu_{GMMb}$,$\sigma_{GMMb}$')
        plt.legend(loc='upper left')
        plt.savefig('1b')
        plt.close()


    # (c) Perform 2 step GMM
    if True:
        err1 = err_vec(df['incomes'], mu_GMM1, sig_GMM1, np.inf, False)
        VCV2 = np.dot(err1, err1.T) / df['incomes'].shape[0]
        W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

        params_init = np.array([mu_init, sig_init])
        gmm_args = (df['incomes'], np.inf, W_hat2)
        results = opt.minimize(criterion, params_init, args=(gmm_args),
                               method='TNC', bounds=((1e-10, None), (1e-10, None)))
        mu_GMM2, sig_GMM2 = results.x
        print('1.(c)')
        print('mu_GMM2 =', mu_GMM2, ' sig_GMM2 =', sig_GMM2)
        print('Criterion value_GMM2 =', criterion((mu_GMM2, sig_GMM2), df['incomes'], np.inf, W_hat2))

        mean_data, var_data = data_moments(df['incomes'])
        mean_model, var_model = model_moments(mu_GMM2, sig_GMM2, np.inf)
        err1 = err_vec(df['incomes'], mu_GMM1, sig_GMM1, np.inf, False).reshape(2,)
        print('Mean of data =', mean_data, ', Variance of data =', var_data)
        print('Mean of model =', mean_model, ', Variance of model =', var_model)
        print('Difference in mean =', mean_data - mean_model, 'Difference in variance =', var_data - var_model)

        # Plot the histogram of the data
        count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
        plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=20)
        plt.xlabel('Annual income')
        plt.ylabel('Percent of students')
        # Plot the estimated GMM PDF
        dist_pts = np.linspace(0, 160000, 500)
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1, np.inf),
                 linewidth=2, color='k', label='1: $\mu_{GMMb}$,$\sigma_{GMMb}$')
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM2, sig_GMM2, np.inf),
                 linewidth=2, color='g', label='2: $\mu_{GMMc}$,$\sigma_{GMMc}$')
        plt.legend(loc='upper left')
        plt.savefig('1c')
        plt.close()

    # (d) Perform 1 step GMM with 3 moments
    if True:
        def data_moments3(xvals):
            bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
            bpct_2_dat = (xvals[(xvals >= 75000) & (xvals < 100000)].shape[0] /
                          xvals.shape[0])
            bpct_3_dat = xvals[xvals >= 100000].shape[0] / xvals.shape[0]
            return bpct_1_dat, bpct_2_dat, bpct_3_dat

        def model_moments3(mu, sigma, cutoff):
            xfx = lambda x: lognorm_pdf(x, mu, sigma, cutoff)
            (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 1e-10, 75000)
            (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
            bpct_3_mod = 1 - bpct_1_mod - bpct_2_mod
            return bpct_1_mod, bpct_2_mod, bpct_3_mod

        def err_vec3(xvals, mu, sigma, cutoff, simple):
            bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
            moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
            bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma, cutoff)
            moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
            if simple:
                err_vec = moms_model - moms_data
            else:
                err_vec = 100 * ((moms_model - moms_data) / moms_data)
            return err_vec

        def criterion3(params, *args):
            mu, sigma = params
            xvals, cutoff, W = args
            err = err_vec3(xvals, mu, sigma, cutoff, simple=False)
            crit_val = np.dot(np.dot(err.T, W), err)
            return crit_val

        params_init = np.array([mu_init, sig_init])
        W_hat = np.eye(3)
        gmm_args = (df['incomes'], np.inf, W_hat)
        results_3 = opt.minimize(criterion3, params_init, args=(gmm_args),
                               method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
        mu_GMM1_3, sig_GMM1_3 = results_3.x
        print('1.(d)')
        print('mu_GMM1_3=', mu_GMM1_3, ' sig_GMM1_3=', sig_GMM1_3)
        print('Criterion value_GMM1_3 =', criterion3((mu_GMM1_3, sig_GMM1_3), df['incomes'], np.inf, W_hat))
        moment1_data, moment2_data, moment3_data = data_moments3(df['incomes'])
        moment1_model, moment2_model, moment3_model = model_moments3(mu_GMM1_3, sig_GMM1_3, np.inf)
        err1 = err_vec3(df['incomes'], mu_GMM1_3, sig_GMM1_3, np.inf, False).reshape(3,)
        print('Moment1 of data =', moment1_data, ', Moment2 of data =', moment2_data, 'Moment3 of data =', moment3_data)
        print('Moment1 of model =', moment1_model, ', Moment2 of model =', moment2_model, 'Moment3 of model =', moment3_model)
        print('Difference in moment1 =', moment1_data - moment1_model, 'Difference in moment2 =', moment2_data - moment2_model, 'Difference in moment3 =', moment3_data - moment3_model)

        # Plot the histogram of the data
        count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
        plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=20)
        plt.xlabel('Annual income')
        plt.ylabel('Percent of students')
        # Plot the estimated GMM PDF
        dist_pts = np.linspace(0, 160000, 500)
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3, np.inf),
                 linewidth=2, color='b', label='1: $\mu_{GMMd}$,$\sigma_{GMMd}$')
        plt.legend(loc='upper left')
        plt.savefig('1d')
        plt.close()


    # (e) Perform 2 step GMM with 3 moments
    if True:
        err1 = err_vec3(df['incomes'], mu_GMM1_3, sig_GMM1_3, np.inf, False)
        VCV2 = np.dot(err1, err1.T) / df['incomes'].shape[0]
        W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

        params_init = np.array([mu_init, sig_init])
        gmm_args = (df['incomes'], np.inf, W_hat2)
        results = opt.minimize(criterion3, params_init, args=(gmm_args),
                               method='TNC',
                               bounds=((1e-10, None), (1e-10, None)))
        mu_GMM2_3, sig_GMM2_3 = results.x
        print('1.(e)')
        print('mu_GMM2_3=', mu_GMM2_3, ' sig_GMM2_3=', sig_GMM2_3)
        print('Criterion value_GMM2_3 =', criterion3((mu_GMM2_3, sig_GMM2_3), df['incomes'], np.inf, W_hat2))

        moment1_data, moment2_data, moment3_data = data_moments3(df['incomes'])
        moment1_model, moment2_model, moment3_model = model_moments3(mu_GMM2_3, sig_GMM2_3, np.inf)
        err1 = err_vec3(df['incomes'], mu_GMM2_3, sig_GMM2_3, np.inf, True).reshape(3,)
        print('Moment1 of data =', moment1_data, ', Moment2 of data =', moment2_data, 'Moment3 of data =', moment3_data)
        print('Moment1 of model =', moment1_model, ', Moment2 of model =', moment2_model, 'Moment3 of model =', moment3_model)
        print('Difference in moment1 =', moment1_data - moment1_model, 'Difference in moment2 =', moment2_data - moment2_model, 'Difference in moment3 =', moment3_data - moment3_model)

        # Plot the histogram of the data
        count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
        plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=20)
        plt.xlabel('Annual income')
        plt.ylabel('Percent of students')
        # Plot the estimated GMM PDF
        dist_pts = np.linspace(0, 160000, 500)
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3, np.inf),
                 linewidth=2, color='b', label='1: $\mu_{GMMd}$,$\sigma_{GMMd}$')
        plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM2_3, sig_GMM2_3, np.inf),
                 linewidth=2, color='r', label='2: $\mu_{GMMe}$,$\sigma_{GMMe}$')
        plt.legend(loc='upper left')
        plt.savefig('1e')
        plt.close()

# Problem 2
if True:
    def data_moments_sick(xdf):
        moments_data = xdf['sick']
        return moments_data

    def model_moments_sick(xdf, b0, b1, b2, b3):
        moments_model = b0 + b1 * xdf['age'] + b2 * xdf['children'] + b3 * xdf['avgtemp_winter']
        return moments_model

    def err_vec_sick(xdf, b0, b1, b2, b3, simple):
        moms_data = data_moments_sick(xdf)
        moms_model = model_moments_sick(xdf, b0, b1, b2, b3)
        if simple:
            err_vec = moms_model - moms_data
        else:
            err_vec = (moms_model - moms_data) / moms_data
        return err_vec

    def criterion_sick(params, *args):
        b0, b1, b2, b3 = params
        xdf, W = args
        err = err_vec_sick(xdf, b0, b1, b2, b3, simple=True)
        crit_val = np.dot(np.dot(err.T, W), err)
        return crit_val

    b0_init, b1_init, b2_init, b3_init = (0, 0, 0, 1)
    params_init = np.array([b0_init, b1_init, b2_init, b3_init])
    W_hat = np.eye(200)
    gmm_args = (df_sick, W_hat)
    results_sick = opt.minimize(criterion_sick, params_init, args=(gmm_args),
                                method='L-BFGS-B')
    b0_GMM1, b1_GMM1, b2_GMM1, b3_GMM1 = results_sick.x
    print('2.(a)')
    print('b0_GMM1=', b0_GMM1, 'b1_GMM1=', b1_GMM1,
          'b2_GMM1=', b2_GMM1, 'b3_GMM1=', b3_GMM1)
    print('criterion_GMM1=',
          criterion_sick((b0_GMM1, b1_GMM1, b2_GMM1, b3_GMM1), df_sick, W_hat))