# Perspectives on Computational Modeling
# Problem Set 2
# Name: Ningyin Xu

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt


### Problem 1a:
def problem1a(data, num_bins):
    graph = True
    if graph:
        weights = (1 / data.shape[0]) * np.ones_like(data)
        count, bins, ignored = plt.hist(data, num_bins, weights = weights)
        plt.title('MACSS student income: 2018-2020', fontsize=20)
        plt.xlabel('Incomes')
        plt.ylabel('Frequency')
        # plt.show()
        output_path = os.path.join(output_dir, 'Fig_1a')
        plt.savefig(output_path)
        plt.close()


### Problem 1b:
def lognorm_pdf(xvals, mu, sigma):
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))    
    return pdf_vals


def log_lik(pdf_vals):
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val


def problem1b(data, mu, sig, dist_pts):
    pdf_vals = sts.lognorm.pdf(x = dist_pts, s = sig, scale = np.exp(mu))
    # pdf_vals = lognorm_pdf(dist_pts, mu, sig)
    
    pdf_vals1 = sts.lognorm.pdf(x = data, s = sig, scale = np.exp(mu))
    # pdf_vals1 = lognorm_pdf(data, mu, sig)
    

    graph = True
    if graph:
        plt.plot(dist_pts, pdf_vals, linewidth=2, color='r',
             label=r'$\mu$ = 9, $\sigma$ = 0.3')
        plt.legend(loc='upper right')
        # plt.show()
        output_path = os.path.join(output_dir, 'Fig_1b')
        plt.savefig(output_path)
        plt.close()

    return log_lik(pdf_vals1)


### Problem 1c:
def crit(params, args):
    mu, sigma = params
    xvals = args
    log_lik_val = log_lik(lognorm_pdf(xvals, mu, sigma))
    neg_log_lik_val = - log_lik_val

    return neg_log_lik_val


def mle_dist(mu_init, sig_init, data):
    params_init = np.array([mu_init, sig_init])
    mle_args = data
    bounds = ((None, None), (0.000000001, None))
    results = opt.minimize(crit, params_init, mle_args,
                           method = 'SLSQP',
                           bounds = bounds)
    return results


def plot_pdf(mu, sig, data, dist_pts, num_bins):
    graph = True
    if graph:
        #fig, ax1 = plt.subplots()
        #weights = (1 / data.shape[0]) * np.ones_like(data)
        count, bins, ignored = plt.hist(data, num_bins, normed = True)#, weights = weights)
        plt.title('MACSS student income: 2018-2020', fontsize=20)
        plt.xlabel('Incomes')
        plt.ylabel('Frequency')

        pdf_vals_b = sts.lognorm.pdf(x = dist_pts, s = sig, scale = np.exp(mu))

        mu_MLE, sig_MLE = mle_dist(mu, sig, data).x
        pdf_vals_c = sts.lognorm.pdf(x = dist_pts, s = sig_MLE, 
                                     scale = np.exp(mu_MLE))

        plt.plot(dist_pts, pdf_vals_b, linewidth=2, color='r',
             label=r'$\mu$ = 9, $\sigma$ = 0.3')
        plt.plot(dist_pts, pdf_vals_c, linewidth=2, color='g',
             label='$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu_MLE, sig_MLE))

        plt.legend(loc='upper right')
        # plt.show()
        output_path = os.path.join(output_dir, 'Fig_1c')
        plt.savefig(output_path)
        plt.close()


### Problem1d:
def problem1d(log_lik_mle, log_lik_h0):
    LR_val = 2 * (log_lik_mle - log_lik_h0)
    pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
    return pval_h0


if __name__ == "__main__":
    np.seterr(all='ignore')
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    data = np.loadtxt("incomes.txt")
    num_bins = 30
    problem1a(data, num_bins)

    mu = 9.0
    sigma = 0.3
    lowerbnd = 0
    upperbnd = 150000
    num_pts = 10000
    dist_pts = np.linspace(lowerbnd, upperbnd, num_pts)
    print()
    log_lik_h0 = problem1b(data, mu, sigma, dist_pts)
    print("1b. The value of the log likelihood value is : {:.2f}".
          format(log_lik_h0))

    plot_pdf(mu, sigma, data, dist_pts, num_bins)
    results = mle_dist(mu, sigma, data)
    mu_MLE, sig_MLE = results.x
    log_lik_mle = - results.fun
    #vcv_mle = results.hess_inv

    print()
    print("1c. The ML estimates for mu is: {:.2f}, for sigma is: {:.2f}."
          .format(mu_MLE, sig_MLE))
    print()
    print("1c. The maximum value of likelihood function is: {:.2f}."
           .format(log_lik_mle))
    #print()
    #print('VCV(MLE) = ', vcv_mle)

    print()
    pval = problem1d(log_lik_mle, log_lik_h0)
    print("1d. The Chi square of H0 with 2 degrees of freedom p-value = {:.4f}."
          .format(pval), "Since this probability is small, the data is unlikely",
          "coming from the distributioon in part(b)")

    finaldist = sts.lognorm(s = sig_MLE, scale = np.exp(mu_MLE))
    prob_highinc = (1 - finaldist.cdf(100000)) * 100
    prob_lowinc = finaldist.cdf(75000) * 100
    print()
    print("1e. The probability that a student would earn more than $100,000 is: {:.2f}%."
          .format(prob_highinc), "The probability of a student earn less than",
          "$75,000 is: {:.2f}%".format(prob_lowinc))








