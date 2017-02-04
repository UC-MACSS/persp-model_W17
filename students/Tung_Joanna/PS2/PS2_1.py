import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import os

'''
THIS SCRIPT IS USED TO CREATE THE PLOTs FOR PS2. THE FULL CODE USED TO GENERATE THE RESPONSES
ON THE PDF CREATED THROUGH LATEX IS PROVIDED IN THE JUPYTER NOTEBOOKS.
'''

# Load data
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

    weights = (1 / pts.shape[0]) * np.ones_like(pts) * 100
    n, bin_cuts, patches = plt.hist(pts, 30, weights=weights)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('Percent of Graduates')

    output_path = os.path.join(output_dir, 'PS2_1a')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# PART 1b: Write lognormal pdf value generator function and save historgram of data with lognormal PDF

def lognorm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.lognorm.pdf(x, loc=mu, scale=exp(mu), s = sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)

    pdf_vals = ((1/(sigma * xvals * np.sqrt(2 * np.pi)) *
                np.exp( -(np.log(xvals) - mu)**2 / (2 * sigma**2))) /
                prob_notcut)

    return pdf_vals

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

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    dist_pts = np.linspace(0, 150000, 500)
    mu_init = 9
    sig_init = 0.3
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_init, sig_init, 'None'),
             linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
    plt.legend(loc='upper right')

    output_path = os.path.join(output_dir, 'PS2_1b')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# Part 1c: Add MLE estimates to the plot from part 1b

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

    count, bins, ignored = plt.hist(pts, 30, normed = True)

    plt.title('Annual Incomes for MACSS 2018-2020 Graduates', fontsize=20)
    plt.xlabel('Annual Salary ($)')
    plt.ylabel('# of Graduates, Normalized')

    dist_pts = np.linspace(0, 150000, 500)
    mu_init = 9
    sig_init = 0.3
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_init, sig_init, 'None'),
             linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
    plt.legend(loc='upper right')

    mu_MLE = 11.3314417737
    sig_MLE = 0.211676166182
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_MLE, sig_MLE, 'None'),
             linewidth=2, color='g', label='2: $\mu$=11.3314417737,$\sigma$=0.211676166182')
    plt.legend(loc='upper right')

    output_path = os.path.join(output_dir, 'PS2_1c')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
