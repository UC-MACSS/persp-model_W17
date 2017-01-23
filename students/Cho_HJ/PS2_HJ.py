'''
---------- ---------- ---------- ---------- ----------
MACS 30100 - Perspectives on Computational Modeling
HyungJin Cho
---------- ---------- ---------- ---------- ----------
'''

# Importing Pacakges

import pandas as pd


import scipy.special as spc
import scipy.optimize as opt


'''
1.(a)
'''
# < Directory >
import os
def directory():
    DIRECTORY = os.path.split(os.path.abspath('__file__'))[0]
    FOLDER = 'images'
    DIRECTORY_FOLDER = os.path.join(DIRECTORY, FOLDER)
    if not os.access(DIRECTORY_FOLDER, os.F_OK):
        os.makedirs(DIRECTORY_FOLDER)
    return(DIRECTORY_FOLDER)

# < Graph >
# %matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

INCOME = np.loadtxt('incomes.txt')
plt.hist(INCOME, bins=30, weights=np.ones_like(INCOME)/len(INCOME)*100)
plt.title('Histogram of Percentages of the Income')
plt.xlabel('Incomes ($)')
plt.ylabel('Income Observations (%)')
plt.xlim(0, 150000)

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(a)'))
plt.close()

'''
1.(b)
'''
# < Parameters >
MU = 9.0
SIGMA = 0.3

# < Defining Function >
# --------------------------------------------------
# FUNCTION_1: Defining a function that generates lognormal Probability Density Function
# INPUT_1: x values
# INPUT_2: mu
# INPUT_3: sigma
# OUTPUT: pdf values
# --------------------------------------------------
import scipy as sp
import scipy.stats as sts
import numpy as np

def FUNCTION_1(INPUT_1, INPUT_2, INPUT_3):
    OUTPUT = sts.lognorm.pdf(INPUT_1, scale=np.exp(INPUT_2), s=INPUT_3)
    return OUTPUT

# < Graph >
# %matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.linspace(0, 150000, 1500000), FUNCTION_1(np.linspace(0, 150000, 1500000), MU, SIGMA))
plt.xlim(0, 150000)
plt.title('lognormal Probability Density Function ($\mu=3$,$\sigma=0.3$)')
plt.xlabel('x = Income($)')
plt.ylabel('$f(x)$')

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(b)'))
plt.close()

# < Defining Function >
# --------------------------------------------------
# FUNCTION_2: Defining a function that computes log likelihood value
# INPUT_1: x values
# INPUT_2: mu
# INPUT_3: sigma
# OUTPUT: log likelihood value
# --------------------------------------------------
import numpy as np

def FUNCTION_2(INPUT_1, INPUT_2, INPUT_3):
    OUTPUT = np.log(FUNCTION_1(INPUT_1, INPUT_2, INPUT_3)).sum()
    return OUTPUT

print('Log likelihood value = ', FUNCTION_2(INCOME, MU, SIGMA))

'''
1.(c)
'''
# < Defining Function >
# --------------------------------------------------
# FUNCTION_3: Defining a function that computes negative log likelihood value
# INPUT_1: (mu, sigma)
# INPUT_2: (x values)
# OUTPUT: negative log likelihood value
# --------------------------------------------------
def FUNCTION_3(INPUT_1, INPUT_2):
    INPUT_1_1, INPUT_1_2 = INPUT_1
    OUTPUT = -FUNCTION_2(INPUT_2, INPUT_1_1, INPUT_1_2)
    return OUTPUT
# print(FUNCTION_3((MU, SIGMA), INCOME))

# < Defining Function >
# --------------------------------------------------
# FUNCTION_4: Defining a function that computes Maximum Likelihood Estimation
# INPUT_1: initial mu
# INPUT_2: initial sigma
# INPUT_3: values
# INPUT_4: method
# OUTPUT: MLE mu, MLE sigma
# --------------------------------------------------
import scipy as sp
import scipy.optimize as opt
import numpy as np

def FUNCTION_4(INPUT_1, INPUT_2, INPUT_3, INPUT_4='L-BFGS-B'):
    PROCESS_1_1, PROCESS_2_1 = opt.minimize(FUNCTION_3, np.array([INPUT_1, INPUT_2]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method='SLSQP').x
    PROCESS_1_2, PROCESS_2_2 = opt.minimize(FUNCTION_3, np.array([PROCESS_1_1, PROCESS_2_1]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method=INPUT_4).x
    OUTPUT_1 = opt.minimize(FUNCTION_3, np.array([PROCESS_1_2, PROCESS_2_2]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method=INPUT_4)
    OUTPUT_2, OUTPUT_3 = OUTPUT_1.x
    return(OUTPUT_1, OUTPUT_2, OUTPUT_3)
MLE, MU_MLE, SIGMA_MLE = FUNCTION_4(MU, SIGMA, INCOME)

print('mu_MLE = ', MU_MLE, 'sigma_MLE = ', SIGMA_MLE)
print('log-likelihood Value', FUNCTION_2(INCOME, MU_MLE, SIGMA_MLE))
print('variance-covariance matrix = ', MLE.hess_inv.todense())

# < Graph >
# %matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np

plt.hist(INCOME, bins=30, normed=True)
plt.plot(np.linspace(0, 150000, 1500000), FUNCTION_1(np.linspace(0, 150000, 1500000), MU, SIGMA), label='$\mu$=9.0, $\sigma$=0.3')
plt.plot(np.linspace(0, 150000, 1500000), FUNCTION_1(np.linspace(0, 150000, 1500000), MU_MLE, SIGMA_MLE), label='$\mu$=11.33, $\sigma$=0.21')
plt.xlim(0, 150000)
plt.title('Histogram of Percentages of the Income and lognormal Probability Density Function')
plt.xlabel('x = Income($)')
plt.ylabel('$f(x)$')
plt.legend(loc='upper right')

# plt.show()
plt.savefig(os.path.join(directory(), 'Figure_1(c)'))
plt.close()

'''
1.(d)
'''
import scipy as sp
import scipy.stats as sts
LIKELIHOOD_RATIO = 2 * (FUNCTION_2(INCOME, MU_MLE, SIGMA_MLE) - FUNCTION_2(INCOME, MU, SIGMA))
CHISQUARE_P = 1.0 - sts.chi2.cdf(LIKELIHOOD_RATIO, 2)

print('Likelihood Ratio Test p-value = ', CHISQUARE_P)

'''
1.(e)
'''
import scipy as sp

INCOME_100000 = 1 - sp.integrate.quad(lambda x: FUNCTION_1(x, MU_MLE, SIGMA_MLE), 0, 100000)[0]
INCOME_75000 = sp.integrate.quad(lambda x: FUNCTION_1(x, MU_MLE, SIGMA_MLE), 0, 75000)[0]

print('Probability of Income > $100,000 = ', INCOME_100000)
print('Probability of Income < $75,000 = ', INCOME_75000)

'''
2.(a)
'''
SICK = pd.read_csv('sick.txt')

# < Defining Function >
# --------------------------------------------------
# FUNCTION_5: Defining a function that generates Probability Density Function values
# INPUT_1: data
# INPUT_2: b0
# INPUT_3: b1
# INPUT_4: b2
# INPUT_5: b3
# INPUT_6: sigma
# OUTPUT: PDF values
# --------------------------------------------------
def FUNCTION_5(INPUT_1, INPUT_2, INPUT_3, INPUT_4, INPUT_5, INPUT_6):
    OUTPUT = sts.norm.pdf((INPUT_1[0] - INPUT_2 - INPUT_3*INPUT_1[1] - INPUT_4*INPUT_1[2] - INPUT_5*INPUT_1[3]), loc=0, scale=INPUT_6)
    return(OUTPUT)

# < Defining Function >
# --------------------------------------------------
# FUNCTION_6: Defining a function that computes log likelihood value
# INPUT_1: data
# INPUT_2: ([b0, b1, b2, b3, sigma])
# OUTPUT: log likelihood value
# --------------------------------------------------
import numpy as np

def FUNCTION_6(INPUT_1, INPUT_2):
    OUTPUT = np.log(FUNCTION_5(INPUT_1, INPUT_2[0], INPUT_2[1], INPUT_2[2], INPUT_2[3], INPUT_2[4])).sum()
    return(OUTPUT)

# < Defining Function >
# --------------------------------------------------
# FUNCTION_7: Defining a function that computes negative log likelihood value
# INPUT_1: (data)
# INPUT_2: ([b0, b1, b2, b3, sigma])
# OUTPUT: negative log likelihood value
# --------------------------------------------------
def FUNCTION_7(INPUT_1, INPUT_2):
    OUTPUT = -FUNCTION_6(INPUT_2, INPUT_1)
    return(OUTPUT)

# < Defining Function >
# --------------------------------------------------
# FUNCTION_8: Defining a function that computes Maximum Likelihood Estimation
# INPUT_1: b0
# INPUT_2: b1
# INPUT_3: b2
# INPUT_4: b3
# INPUT_5: sigma
# OUTPUT_1: MLE
# OUTPUT_2: b0_MLE
# OUTPUT_3: b1_MLE
# OUTPUT_4: b2_MLE
# OUTPUT_5: b3_MLE
# OUTPUT_6: sigma_MLE
# --------------------------------------------------
import scipy as sp
import scipy.optimize as opt
import numpy as np

def FUNCTION_8(INPUT_1, INPUT_2, INPUT_3, INPUT_4, INPUT_5):
    PROCESS_1, PROCESS_2, PROCESS_3, PROCESS_4, PROCESS_5 = opt.minimize(FUNCTION_7, ([INPUT_1, INPUT_2, INPUT_3, INPUT_4, INPUT_5]), args=([SICK[SICK.columns[0]], SICK[SICK.columns[1]], SICK[SICK.columns[2]], SICK[SICK.columns[3]]]), bounds=((None,None),(None,None),(None,None),(None,None),(1e-10,None)), method='SLSQP').x
    OUTPUT_1 = opt.minimize(FUNCTION_7, ([PROCESS_1, PROCESS_2, PROCESS_3, PROCESS_4, PROCESS_5]), args=([SICK[SICK.columns[0]], SICK[SICK.columns[1]], SICK[SICK.columns[2]], SICK[SICK.columns[3]]]), bounds=((None,None),(None,None),(None,None),(None,None),(1e-10,None)), method='L-BFGS-B')
    OUTPUT_2, OUTPUT_3, OUTPUT_4, OUTPUT_5, OUTPUT_6 = OUTPUT_1.x
    return(OUTPUT_1, OUTPUT_2, OUTPUT_3, OUTPUT_4, OUTPUT_5, OUTPUT_6)
MLE, b0_MLE, b1_MLE, b2_MLE, b3_MLE, SIGMA_MLE = FUNCTION_8(INPUT_1=1.0, INPUT_2=0.0, INPUT_3=0.0, INPUT_4=0.0, INPUT_5=np.sqrt(0.01))


print('b0_MLE = ', b0_MLE, 'b1_MLE = ', b1_MLE, 'b2_MLE = ', b2_MLE, 'b3_MLE = ', b3_MLE, 'sigma_MLE = ', SIGMA_MLE)
print('log-likelihood Value', -1 * MLE.fun)
print('variance-covariance matrix = ', MLE.hess_inv.todense())

'''
2.(b)
'''
import scipy as sp
import scipy.stats as sts

PARAMETER_MLE = (b0_MLE, b1_MLE, b2_MLE, b3_MLE, SIGMA_MLE)
PARAMETER = (1.0, 0.0, 0.0, 0.0, np.sqrt(0.01))
LIKELIHOOD_RATIO = 2 * (FUNCTION_6(SICK, PARAMETER_MLE) - FUNCTION_2(INCOME, PARAMETER))
CHISQUARE_P = 1.0 - sts.chi2.cdf(LIKELIHOOD_RATIO, 2)

print('Likelihood Ratio Test p-value = ', CHISQUARE_P)
