# Import packages
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import PS3_2_Funcs as FUNC
import os

# Import data using pandas
df = pd.read_csv('sick.txt', skiprows = [0], names = ['sick', 'age', 'children', 'avgtemp_winter'])

# Convert dataframe into numpy array
data_array = df.values

# Initial parameter guess
beta0_init, beta1_init, beta2_init, beta3_init = (-1,-1, 2, 1)
# Assign initial parameters to initial parameter array
params_init = np.array([beta0_init, beta1_init, beta2_init, beta3_init])
# Assign identity matrix as Weighting matrix
W_hat = np.eye(200)
# Assign arguments
gmm_args = (data_array, W_hat)
# Run optimization, using criterion functions and initial values
results = opt.minimize(FUNC.criterion, params_init, args=(gmm_args),
                       method='SLSQP')

# Assign outputs from optimization to variables
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results.x
fun = results.fun
# Print results
print('beta0_GMM=', beta0_GMM, 'beta1_GMM=', beta1_GMM, 'beta2_GMM=', beta2_GMM, 'beta3_GMM=', beta3_GMM)
print('beta0_init=', beta0_init, 'beta1_init=', beta1_init, 'beta2_init=', beta2_init, 'beta3_init=', beta3_init)
print('value of the GMM Criterion Function=', fun)
