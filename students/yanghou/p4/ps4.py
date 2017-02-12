#Yang Hou
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt
import numpy.linalg as lin
import scipy.integrate as intgr
import matplotlib

def plot_graph(name):
    cur_path=os.path.split(os.path.abspath("__file__"))[0]
    output_fldr="images"
    output_dir=os.path.join(cur_path,output_fldr)
    if not os.access(output_dir,os.F_OK):
        os.makedirs(output_dir)
    output_path=os.path.join(output_dir,name)
    return output_path

def LN_pdf(xvals,mu,sigma):
    pdf_vals=((1/(xvals*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(xvals)-mu)**2/(2*sigma**2))))
    return pdf_vals

def trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a truncated normal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                  cutoff of truncated normal distribution
    cut_lb_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                  cutoff of truncated normal distribution
    unif2_vals  = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  rescaled uniform derived from original.
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma) and cutoffs
                  (cut_lb, cut_ub)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''
    # No cutoffs: truncated normal = normal
    if (cut_lb == None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb != None) & (cut_ub == None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb == None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb != None) & (cut_ub != None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    
    unif2_vals = unif_vals * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    tnorm_draws = sts.norm.ppf(unif2_vals, loc=mu, scale=sigma)
    draws=np.exp(tnorm_draws)
    
    return draws

def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        std_data = xvals.std()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        std_data = xvals.std(axis=0)
    
    return mean_data, std_data

def err_vec(data_vals, sim_vals, mu, sigma, cut_lb, cut_ub, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    moms_model = np.array([[mean_model], [std_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, cut_lb, cut_ub, W_hat = args
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    err = err_vec(xvals, sim_vals, mu, sigma, cut_lb, cut_ub,
                  simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val


#1a
data=np.loadtxt('incomes.txt')
plt.hist(data,bins=30,normed=True)
plt.title('Histogram of incomes')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('Percentage of Incomes (%)')
output_path=plot_graph('1a')
plt.savefig(output_path)
plt.close()

#1b
xvals=np.array([[200.0,270.0],[180.0,195.5]])
print('2b')
print('The test result is: ',LN_pdf(xvals,5.0,1.0))

#1c
np.random.seed(seed=1234)
N=200
S=300
mu_init=11.0
sig_init=0.2
cut_lb=0.0
cut_up=1500000
w_hat=np.eye(2)
unif_vals=sts.uniform.rvs(0,1,size=(N,S))
smm_args=(data,unif_vals,cut_lb,cut_up,w_hat)
para_list=np.array([mu_init,sig_init])
result=opt.minimize(criterion,para_list,args=(smm_args),method='L-BFGS-B',bounds=((1e-10,None),(1e-10,None)))
mu_smm1,sig_smm1=result.x
print('1c')
print('The mu_smm = ',mu_smm1,' and the sig_smm = ',sig_smm1)
mean_data,std_data=data_moments(data)
sim_vals=trunc_norm_draws(unif_vals,mu_smm1,sig_smm1,cut_lb,cut_up)
mean_sim,std_sim=data_moments(sim_vals)
mean_model=mean_sim.mean()
std_model=std_sim.mean()
para_smm=np.array([mu_smm1,sig_smm1])
value=criterion(para_smm,*smm_args)
print('Data mean is: ',mean_data,'Data std is: ',std_data)
print('model mean is: ',mean_model,'model std is: ',std_model)
print('The value of the criterion function is: ',value)
plt.hist(data,bins=30,normed=True)
plt.title('Histogram of incomes')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('Percentage (%)')
dist_pts=np.linspace(0.000001,150000,150000)
plt.plot(dist_pts,LN_pdf(dist_pts,mu_smm1,sig_smm1),label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.legend(loc='upper right')
output_path=plot_graph('1c')
plt.savefig(output_path)
plt.close()

#1d
err=err_vec(data,sim_vals,mu_smm1,sig_smm1,cut_lb,cut_up,False)
vcv2=np.dot(err,err.T)/data.shape[0]
w_hat2=lin.pinv(vcv2)
smm_args=(data,unif_vals,cut_lb,cut_up,w_hat2)
result=opt.minimize(criterion,para_smm,args=(smm_args),method='L-BFGS-B',bounds=((1e-10,None),(1e-10,None)))
mu_smm2,sig_smm2=result.x
print('1d')
print('The mu_smm2 = ',mu_smm2,' the sig_smm2 =',sig_smm2)
sim_vals=trunc_norm_draws(unif_vals,mu_smm2,sig_smm2,cut_lb,cut_up)
mean_sim,std_sim=data_moments(sim_vals)
mean_model=mean_sim.mean()
std_model=std_sim.mean()
para_smm=np.array([mu_smm2,sig_smm2])
value=criterion(para_smm,*smm_args)
print('Data mean is: ',mean_data,'Data std is: ',std_data)
print('model mean is: ',mean_model,'model std is: ',std_model)
print('The value of the criterion function is: ',value)
plt.hist(data,bins=30,normed=True)
plt.title('Histogram of incomes')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('Percentage (%)')
plt.plot(dist_pts,LN_pdf(dist_pts,mu_smm1,sig_smm1),label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.plot(dist_pts,LN_pdf(dist_pts,mu_smm2,sig_smm2),label='1: $\mu_{SMM2}$,$\sigma_{SMM2}$')
plt.legend(loc='upper right')
output_path=plot_graph('1d')
plt.savefig(output_path)
plt.close()