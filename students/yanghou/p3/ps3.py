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

def lognorm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
            
    pdf_vals    = (1/(xvals*sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))
    
    return pdf_vals

def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    var_data  = scalar > 0, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()
    
    return mean_data, std_data

def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, var_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0,150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0,150000)
    std_model=np.sqrt(var_model)
    
    return mean_model, std_model


def err_vec(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [std_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
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
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for GMM
    (binpct_1, binpct_2, binpct_3, binpct_4).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 220
    bpct_2_dat = scalar in [0, 1], percent of observations
                 220 <= x < 320
    bpct_3_dat = scalar in [0, 1], percent of observations
                 320 <= x < 430
    

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3, bpct_4
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[xvals >=100000].shape[0] /
                  xvals.shape[0])

    return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3, binpct_4).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
        xfx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_mod = scalar in [0, 1], percent of model observations in
                 bin 1
    bp_1_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_1_mod
    bpct_2_mod = scalar in [0, 1], percent of model observations in
                 bin 2
    bp_2_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_2_mod
    bpct_3_mod = scalar in [0, 1], percent of model observations in
                 bin 3
    bp_3_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_3_mod

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod, bpct_4_mod
    --------------------------------------------------------------------
    '''
    xfx = lambda x: lognorm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments4()
        model_moments4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    bpct_1_dat, bpct_2_dat, bpct_3_dat= \
        data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod= \
        model_moments3(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec


def criterion3(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
   
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (4, 1) matrix, column vector of four moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec3(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

def err_vec_sick(data,b0,b1,b2,b3):
    moms_data=data['sick']
    moms_model=b0+b1*data['age']+b2*data['children']+b3*data['avgtemp_winter']
    err_vec=np.array(moms_model-moms_data)
    return err_vec

def criterion_sick(params,*args):
    data,w=args
    b0,b1,b2,b3=params
    err=err_vec_sick(data,b0,b1,b2,b3)
    crit_val=np.dot(np.dot(err.T,w),err)
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
mu_init=9.0
std_init=0.3
para_list=np.array([mu_init,std_init])
w_hat=np.eye(2)
gmm_args=(data,w_hat)
results=opt.minimize(criterion,para_list,args=(gmm_args),method='L-BFGS-B',bounds=((None,None),(1e-10,None)))
mu_gmm1,sig_gmm1=results.x
print('1b')
print('mu_GMM1=',mu_gmm1,' sig_GMM1=',sig_gmm1)
para_gmm=np.array([mu_gmm1,sig_gmm1])
value=criterion(para_gmm,*gmm_args)
mu_data,std_data=data_moments(data)
mu_model,std_model=model_moments(mu_gmm1,sig_gmm1)
print('The value of the criterion function is: ',value)
print('Data moments are: mean:',mu_data,' std:',std_data)
print('Model moments are: mean:',mu_model,' std:',std_model)
output_path=plot_graph('1b')
plt.hist(data,bins=30,normed=True)
plt.title('Lognormal PDF')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('PDF value')
dist_pts=np.linspace(0,150000,10000)
y=lognorm_pdf(dist_pts,mu_gmm1,sig_gmm1)
plt.plot(dist_pts,y,label='1: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm1,sig_gmm1))
plt.legend(loc='upper right')
plt.savefig(output_path)
plt.close()

#1c
err1=err_vec(data,mu_gmm1,sig_gmm1,False)
vcv2=np.dot(err1,err1.T)/data.shape[0]
w_hat2=lin.pinv(vcv2)
gmm_args=(data,w_hat2)
results=opt.minimize(criterion,para_list,args=(gmm_args),method='L-BFGS-B',bounds=((None,None),(1e-10,None)))
mu_gmm2,sig_gmm2=results.x
print('1c')
print('mu_GMM2=',mu_gmm2,' sig_GMM2=',sig_gmm2)
para_gmm=np.array([mu_gmm2,sig_gmm2])
value=criterion(para_gmm,*gmm_args)
mu_data,std_data=data_moments(data)
mu_model2,std_model2=model_moments(mu_gmm2,sig_gmm2)
print('The value of the criterion function is: ',value)
print('Data moments are: mean:',mu_data,' std:',std_data)
print('Model moments are: mean:',mu_model2,' std:',std_model2)
output_path=plot_graph('1c')
plt.hist(data,bins=30,normed=True)
plt.title('Lognormal PDF')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('PDF value')
y1=lognorm_pdf(dist_pts,mu_gmm1,sig_gmm1)
plt.plot(dist_pts,y1,label='1: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm1,sig_gmm1))
y2=lognorm_pdf(dist_pts,mu_gmm2,sig_gmm2)
plt.plot(dist_pts,y2,label='2: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm2,sig_gmm2))
plt.legend(loc='upper right')
plt.savefig(output_path)
plt.close()

#1d
mu_init=11
std_init=0.2
para_list=np.array([mu_init,std_init])
w_hat=np.eye(3)
gmm_args=(data,w_hat)
results=opt.minimize(criterion3,para_list,args=(gmm_args),method='L-BFGS-B',bounds=((None,None),(1e-10,None)))
mu_gmm3,sig_gmm3=results.x
print('1d')
print('mu_GMM3=',mu_gmm3,' sig_GMM3=',sig_gmm3)
para_gmm=np.array([mu_gmm3,sig_gmm3])
value=criterion3(para_gmm,*gmm_args)
m1_d,m2_d,m3_d=data_moments3(data)
m1_m,m2_m,m3_m=model_moments3(mu_gmm3,sig_gmm3)
print('The value of the criterion function is: ',value)
print('Data moments are: ',m1_d,m2_d,m3_d)
print('Model moments are: ',m1_m,m2_m,m3_m)
output_path=plot_graph('1d')
plt.hist(data,bins=30,normed=True)
plt.title('Lognormal PDF')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('PDF value')
y=lognorm_pdf(dist_pts,mu_gmm3,sig_gmm3)
plt.plot(dist_pts,y,label='1: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm3,sig_gmm3))
plt.legend(loc='upper right')
plt.savefig(output_path)
plt.close()

#1e
err3=err_vec3(data,mu_gmm3,sig_gmm3,False)
vcv2=np.dot(err3,err3.T)/data.shape[0]
w_hat2=lin.pinv(vcv2)
gmm_args=(data,w_hat2)
results=opt.minimize(criterion3,para_list,args=(gmm_args),method='TNC',bounds=((None,None),(1e-10,None)))
mu_gmm4,sig_gmm4=results.x
print('1e')
print('mu_GMM4=',mu_gmm4,' sig_GMM4=',sig_gmm4)
para_gmm=np.array([mu_gmm4,sig_gmm4])
value=criterion3(para_gmm,*gmm_args)
m1_d,m2_d,m3_d=data_moments3(data)
m1_m,m2_m,m3_m=model_moments3(mu_gmm4,sig_gmm4)
print('The value of the criterion function is: ',value)
print('Data moments are: ',m1_d,m2_d,m3_d)
print('Model moments are: ',m1_m,m2_m,m3_m)
output_path=plot_graph('1e')
plt.hist(data,bins=30,normed=True)
plt.title('Lognormal PDF')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('PDF value')
y1=lognorm_pdf(dist_pts,mu_gmm3,sig_gmm3)
plt.plot(dist_pts,y1,label='1: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm3,sig_gmm3))
y2=lognorm_pdf(dist_pts,mu_gmm4,sig_gmm4)
plt.plot(dist_pts,y2,label='1: $\mu$={:.4f},$\sigma$={:.4f}'.format(mu_gmm4,sig_gmm4))
plt.legend(loc='upper right')
plt.savefig(output_path)
plt.close()

#2
sick_data=pd.read_csv('sick.txt',skiprows=1,names={'sick','age','children','avgtemp_winter'})
para_list=np.array([0,0,0,0])
w=np.eye(200)
gmm_args=(sick_data,w)
results=opt.minimize(criterion_sick,para_list,args=(gmm_args),method='L-BFGS-B')
b0_gmm,b1_gmm,b2_gmm,b3_gmm=results.x
print('2')
print('b0:',b0_gmm,' b1:',b1_gmm,' b2:',b2_gmm,' b3:',b3_gmm)
para_gmm=np.array([b0_gmm,b1_gmm,b2_gmm,b3_gmm])
value=criterion_sick(para_gmm,*gmm_args)
print('The value of the criterion function is: ',value)