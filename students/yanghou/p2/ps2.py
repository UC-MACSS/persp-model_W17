#Yang Hou
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt


def plot_graph(name):
    cur_path=os.path.split(os.path.abspath("__file__"))[0]
    output_fldr="images"
    output_dir=os.path.join(cur_path,output_fldr)
    if not os.access(output_dir,os.F_OK):
        os.makedirs(output_dir)
    output_path=os.path.join(output_dir,name)
    return output_path

# Define function that generates values of a normal pdf
def norm_pdf(xvals, mu, sigma):
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
            
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    
    return pdf_vals

# Define log likelihood function for the normal distribution
def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = sts.lognorm(scale=np.exp(mu), s=sigma).pdf(xvals)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

def log_lik_sick(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = norm_pdf(xvals,mu,sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

def crit(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_lognorm(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

def crit2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    sigma,b0,b1,b2,b3 = params
    sick,age,children,avgtemp_winter = args
    error=sick-b0-b1*age-b2*children-b3*avgtemp_winter
    log_lik_val = log_lik_sick(error, 0, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
#1a
data=np.loadtxt('incomes.txt')
weights=(1/len(data))*np.ones_like(data)
plt.hist(data,bins=30,weights=weights)
plt.title('Histogram of incomes')
plt.xlabel('Annual Incomes ($)')
plt.ylabel('Percentage of Incomes (%)')
output_path=plot_graph('1a')
plt.savefig(output_path)
plt.close()

#1b
dist_pts=np.linspace(0,150000,500)
y=sts.lognorm(s=0.3,scale=np.exp(9.0)).pdf(dist_pts)
plt.plot(dist_pts,y,linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,150000)
plt.title('Lognormal pdf with mu=9.0 sigma=0.3')
output_path=plot_graph('1b')
plt.savefig(output_path)
plt.close()
answer_1b=log_lik_lognorm(data,9.0,0.3)
print('The log likelihood given this data is: ', answer_1b)

#1c
para_list=np.array([11,0.2])
mle_args=(data,data.max())
result=opt.minimize(crit,para_list,args=(mle_args),bounds=((None,None),(0.00000000001,None)))
mu_mle,sig_mle=result.x
answer_1c=log_lik_lognorm(data,mu_mle,sig_mle)
vcv_mle=result.hess_inv.todense()
print("mu_mle=",mu_mle,' sig_mle=',sig_mle)
print("The value of the likelihood function is: ",answer_1c)
print('VCV(MLE) = ',vcv_mle)
output_path=plot_graph('1c')
plt.hist(data,bins=30,normed=True)
y1=sts.lognorm(s=0.3,scale=np.exp(9.0)).pdf(dist_pts)
y2=sts.lognorm(s=sig_mle,scale=np.exp(mu_mle)).pdf(dist_pts)
plt.plot(dist_pts,y1,linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
plt.plot(dist_pts,y2,linewidth=2, color='G', label='2: $\mu$ and $\sigma$ estimated by MLE')
plt.legend(loc='upper right')
plt.xlabel('Annual Income')
plt.ylabel('Percent')
plt.savefig(output_path)
plt.close()

#1d
log_lik_h0=log_lik_lognorm(data,9.0,0.3)
log_lik_mle=log_lik_lognorm(data,mu_mle,sig_mle)
lr_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(lr_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print('so we can reject the null hypothesis')

#1e
answer_e1=sts.lognorm.cdf(75000,s=sig_mle,loc=mu_mle,scale=np.exp(mu_mle))
answer_e2=1-sts.lognorm.cdf(100000,s=sig_mle,loc=mu_mle,scale=np.exp(mu_mle))
print('The probability that I will earn less than 75000 is: ',round(answer_e1,4)*100,'%')
print('The probability that I will earn less than 100000 is: ',round(answer_e2,4)*100,'%')

#2a
sick_data=pd.read_csv('sick.txt',skiprows=1,names={'sick','age','children','avgtemp_winter'})
para_list=np.array([1,0.1,0.1,0.1,0.1])
bound=((0.000001,None),(None,None),(None,None),(None,None),(None,None))
mle_args=(sick_data['sick'],sick_data['age'],sick_data['children'],sick_data['avgtemp_winter'])
results=opt.minimize(crit2,para_list,args=(mle_args),bounds=bound,method='SLSQP')
sig_mle,b0_mle,b1_mle,b2_mle,b3_mle=results.x
para_list=np.array([sig_mle,b0_mle,b1_mle,b2_mle,b3_mle])
results2=opt.minimize(crit2,para_list,args=(mle_args),bounds=bound,method='L-BFGS-B')
vcv_mle=results2.hess_inv.todense()
print('sigma_mle:',sig_mle,' beta0_mle:',b0_mle,' beta1_mle:',b1_mle,' beta2_mle:',b2_mle,' beta3_mle:',b3_mle)
xvals_mle=sick_data['sick']-b0_mle-b1_mle*sick_data['age']-b2_mle*sick_data['children']-b3_mle*sick_data['avgtemp_winter']
answer_2a=answer_2a=log_lik_sick(xvals_mle,0,sig_mle)
print('The value of the maximaized likelihood function is: ',answer_2a)
print('The varibale-covariance matrix is: ',vcv_mle)

#2b
xvals=sick_data['sick']-1
log_lik_h0=log_lik_sick(xvals,0,0.01)
log_lik_mle=log_lik_sick(xvals_mle,0,sig_mle)
lr_val=2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(lr_val, 2)
print('chi squared of H0 with 5 degrees of freedom p-value = ', pval_h0)
print('so we can reject the null hypothesis')