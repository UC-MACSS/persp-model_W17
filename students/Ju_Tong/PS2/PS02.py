import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# set the seed
np.random.seed(1234)

...
graph = True
...
if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    year_vec    = (lf_years,) vector, years from beg_year to
                  beg_year + lf_years
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)


'''
------------------------------------------------------------------------
Exercise 1a: Simulate the data
------------------------------------------------------------------------
plot_1a     = Boolean, =True if make a plot of one series of the
              simulated income data
norm_errors = (lf_years, num_draws) matrix, normally distributed errors
              with mean 0 and standard deviation sigma
------------------------------------------------------------------------
'''
plot_1a = True

#set up  parameters
mu, sigma = 0, 0.1 # mean and standard deviation
inc0=80000.
rou=0.2
g=0.03
lf_years=40
num_draws=10000
year_vec=list(range(2018,2058))
# generate the normal distribution
norm_error = np.random.normal(mu, sigma, (lf_years, num_draws))
# create the array to collect income results.
ln_inc=np.zeros((lf_years, num_draws))

# simulation for the first year`s ln_income
ln_inc[0,:]=np.log(inc0)+norm_error[0,:]
# simulation for the following years` ln_income
for yr in range(1,40):
    ln_inc[yr,:]=(1-rou)*(np.log(inc0)+g*yr)+rou*ln_inc[yr-1,:]+norm_error[yr,:]

inc=np.exp(ln_inc)

# Plot one lifetime income series from set of simulations
x_vals = year_vec
y_vals = inc[:, 0]
fig, ax = plt.subplots()
plt.plot(x_vals, y_vals)

# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('One simulated lifetime income path', fontsize=15)
plt.xlabel(r'Year $t$')
plt.ylabel(r'Annual income (\$s)')
# plt.xlim((xmin, xmax))
# plt.ylim((ymin, ymax))
# plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'Fig_1a')
plt.savefig(output_path)
# plt.show()
plt.close()

'''
------------------------------------------------------------------------
Exercise 1b: histogram plot for the first year income
------------------------------------------------------------------------
plot_1b     = Boolean, =True if make a plot of first year income
------------------------------------------------------------------------
'''
plot_1b = True

# extract the first year`s income
data=inc[0, :]

# iterate the first year`s income, to calculate the percentage
percent_lower=(sum(i< 70000 for i in data)/num_draws)*100
percent_higher=(sum(i>100000 for i in data)/num_draws)*100

print('1b. Percent of students getting more than $100k in first period: ',
      percent_higher, '%')
print('1b. Percent of students getting less than $70k in first period: ',
      percent_lower, '%')

fig, ax = plt.subplots()
hist_wgts = (1 / num_draws) * np.ones(num_draws)
num_bins = 50
plt.hist(data, num_bins, weights=hist_wgts)
plt.title('Histogram of first year ($t$=2018) income', fontsize=15)
plt.xlabel(r'Annual income (\$s)')
plt.ylabel(r'Percent of students')

output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
# plt.show()
plt.close()


'''
------------------------------------------------------------------------
Exercise 1c: Simulation for years when being able to pay off the debts
------------------------------------------------------------------------
plot_1c     = Boolean, =True if make a plot of years when students are
able to pay off the debts
------------------------------------------------------------------------
'''
plot_1c = True

# set up the amount of debt to pay
debt=95000
# sum of the 0.1 times inc from year 1-40
payment=np.cumsum(0.1*inc, axis=0)
# convert the array of payment to a Bool array (TRUE when payment>debt)
Bool=payment>debt
# using np.argmax() to extract TURE from the array
# np.argmax only return the index, so +1
result=np.argmax(Bool, axis=0)+1

percent_ten_year=(sum(i <=10 for i in result)/num_draws)*100

print("1c. There will be ",percent_ten_year, "%"," of MACSS studetns to pay off loans in 10 years." )

# Histogram  Plotting
fig, ax = plt.subplots()
hist_wgts = (1 / num_draws) * np.ones(num_draws)
plt.hist(result, len(np.unique(result)), weights=hist_wgts)
plt.title('Histogram of years when students can pay off the debt: $\mu2018=80000$, $\sigma=0.1$', fontsize=12)
plt.xlabel(r'year')
plt.ylabel(r'Percent of students')
plt.xlim((8, 14))

output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
# plt.show()
plt.close()



'''
------------------------------------------------------------------------
Exercise 1d: Change ininitial income and sigma: repeat simulation for 1c
------------------------------------------------------------------------
plot_1d     = Boolean, =True if make a plot of years when students are
able to pay off the debts
------------------------------------------------------------------------
'''
plot_1d = True

# reset the parameters
sigma = 0.15 # mean and standard deviation
inc0=85000.
# generate the normal distribution of error again
norm_error = np.random.normal(mu, sigma, (lf_years, num_draws))


# repeat simulaiton in 1.a

# simulation for the first year`s ln_income
ln_inc[0,:]=np.log(inc0)+norm_error[0,:]
# simulation for the following years` ln_income
for yr in range(1,40):
    ln_inc[yr,:]=(1-rou)*(np.log(inc0)+g*yr)+rou*ln_inc[yr-1,:]+norm_error[yr,:]
# logistic transform of the data
inc=np.exp(ln_inc)

# repeat the calculation in 1.c
payment=np.cumsum(0.1*inc, axis=0)
Bool=payment>debt
result=np.argmax(Bool, axis=0)+1
percent_ten_year=(sum(i <=10 for i in result)/num_draws)*100

print("1d. There will be ",percent_ten_year, "%"," of MACSS studetns to pay off loans in 10 years." )


# Histogram plot
fig, ax = plt.subplots()
hist_wgts = (1 / num_draws) * np.ones(num_draws)
plt.hist(result, len(np.unique(result)), weights=hist_wgts)
plt.title('Histogram of years when students can pay off the debt: $\mu2018=85000$, $\sigma=0.15$', fontsize=12)
plt.xlabel(r'year')
plt.ylabel(r'Percent of students')
plt.xlim((8, 14))

output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
# plt.show()
plt.close()
