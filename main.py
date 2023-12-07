"""
Main file for running code
"""

import numpy as np
import matplotlib.pyplot as plt

import function as fc
import load_data as ld
import minimiser as mi
import uncertainty as un
import plot as pl_func

plot = False

# load data
data = ld.LoadData()
data_osc, data_unosc, energy, width = data.get_data()  # energy (GeV) in each bin

if plot:
    pl_func.data(energy, data_osc, data_unosc, width)

pl_func.data_aligned(energy, data_osc, data_unosc, width, plot=False)

# calculate prob and plot
pl_func.neutrino_prob_sing(energy, plot=False)

if plot:
    pl_func.neutrino_prob_mul(energy)

# apply prob to unoscillated data
prob = fc.neutrino_prob(E=energy)
data_unosc_prob = data_unosc*prob

if plot:
    pl_func.data_with_prob(
        energy,
        data_unosc_prob, data_unosc, data_osc,
        width
        )

# NLL against theta
min_func = mi.Minimiser()

if plot:
    num = 100  # number of values to generate
    theta_max = np.pi/2  # full range, whould have two minima
    theta_min = 0

    pl_func.nll_1d_theta(min_func, theta_min, theta_max, num, dm2=2.4)

'''
1D parabolic minimisor
'''
print('-'*30)
print('--- 1D parabolic minimiser ---\n')

print('dm2 = 2.4 e-3 used\n')

# data lists
num = 100  # number of values to generate
theta_max = np.pi/2  # full range, whould have two minima
theta_min = 0
theta_list = np.linspace(theta_min, theta_max, num)  # list of theta values

nll_list = []
for theta in theta_list:  # calculate NLL for each theta
    nll = min_func.cal_nll(theta)
    nll_list += [nll]

nll_list = np.array(nll_list)

# parabolic minimiser
theta_min, nll_min, err_list = min_func.parabolic_1d(
    x_list=theta_list, y_list=nll_list, stop_cond=1e-10)

print('where x is theta and y is NLL')
print('\nUse 2e-06 or below cannot converge after 100 iterations')

print(f'\ntheta should be lower than pi/4 = {(np.pi/4):.4f} which is the local maxima')

if plot:
    pl_func.change_nll(err_list, label=r"$\theta_{23}$")


'''
Finding the accuracy using one std
'''
print('\n-- Uncertainty of theta_min --')

print('\n--- Use NLL changes by +/- 1 in absolute units')

# values
theta_plus = np.pi/4  # upper bound for theta_min
theta_minus = theta_min - theta_min/10  # lower bound for theta_min
nll_change = nll_min + 1  # upper and lower bound for NLL


print('\n---- Select the closest value from a list')

theta_stdup, theta_stddown = un.list(
    min_func,
    theta_min, theta_plus, theta_minus,
    nll_change)

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n---- Use secant method to solve theta giving expected NLL')

theta_stdup, nll_stdup = un.secant(
    min_func=min_func,
    x1=theta_min, x2=theta_plus,
    sol_true=nll_change
    ) 
theta_stddown, nll_stddown = un.secant(
    min_func=min_func,
    x1=theta_min, x2=theta_minus,
    sol_true=nll_change
    ) 

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n---- Use Hessian and thus covariance matrix')

hes = un.hessian(func=min_func.cal_nll, x=np.array([theta_min]))

hes_inv = np.linalg.inv(hes)  # inverse hessian to get covariance
sig = np.sqrt(2) * np.sqrt(hes_inv)[0][0]  # std is sqrt of covariance diagonal

print(f'\ntheta_min = {theta_min:.4f} +/- {sig:.4f}')
print(f'with % error +/- {(sig*100/theta_min):.2f}')


'''
Univariate method
'''
print()
print('-'*25)
print('--- Univariate method ---\n')

# plot a big 2D diagram
if plot:
    N = 100  # total number per dimension
    theta_list = np.linspace(0, np.pi/2, N)
    dm2_list = np.linspace(0, 40, N)

    pl_func.nll_2d_theta_dm2(min_func, N, theta_list, dm2_list)


# plot NLL against dm2
if plot:
    num = 1000  # number of values to generate
    dm2_max = 100
    dm2_min = 0

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min
            )

if plot:
    num = 100  # number of values to generate
    dm2_max = 2.7
    dm2_min = 2.1

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min
            )


# minimise and plot a small 2D diagram
N = 100  # total number per dimension

# zoomed on four minima
dm2_high = 2.7
dm2_low = 2.1
theta_high = 0.95
theta_low = 0.62

# zoomed on two minima
# dm2_high = 2.4
# dm2_low = 2.34
# theta_high = 0.845
# theta_low = 0.725

# univariate
theta_guess = 0.65
dm2_guess = 2.25

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot,
 theta_all, dm2_all) = min_func.univariate(
    theta_guess, dm2_guess,
    gue_step_the=50, gue_step_dm2=150,
    num_max=100, stop_cond=1e-10
    )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if plot:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            theta_all, dm2_all,
            plot_points=False
            )

if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$"
        )


'''
Newton's method
'''
print()
print('-'*23)
print('--- Newton\'s method ---\n')

# Newtons
theta_guess = 0.65
dm2_guess = 2.25

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = min_func.Newtons(
     theta_guess, dm2_guess,
     num_max=100, stop_cond=1e-10
     )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if plot:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            )

if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$"
        )


'''
Quasi-Newton method
'''
print()
print('-'*27)
print('--- Quasi-Newton method ---\n')

# Quasi-Newton
theta_guess = 0.65
dm2_guess = 2.35

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = min_func.quasi_Newton(
     theta_guess, dm2_guess,
     alpha=1e-5,
     num_max=100, stop_cond=1e-10
     )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if plot:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            )

if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$"
        )


'''
Gradient descent
'''
print()
print('-'*24)
print('--- Gradient descent ---\n')

# gradient descent
theta_guess = 0.65
dm2_guess = 2.35

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = min_func.gradient_descent(
     theta_guess, dm2_guess,
     alpha=5e-5,
     num_max=100, stop_cond=1e-10
     )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if plot:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            )

if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$"
        )


'''
Monte-Carlo method
'''
print()
print('-'*26)
print('--- Monte-Carlo method ---')

print('\n- Classical simulated annealing -\n')

# inital guess same as before
theta_guess = 0.65
dm2_guess = 2.35
T0 = 25
step = 7.5e-3
rho = 0.2
stop_cond = 5e-4

# initial guess to show it can find the global maxima
# theta_guess = 0.75
# dm2_guess = 2.6
# T0 = 80
# step = 1e-2
# rho = 0.25
# stop_cond = 1e-4

# Monte-Carlo classical simulated annealing
(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = min_func.Monte_Carlo(
            theta_guess, dm2_guess,
            T0, step, rho,
            num_max=5000, stop_cond=stop_cond,
            method='CSA'
            )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if plot:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            )

if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$",
        stop=stop_cond
        )


print('\n- Fast simulated annealing -\n')

# inital guess same as before
theta_guess = 0.65
dm2_guess = 2.35
T0 = 25
step = 5e-4
stop_cond = 5e-6

# initial guess to show it can find the global maxima
# theta_guess = 0.75
# dm2_guess = 2.6
# T0 = 80
# step = 2.5e-4
# stop_cond = 5e-6

# Monte-Carlo fast simulated annealing
(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = min_func.Monte_Carlo(
            theta_guess, dm2_guess,
            T0, step,
            num_max=5000, stop_cond=stop_cond,
            method='FSA'
            )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)

# plot
if True:
    pl_func.visual_one_method(
            min_func,
            N,
            theta_low, theta_high,
            dm2_low, dm2_high,
            theta_min, dm2_min,
            theta_plot, dm2_plot,
            )

if True:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$",
        stop=stop_cond
        )


# read number
var_list = theta_plot
var = r"$\theta_{23}$"
num = 10

def Gaussian(x, A, B, C): 
    return A * np.exp(-(x-B)**2 / (2 * (C**2)))

from scipy.optimize import curve_fit


number, edges = np.histogram(var_list, bins=50)
uncer = np.sqrt(number)/number.sum()  # uncertainty
num = number/number.sum()  # normalised number
bincenters = 0.5 * (edges[1:] + edges[:-1])
widths = edges[1:] - edges[:-1]

# plot
img = plt.figure(1, figsize=(4,3))

plt.bar(bincenters, num, widths, color='C1', label=f'Run MC {num} times')

totalWeight = num.sum()
print('Total weight =', totalWeight)

plt.xlabel(var)
plt.ylabel('Number of events')
plt.show()


xt1_mc=np.array(path_xt1)/(np.pi/4)
xd1_mc=np.array(path_xd1)/(1e-3)

num_t,edges_t=np.histogram(xt1_mc, bins=50)
num_d,edges_d=np.histogram(xd1_mc, bins=50)
centers_t=find_centers(edges_t)
centers_d=find_centers(edges_d)

index_t=find_max_index(num_t)
index_d=find_max_index(num_d)

fit_t, cov_t = curve_fit(Gaussian, centers_t, num_t, [max(num_t), centers_t[index_t], 0.001])
fit_d, cov_d = curve_fit(Gaussian, centers_d, num_d, [max(num_d), centers_d[index_d], 0.001])

xt_fit=np.linspace(edges_t[0],edges_t[-1],100)
xd_fit=np.linspace(edges_d[0],edges_d[-1],100)

yt_fit=Gauss(xt_fit,fit_t[0],fit_t[1],fit_t[2])
yd_fit=Gauss(xd_fit,fit_d[0],fit_d[1],fit_d[2])

print(u'  Δm^2=%.3fe-3, θ=%.3fπ/4'%(fit_d[1],fit_t[1]))


fig,ax=plt.subplots()
ax.plot(xt_fit,yt_fit)
ax.hist(xt1_mc,bins=50)
ax.set_xlabel(u'$θ_{23}$')
  


fig,ax=plt.subplots()
ax.plot(xd_fit,yd_fit)
ax.hist(xd1_mc,bins=50)
ax.set_xlabel(u'$Δm_{23}^2$')
