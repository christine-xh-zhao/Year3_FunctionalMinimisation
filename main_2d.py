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

pl_func.data_aligned(energy, data_osc, data_unosc, width, plot=True)

# calculate prob and plot
pl_func.neutrino_prob_sing(energy, plot=True)

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


# plot a big 2D diagram
if True:
    N = 100  # total number per dimension
    theta_list = np.linspace(0, np.pi/2, N)
    dm2_list = np.linspace(0, 40e-3, N)

    pl_func.nll_2d_theta_dm2(min_func, N, theta_list, dm2_list)


# plot NLL against dm2
if plot:
    num = 1000  # number of values to generate
    dm2_max = 100e-3
    dm2_min = 0

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min
            )

if plot:
    num = 100  # number of values to generate
    dm2_max = 2.7e-3
    dm2_min = 2.1e-3

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min
            )


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
print('\n- Uncertainty of theta_min -')

print('\n-- Use NLL changes by +/- 1 in absolute units')

# values
theta_plus = np.pi/4  # upper bound for theta_min
theta_minus = theta_min - theta_min/10  # lower bound for theta_min
nll_change = nll_min + 1  # upper and lower bound for NLL


print('\n--- Select the closest value from a list')

theta_stdup, theta_stddown = un.list(
    min_func,
    theta_min, theta_plus, theta_minus,
    nll_change)

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n--- Use secant method to solve theta giving expected NLL')

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


print('\n--- Use Hessian and thus covariance matrix')

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

# minimise and plot a small 2D diagram
N = 100  # total number per dimension

# zoomed on four minima
dm2_high = 2.7e-3
dm2_low = 2.1e-3
theta_high = 0.95
theta_low = 0.62

# zoomed on two minima
# dm2_high = 2.4e-3
# dm2_low = 2.34e-3
# theta_high = 0.845
# theta_low = 0.725

# univariate
theta_guess = 0.65
dm2_guess = 2.25e-3

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot,
 theta_all, dm2_all) = min_func.univariate(
    [theta_guess, dm2_guess],
    [50, 150],
    num_max=100, stop_cond=1e-10
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
dm2_guess = 2.25e-3

params, nll_min, err_list, params_list = min_func.Newtons(
    [theta_guess, dm2_guess],
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

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
dm2_guess = 2.35e-3

params, nll_min, err_list, params_list = min_func.quasi_Newton(
    [theta_guess, dm2_guess],
    alpha=3e-9,
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

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
dm2_guess = 2.35e-3

params, nll_min, err_list, params_list = min_func.gradient_descent(
    [theta_guess, dm2_guess],
    alpha=3e-5,
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

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

- MC is a random process
- If the plots produced are not as expected, run the script few more times
- Especially for CSA, since T decreases too slow, which means the distribution might be significantly messed up
'''
print()
print('-'*26)
print('--- Monte-Carlo method ---')

run_once = False

print('\n- Classical simulated annealing -\n')

# inital guess same as before
theta_guess = 0.65
dm2_guess = 2.35e-3
T0 = 25
step = 7.5e-3
rho = 0.9
stop_cond = 5e-4

# initial guess to show it can find the global maxima
# theta_guess = 0.75
# dm2_guess = 2.6e-3
# T0 = 80
# step = 1e-2
# rho = 0.9
# stop_cond = 1e-4

if run_once:
    # Monte-Carlo classical simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step, rho,
        num_max=5000, stop_cond=stop_cond,
        method='CSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

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


# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run CSA {N_MC} times and estimate from distribution --')

theta_entry = []
dm2_entry = []
for i in range(N_MC):
    # Monte-Carlo classical simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=4000,
        method='CSA',
        printout=False
        )

    theta_plot, dm2_plot = params_list[0], params_list[1]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()

# fit distribution with Gaussian
plot_mul = plot
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$", N=N_MC, plot=plot_mul)
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$", N=N_MC, plot=plot_mul)

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)


print('\n\n- Fast simulated annealing -\n')

# inital guess same as before
theta_guess = 0.65
dm2_guess = 2.35e-3
T0 = 25
step = 5e-4
stop_cond = 5e-6

# initial guess to show it can find the global maxima
# theta_guess = 0.75
# dm2_guess = 2.6e-3
# T0 = 80
# step = 2.5e-4
# stop_cond = 5e-6

if run_once:
    # Monte-Carlo fast simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=5000, stop_cond=stop_cond,
        method='FSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

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


# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run FSA {N_MC} times and estimate from distribution --')

theta_entry = []
dm2_entry = []
for i in range(N_MC):
    # Monte-Carlo fast simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=2000,
        method='FSA',
        printout=False
        )

    theta_plot, dm2_plot = params_list[0], params_list[1]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()

# fit distribution with Gaussian
plot_mul = plot
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$", N=N_MC, plot=plot_mul)
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$", N=N_MC, plot=plot_mul)

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func, theta_min, dm2_min)