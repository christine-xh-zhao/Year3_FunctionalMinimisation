"""
Main file for 2D minimisation
"""

import io
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import minimiser as mi
import uncertainty as un
import plot as pl_func

# plot one method per graph
plot = False

# plot for report
plot_all = True

if plot_all:
    # set the directory path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path)

    # make new folder
    folder_name = '/plots-2D'
    dir_folder = dir_path + folder_name
    filename = dir_folder + '/placeholder.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)


# minimisation class
min_func = mi.Minimiser()

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


# plot multiple methods on one graph
if plot_all:
    # generate data to plot
    theta_list = np.linspace(theta_low, theta_high, N)
    dm2_list = np.linspace(dm2_low, dm2_high, N)

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # plot colours
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    ax1 = axes[0]
    ax2 = axes[1]

    cntr1 = ax1.contourf(
        theta_list, dm2_list, nll_list,
        levels=500,
        cmap='nipy_spectral',
        )

    ax1.set_ylabel(r"$\Delta m_{23}^2$ $[eV^2]$")
    ax1.annotate ("a)", (-0.15, 1.075), xycoords = "axes fraction")

    # plot contours
    cntr1.levels = cntr1.levels.tolist()
    ax1.contour(cntr1, levels=cntr1.levels[1:30:8], colors='w', alpha=0.5)
    ax1.contour(cntr1, levels=cntr1.levels[40:-1:42], colors='w', alpha=0.5)


'''
Univariate method
'''
print()
print('-'*25)
print('--- Univariate method ---\n')

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
un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax1.quiver(X, Y, U, V, color="white", angles='xy', scale_units='xy', scale=1, label='Univariate')


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
theta_new, dm2_new = theta_min, dm2_min  # save values for plotting later
theta_plot, dm2_plot = params_list[0], params_list[1]

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax1.quiver(X, Y, U, V, color="yellow", angles='xy', scale_units='xy', scale=1, label='Newton')


'''
Quasi-Newton method
'''
print()
print('-'*27)
print('--- Quasi-Newton method ---\n')

# Quasi-Newton
theta_guess = 0.65
dm2_guess = 2.25e-3

params, nll_min, err_list, params_list = min_func.quasi_Newton(
    [theta_guess, dm2_guess],
    alpha=0.8e-9,
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax1.quiver(X, Y, U, V, color="orange", angles='xy', scale_units='xy', scale=1, label='Quasi-Newton')


'''
Gradient descent
'''
print()
print('-'*24)
print('--- Gradient descent ---\n')

# gradient descent
theta_guess = 0.65
dm2_guess = 2.25e-3

params, nll_min, err_list, params_list = min_func.gradient_descent(
    [theta_guess, dm2_guess],
    alpha=3e-5,
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax1.quiver(X, Y, U, V, color="silver", angles='xy', scale_units='xy', scale=1, label='Gradient descent')

    ax1.legend()


'''
Monte-Carlo method

- MC is a random process
- If the plots produced are not as expected, run the script few more times
- Especially for CSA, since T decreases too slow, which means the distribution might be significantly messed up
'''
print()
print('-'*26)
print('--- Monte-Carlo method ---')

# run each MC method once
run_once = True


if plot_all:
    # generate data to plot
    theta_list = np.linspace(theta_low, theta_high, N)
    dm2_list = np.linspace(dm2_low, dm2_high, N)

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # plot colours
    plt.subplot(2, 1, 2)
    cntr2 = ax2.contourf(
        theta_list, dm2_list, nll_list,
        levels=500,
        cmap='nipy_spectral',
        )

    ax2.set_xlabel(r"$\theta_{23}$ $[rad]$")
    ax2.set_ylabel(r"$\Delta m_{23}^2$ $[eV^2]$")
    
    ax2.annotate ("b)", (-0.15, 1.075), xycoords="axes fraction")

    # plot contours
    cntr2.levels = cntr2.levels.tolist()
    ax2.contour(cntr2, levels=cntr2.levels[1:30:8], colors='w', alpha=0.5)
    ax2.contour(cntr2, levels=cntr2.levels[40:-1:42], colors='w', alpha=0.5)


print('\n- Classical simulated annealing -\n')

# inital guess same as before
# theta_guess = 0.65
# dm2_guess = 2.35e-3
# T0 = 25
# step = 7.5e-3
# rho = 0.9
# stop_cond = 5e-4

# initial guess to show it can find the global maxima
theta_guess = 0.75
dm2_guess = 2.6e-3
T0 = 80
step = 1e-2
rho = 0.9
stop_cond = 1e-4

if run_once:
    # Monte-Carlo classical simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step, rho,
        num_max=1.2e4, stop_cond=stop_cond,
        method='CSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

    # estimate error
    print('\nUncertainties from Hessian')
    un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax2.quiver(X, Y, U, V, color="silver", angles='xy', scale_units='xy', scale=1, label='CSA')


print('\n\n- Fast simulated annealing -\n')

# inital guess same as before
# theta_guess = 0.65
# dm2_guess = 2.35e-3
# T0 = 25
# step = 5e-4
# stop_cond = 5e-6

# initial guess to show it can find the global maxima
theta_guess = 0.75
dm2_guess = 2.6e-3
T0 = 80
step = 2.5e-4
stop_cond = 1e-6

if run_once:
    # Monte-Carlo fast simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=1.2e4, stop_cond=stop_cond,
        method='FSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

    # estimate error
    print('\nUncertainties from Hessian')
    un.std_2(min_func.cal_nll, theta_min, dm2_min)

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


if plot_all:
    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax2.quiver(X, Y, U, V, color="white", angles='xy', scale_units='xy', scale=1, label='FSA')

    ax2.legend()

    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

    fig.colorbar(cntr1, ax=axes, label="Negative Log Likelihood")

    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                bbox_inches='tight')
    png2 = Image.open(png1)
    png2.save(dir_folder + "/visualise-all.png")
    png1.close()

    plt.show()


# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run CSA {N_MC} times and estimate from distribution --')

# inital guess same as before
# theta_guess = 0.65
# dm2_guess = 2.35e-3
# T0 = 25
# step = 7.5e-3
# rho = 0.9
# stop_cond = 5e-4

# initial guess to show it can find the global maxima
theta_guess = 0.75
dm2_guess = 2.6e-3
T0 = 80
step = 1e-2
rho = 0.9
stop_cond = 1e-4

theta_entry = []
dm2_entry = []
for i in range(N_MC):
    # Monte-Carlo classical simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=8000,
        method='CSA',
        printout=False
        )

    theta_plot, dm2_plot = params_list[0], params_list[1]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()

# fit distribution with Gaussian
plot_mul = plot_all
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(
    var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$",
    N=N_MC, dir_folder=dir_folder, var_string='theta', plot=plot_mul
    )
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(
    var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$",
    N=N_MC, dir_folder=dir_folder, var_string='dm2', plot=plot_mul
    )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)


# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run FSA {N_MC} times and estimate from distribution --')

# inital guess same as before
# theta_guess = 0.65
# dm2_guess = 2.35e-3
# T0 = 25
# step = 5e-4
# stop_cond = 5e-6

# initial guess to show it can find the global maxima
theta_guess = 0.75
dm2_guess = 2.6e-3
T0 = 80
step = 2.5e-4
stop_cond = 1e-6

theta_entry = []
dm2_entry = []
for i in range(N_MC):
    # Monte-Carlo fast simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=4000,
        method='FSA',
        printout=False
        )

    theta_plot, dm2_plot = params_list[0], params_list[1]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()

# fit distribution
plot_mul = plot_all
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(
    var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$",
    N=N_MC, dir_folder=dir_folder, var_string='theta', FSA=True, plot=plot_mul
    )
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(
    var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$",
    N=N_MC, dir_folder=dir_folder, var_string='dm2', FSA=True, plot=plot_mul
    )

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)


'''
Plot to compare with the observed data
'''
if plot_all:
    pl_func.data_aligned_2D(theta_new, dm2_new, dir_folder, plot=True)
