"""
Main file for 3D minimisation
"""

import os
import sys

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
    folder_name = '/plots-3D'
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


'''
Monte-Carlo method

- MC is a random process
- If the plots produced are not as expected, run the script few more times
- Especially for CSA, since T decreases too slow, which means the distribution might be significantly messed up
'''
print()
print('-'*29)
print('--- 3D Monte-Carlo method ---')

# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run FSA {N_MC} times and estimate from distribution --')

# inital guess
theta_guess = 0.65
dm2_guess = 2.35e-3
alpha_guess = 1
T0 = 25
step = 5e-4

# loop
theta_entry = []
dm2_entry = []
alpha_entry = []
for i in range(N_MC):
    # Monte-Carlo fast simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess, alpha_guess],
        T0, step,
        num_max=4000,
        method='FSA',
        printout=False
        )

    theta_plot, dm2_plot, alpha_plot = params_list[0], params_list[1], params_list[2]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()
    alpha_entry += alpha_plot.tolist()

# fit distribution
plot_mul = plot
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(
    var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$",
    N=N_MC, FSA=True, plot=plot_mul
    )
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(
    var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$",
    N=N_MC, FSA=True, plot=plot_mul
    )
print('\nalpha_min')
alpha_min, _ = pl_func.fit_MC(
    var_list=alpha_entry, var=r"$\alpha$ $[GeV^{-1}]$",
    N=N_MC, FSA=True, plot=plot_mul
    )


# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)


# run MC N times and estimate minima by fitting Gaussian to the distribution
N_MC = 10
print(f'\n-- Run CSA {N_MC} times and estimate from distribution --')

# inital guess same as before
theta_guess = 0.7
dm2_guess = 2.5e-3
alpha_guess = 1.2
T0 = 750
step = 1.25e-1
rho = 0.8

# loop
theta_entry = []
dm2_entry = []
alpha_entry = []
for i in range(N_MC):
    # Monte-Carlo fast simulated annealing
    _, _, _, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess, alpha_guess],
        T0, step, rho,
        num_max=2e4,
        method='CSA',
        printout=False
        )

    theta_plot, dm2_plot, alpha_plot = params_list[0], params_list[1], params_list[2]

    theta_entry += theta_plot.tolist()
    dm2_entry += dm2_plot.tolist()
    alpha_entry += alpha_plot.tolist()

# fit distribution with Gaussian
plot_mul = plot
print('\ntheta_min')
theta_min, _ = pl_func.fit_MC(
    var_list=theta_entry, var=r"$\theta_{23}$ $[rad]$",
    N=N_MC, plot=plot_mul
    )
print('\ndm2_min')
dm2_min, _ = pl_func.fit_MC(
    var_list=dm2_entry, var=r"$\Delta m_{23}^2$ $[eV^2]$",
    N=N_MC, plot=plot_mul
    )
print('\nalpha_min')
alpha_min, _ = pl_func.fit_MC(
    var_list=alpha_entry, var=r"$\alpha$ $[GeV^{-1}]$",
    N=N_MC, plot=plot_mul
    )

# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)


'''
Gradient descent
'''
print()
print('-'*27)
print('--- 3D Gradient descent ---\n')

# gradient descent
theta_guess = 0.7
dm2_guess = 2.5e-3
alpha_guess = 1.2
stop_cond = 1e-6

params, nll_min, err_list, params_list = min_func.gradient_descent(
    [theta_guess, dm2_guess, alpha_guess],
    alpha=1e-4,
    num_max=100, stop_cond=stop_cond
    )

theta_min, dm2_min, alpha_min = params[0], params[1], params[2]
theta_plot, dm2_plot, alpha_plot = params_list[0], params_list[1], params_list[2]

# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)

# plot
if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$" + ' & ' + r"$\alpha$",
        stop=stop_cond
        )
    

'''
Quasi-Newton method
'''
print()
print('-'*30)
print('--- 3D Quasi-Newton method ---\n')

# Quasi-Newton
theta_guess = 0.7
dm2_guess = 2.5e-3
alpha_guess = 1.2
stop_cond = 1e-10

params, nll_min, err_list, params_list = min_func.quasi_Newton(
    [theta_guess, dm2_guess, alpha_guess],
    alpha=3e-9,
    num_max=100, stop_cond=stop_cond
    )

theta_min, dm2_min, alpha_min = params[0], params[1], params[2]
theta_plot, dm2_plot, alpha_plot = params_list[0], params_list[1], params_list[2]

# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)

# plot
if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$" + ' & ' + r"$\alpha$",
        stop=stop_cond
        )


'''
Newton's method
'''
print()
print('-'*26)
print('--- 3D Newton\'s method ---\n')

# Newtons
theta_guess = 0.7
dm2_guess = 2.5e-3
alpha_guess = 1.2
stop_cond = 1e-10

params, nll_min, err_list, params_list = min_func.Newtons(
    [theta_guess, dm2_guess, alpha_guess],
    num_max=100, stop_cond=stop_cond
    )

theta_min, dm2_min, alpha_min = params[0], params[1], params[2]
theta_new, dm2_new, alpha_new = theta_min, dm2_min, alpha_min  # save values for plotting later
theta_plot, dm2_plot, alpha_plot = params_list[0], params_list[1], params_list[2]

# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)

# plot
if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$" + ' & ' + r"$\alpha$",
        stop=stop_cond
        )
    

'''
Univariate method
'''
print()
print('-'*25)
print('--- Univariate method ---\n')

# univariate
theta_guess = 0.7
dm2_guess = 2.5e-3
alpha_guess = 1.2
stop_cond = 1e-10

(theta_min, dm2_min, alpha_min, nll_min,
 err_list,
 theta_plot, dm2_plot, alpha_plot) = min_func.univariate(
    [theta_guess, dm2_guess, alpha_guess],
    [50, 150, 50],
    num_max=100, stop_cond=stop_cond
    )

# estimate error
print('\nUncertainties from Hessian')
un.std_3(min_func, theta_min, dm2_min, alpha_min)

# plot
if plot:
    pl_func.change_nll(
        err_list,
        label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$" + ' & ' + r"$\alpha$",
        stop=stop_cond
        )


'''
Plot to compare with the observed data
'''
if plot_all:
    pl_func.data_aligned_3D(theta_new, dm2_new, alpha_new, dir_folder, plot=True)
