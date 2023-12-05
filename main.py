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

num = 100  # number of values to generate
theta_max = np.pi/2  # full range, whould have two minima
theta_min = 0
theta_list = np.linspace(theta_min, theta_max, num)  # list of theta values

nll_list = []
for theta in theta_list:  # calculate NLL for each theta
    nll = min_func.cal_nll(theta)
    nll_list += [nll]

nll_list = np.array(nll_list)

if plot:
    pl_func.nll_1d_theta(theta_list, nll_list)


'''
1D parabolic minimisor
'''
print('-'*30)
print('--- 1D parabolic minimiser ---\n')

print('dm2 = 2.4 e-3 used\n')

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

if plot:
    pl_func.visual_univeriate(
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


# estimate error
print('\n-- Uncertainties from Hessian --')

hes = un.hessian(func=min_func.cal_nll, x=np.array([theta_min, dm2_min]))

hes_inv = np.linalg.inv(hes)  # inverse hessian to get covariance
sig_theta = np.sqrt(2) * np.sqrt(hes_inv[0][0])  # std is sqrt of covariance diagonal
sig_dm2 = np.sqrt(2) * np.sqrt(hes_inv[1][1])

print(f'\ntheta_min = {theta_min:.4f} +/- {sig_theta:.4f}')
print(f'with % error +/- {(sig_theta*100/theta_min):.2f}')

print(f'\ndm2_min = {dm2_min:.4f} +/- {sig_dm2:.4f}')
print(f'with % error +/- {(sig_dm2*100/dm2_min):.2f}')


'''
Newton's method
'''
print()
print('-'*23)
print('--- Newton\'s method ---\n')

def Newtons(self, theta0, dm20, num_max=100, stop_cond=1e-10):
    """
    Newton's method
    """

    # initalise
    params = np.array([theta0, dm20])
    nll = min_func.cal_nll(params)
    theta_list = []
    dm2_list = []
    theta_list += [theta0]
    dm2_list += [dm20]
    err_list = []

    # iterate
    num = 1
    while True:
        # grad of function
        grad = un.gradient(f=min_func.cal_nll, x=params)

        # inverse hessian
        hes = un.hessian(func=min_func.cal_nll, x=params)
        hes_inv = np.linalg.inv(hes)

        # update parameters and nll
        params_new = params - np.dot(hes_inv, grad)
        nll_new = min_func.cal_nll(params_new)
        theta_list += [params_new[0]]
        dm2_list += [params_new[1]]

        # error
        err = abs(nll_new - nll)
        err_list += [err]

        if err <= stop_cond:
            print(f'Stopping condition {stop_cond} reached after {num} iterations')
            print(f'Minimum of nll = {nll_new} is at\ntheta = {params_new[0]}\ndm2 = {params_new[1]} e-3')
            break

        if num == num_max:
            print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
            break

        num += 1
        nll = 1. * nll_new

    return params_new[0], params_new[1], nll_new, err_list, theta_list, dm2_list

# Newtons
theta_guess = 0.65
dm2_guess = 2.25

(theta_min, dm2_min, nll_min,
 err_list,
 theta_plot, dm2_plot) = Newtons(
     theta_guess, dm2_guess,
     num_max=100, stop_cond=1e-10
     )


def visual_methods(
        min_func,
        N,
        theta_low, theta_high,
        dm2_low, dm2_high,
        theta_min, dm2_min,
        theta_plot, dm2_plot,
        ):

    theta_list = np.linspace(theta_low, theta_high, N)
    dm2_list = np.linspace(dm2_low, dm2_high, N)

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # possible colors: "YlGnBu", "Reds", "viridis", "bone", "nipy_spectral", "gist_ncar", "jet"
    for i in ["nipy_spectral"]:

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

        cntr1 = ax1.contourf(theta_list, dm2_list, nll_list, 300, cmap=i)
        ax1.set_xlabel(r"$\theta_{23}$ $[rad]$")
        ax1.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3}\/ \/eV^2]$")
        ax1.plot(theta_min, dm2_min, 'x', color='red', label='Minimum')
        
        # plot path
        X, Y = theta_plot[:-1], dm2_plot[:-1]
        U = np.subtract(theta_plot[1:], theta_plot[:-1])
        V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
        ax1.quiver(X, Y, U, V, color="white", angles='xy', scale_units='xy', scale=1, label='Min of the step')

        plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

        fig.colorbar(cntr1, ax=ax1, label="Negative Log Likelihood")
        
        ax1.legend()
        plt.show()

if plot:
    visual_methods(
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