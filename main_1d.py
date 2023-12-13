"""
Main file for 1D minimisation and some other plots
"""

import os
import sys
import numpy as np

import function as fc
import load_data as ld
import minimiser as mi
import uncertainty as un
import plot as pl_func

# plot one method per graph
plot = False


# set the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# make new folder
folder_name = '/plots-1D'
dir_folder = dir_path + folder_name
filename = dir_folder + '/placeholder.txt'
os.makedirs(os.path.dirname(filename), exist_ok=True)


# load data
data = ld.LoadData()
data_osc, data_unosc, energy, width = data.get_data()  # energy (GeV) in each bin

if plot:
    pl_func.data(energy, data_osc, data_unosc, width)

pl_func.data_aligned(energy, data_osc, data_unosc, width, dir_folder, plot=True)

# calculate prob and plot
pl_func.neutrino_prob_sing(energy, dir_folder, plot=True)

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


min_func = mi.Minimiser()

# NLL against theta
if True:
    num = 500  # number of values to generate
    theta_max = np.pi/2  # full range, whould have two minima
    theta_min = 0

    pl_func.nll_1d_theta(
        min_func,
        theta_min, theta_max,
        num,
        dir_folder,
        dm2=2.4,
        plot=True
        )


# plot a big 2D diagram
if plot:
    N = 100  # total number per dimension, use 1000 to regenerate the plot in report
    theta_list = np.linspace(0, np.pi/2, N)
    dm2_list = np.linspace(0, 40e-3, N)

    pl_func.nll_2d_theta_dm2(min_func, N, theta_list, dm2_list, dir_folder)


# plot NLL against dm2
if plot:
    num = 100  # number of values to generate, use 1000 to regenerate the plot in report
    dm2_max = 100e-3
    dm2_min = 0

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min=np.pi/4
            )

if plot:
    num = 100  # number of values to generate
    dm2_max = 2.7e-3
    dm2_min = 2.1e-3

    pl_func.nll_1d_dm2(
            min_func,
            dm2_min, dm2_max,
            num,
            theta_min=np.pi/4
            )


# plot NLL against alpha
if plot:
    num = 500  # total number per dimension
    alpha_min = 1e-3
    alpha_max = 20

    pl_func.nll_1d_alpha(
            min_func,
            alpha_min, alpha_max,
            num,
            dir_folder,
            theta=np.pi/4, dm2=2.4e-3,
            plot=True
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