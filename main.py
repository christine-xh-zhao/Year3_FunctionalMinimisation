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
    pl_func.data_with_prob(energy, data_unosc_prob, data_unosc, data_osc, width)

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

theta_min, nll_min = min_func.parabolic_1d(
    x_list=theta_list, y_list=nll_list, stop_cond=1e-10)

print('where x is theta and y is NLL')
print('\nUse 2e-06 or below cannot converge after 100 iterations')

print(f'\ntheta should be lower than pi/4 = {(np.pi/4):.4f} which is the local maxima')


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

theta_stdup, theta_stddown = un.list(min_func, theta_min, theta_plus, theta_minus, nll_change)

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n---- Use secant method to solve theta giving expected NLL')

theta_stdup, nll_stdup = un.secant(min_func=min_func, x1=theta_min, x2=theta_plus, sol_true=nll_change) 
theta_stddown, nll_stddown = un.secant(min_func=min_func, x1=theta_min, x2=theta_minus, sol_true=nll_change) 

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

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    pl_func.nll_2d_theta_dm2(theta_list, dm2_list, nll_list, plot=True)


# plot NLL against dm2
if plot:
    num = 1000  # number of values to generate
    dm2_max = 100  # full range/2 to exclue one minimum
    dm2_min = 0
    dm2_list = np.linspace(dm2_min, dm2_max, num)  # list of theta values

    nll_list = []
    for dm2 in dm2_list:  # calculate NLL for each dm2
        nll = min_func.cal_nll([theta_min, dm2])
        nll_list += [nll]

    nll_list = np.array(nll_list)

    pl_func.nll_1d_dm2(dm2_list, nll_list, theta=theta_min)

if plot:
    num = 100  # number of values to generate
    dm2_max = 2.7
    dm2_min = 2.1
    dm2_list = np.linspace(dm2_min, dm2_max, num)  # list of theta values

    nll_list = []
    for dm2 in dm2_list:  # calculate NLL for each dm2
        nll = min_func.cal_nll([theta_min, dm2])
        nll_list += [nll]

    nll_list = np.array(nll_list)

    pl_func.nll_1d_dm2(dm2_list, nll_list, theta=theta_min)


# minimise and plot a small 2D diagram
N = 100  # total number per dimension

# zoomed on four minima
# dm2_max = 2.52
# dm2_min = 2.28
# theta_max = 0.87
# theta_min = 0.7

# zoomed on two minima
dm2_max = 2.4
dm2_min = 2.34
theta_max = 0.845
theta_min = 0.725

theta_list = np.linspace(theta_min, theta_max, N)
dm2_list = np.linspace(dm2_min, dm2_max, N)

nll_list = np.zeros((N, N))
for i in range(len(theta_list)):
    for j in range(len(dm2_list)):
        nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
        nll1 = 1. * nll
        nll_list[j][i] = nll1

nll_list = np.array(nll_list)

theta_guess = 0.75
dm2_guess = 2.37
theta_min, dm2_min, nll_min, theta_all, dm2_all, theta_update, dm2_update = min_func.univariate(
    theta_guess, dm2_guess, gue_step_the=50, gue_step_dm2=150, num_max=100, stop_cond=1e-10)

if plot:
    pl_func.visual_univeriate(
        theta_list, dm2_list, nll_list,
        theta_min, dm2_min,
        theta_all, dm2_all,
        theta_update, dm2_update,
        theta_guess, dm2_guess
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