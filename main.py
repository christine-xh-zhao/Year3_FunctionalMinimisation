"""
Main file for running code
"""

import numpy as np
import matplotlib.pyplot as plt

import function as fc
import load_data as ld
import minimiser as mi
import uncertainty as un

plot = False

# load data
data = ld.LoadData()
data_osc, data_unosc, energy, width = data.get_data()  # energy (GeV) in each bin

data.simple_plot(plot=plot)

# calculate prob and plot
if plot:
    fig = plt.figure()
    plt.plot(energy, fc.neutrino_prob(E=energy), label='default params')
    plt.plot(energy, fc.neutrino_prob(E=energy, theta=np.pi/8), '-.', label='halve theta')
    plt.plot(energy, fc.neutrino_prob(E=energy, del_m=np.sqrt(2.4e-3)/2), ':', label='halve del_m')
    plt.plot(energy, fc.neutrino_prob(E=energy, L=295/2), '--', label='halve L')
    plt.ylabel('prob')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

# apply prob to unoscillated data
prob = fc.neutrino_prob(E=energy)
data_unosc_prob = data_unosc*prob

if plot:
    fig = plt.figure()
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, color='C1', label='unoscillated')
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob')
    plt.bar(energy, data_osc, width, color='C4', label='oscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

# NLL against theta
num = 100  # number of values to generate
theta_max = (np.pi/2)  # full range/2 to exclue one minimum
theta_min = 0
theta_list = np.linspace(theta_min, theta_max, num)  # list of theta values

nll_list = []
for i in range(num):  # calculate NLL for each theta
    theta = theta_list[i]
    prob = fc.neutrino_prob(E=energy, theta=theta)
    data_unosc_prob = data_unosc*prob
    nll = fc.NLL(lamb=data_unosc_prob, m=data_osc)
    nll_list += [nll]

nll_list = np.array(nll_list)

if plot:
    fig = plt.figure()
    plt.plot(theta_list, nll_list, '.')
    plt.ylabel('NLL')
    plt.xlabel('theta')
    plt.show()


'''
1D minimisor
'''
print('-'*30)
print('--- 1D parabolic minimiser ---\n')

min_func = mi.Minimiser()
theta_min, nll_min = min_func.parabolic_1d(x_list=theta_list, y_list=nll_list)

print('where x is theta and y is NLL')
print('\nUse 2e-06 cannot converge after 100 iterations')

print(f'\ntheta should be lower than pi/4 = {(np.pi/4):.4f} which is the local maxima')


'''
Finding the accuracy using one std
'''
print('\n--- Uncertainty of theta_min ---')

print('\n-- Use NLL changes by +/- 1 in absolute units')

# values
theta_plus = np.pi/4  # upper bound for theta_min
theta_minus = theta_min - theta_min/10  # lower bound for theta_min
nll_change = nll_min + 1  # upper and lower bound for NLL


print('\n- Select the closest value from a list')

theta_stdup, theta_stddown = un.list(min_func, theta_min, theta_plus, theta_minus, nll_change)

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n- Use secant method to solve theta giving expected NLL')

theta_stdup, nll_stdup = un.secant(min_func=min_func, x1=theta_min, x2=theta_plus, sol_true=nll_change) 
theta_stddown, nll_stddown = un.secant(min_func=min_func, x1=theta_min, x2=theta_minus, sol_true=nll_change) 

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')


print('\n-- Use Hessian and thus covariance matrix')

hes = un.hessian(func=min_func.cal_nll, x=np.array([theta_min]))

hes_inv = np.linalg.inv(hes)  # inverse hessian to get covariance
sig = np.sqrt(hes_inv)[0][0]  # std is sqrt of covariance diagonal

print(f'\ntheta_min = {theta_min:.4f} +/- {sig:.4f}')
print(f'with % error +/- {(sig*100/theta_min):.2f}')
