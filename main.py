"""
Main file for running code
"""

import numpy as np
import matplotlib.pyplot as plt

import function as fc
import load_data as ld
import minimiser as mi

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

# 1D minimisor
print('-'*30)
print('--- 1D parabolic minimiser ---\n')

min_func = mi.Minimiser()
theta_min, nll_min = min_func.parabolic_1d(x_list=theta_list, y_list=nll_list)

print('where x is theta and y is NLL')
print('\nUse 2e-06 cannot converge after 100 iterations')

# finding the accuracy using one std
print('\n--- Uncertainty of theta_min ---')

print('\n- Use NLL changes by +/- 1 in absolute units')

print('\n-- Select the closest value from a list')

num = 500  # number of values to generate
theta_plus = np.pi/4
theta_minus = theta_min - theta_min/10
theta_plus_list = np.linspace(theta_min, theta_plus, num)  # list of theta values above theta min
theta_minus_list = np.linspace(theta_minus, theta_min-theta_min/100, num)  # list of theta values below theta min

# upper and lower bound for NLL
nll_change = nll_min + 1

for i in range(num):  # calculate NLL for each theta above theta min
    theta = theta_plus_list[i]
    nll = min_func.cal_nll(theta=theta)
    if np.isclose(nll, nll_change):
        print(f'\nUpper theta = {theta} gives roughly nll_min + 1 = {nll_change}')
        print(f'which the % difference from expected nll_change is {(nll - nll_change)*100/nll_change:.4f}')
        theta_stdup = theta
        break

for i in range(num):  # calculate NLL for each theta above theta min
    theta = theta_minus_list[-i]
    nll = min_func.cal_nll(theta=theta)
    if np.isclose(nll, nll_change):
        print(f'\nLower theta = {theta} gives roughly nll_min + 1 = {nll_change}')
        print(f'which the % difference from expected nll_change is {(nll - nll_change)*100/nll_change:.4f}')
        theta_stddown = theta
        break

print(f'\ntheta_min = {theta_min:.4f} +{(theta_stdup-theta_min):.4f} or {(theta_stddown-theta_min):.4f}')
print(f'with % error +{((theta_stdup-theta_min)*100/theta_min):.2f} or {((theta_stddown-theta_min)*100/theta_min):.2f}')

print(f'\ntheta should be lower than pi/4 = {(np.pi/4):.4f} which is the local maxima')

print('\n-- Use secant method to solve')

def secant(x1, x2, err, sol_true, stop_err=1e-6, iter_max=100):
    """
    Secant method for root finding
    """
    n = 0
    
    while True: 
        # calculate the intermediate value
        sol1 = euler_sec(del_t, ti, tf, x0, x1)
        sol2 = euler_sec(del_t, ti, tf, x0, x2)

        gradient = (sol1 - sol2) / (x1 - x2)
        x_dot_new = x1 + ((sol_true - sol1) / gradient)

        # update the value of interval
        x1 = x2
        x2 = x_dot_new

        # update number of iteration 
        n += 1 

        # compare with true value and stopping condition
        sol_new = euler_sec(del_t, ti, tf, x0, x_dot_new)
        if (abs(sol_new - sol_true) < stop_err):
            print("Root of the given equation =", round(x_dot_new, 6)) 
            print('(Estimated gradient at t = 0 s)\n')
            print("No. of iterations =", n) 
            return x_dot_new

        # if reaches max iteration number
        if (n == iter_max):
            print('Reaches max', iter_max, 'iteration')
            break

print('\n- Use covariance matrix')