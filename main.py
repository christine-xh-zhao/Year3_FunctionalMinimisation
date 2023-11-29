"""
Main file for running code
"""

import numpy as np
import matplotlib.pyplot as plt

import function as fc
import load_data as ld

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
num = 70  # number of values to generate
theta_max = np.pi/2
theta_min = 0
theta_step = (theta_max - theta_min)/num
theta_list = np.arange(theta_min, theta_max, theta_step)  # list of theta values

nll_list = []
for i in range(num):
    theta = theta_list[i]
    prob = fc.neutrino_prob(E=energy, theta=theta)
    data_unosc_prob = data_unosc*prob
    nll = fc.NLL(lamb=data_unosc_prob, m=data_osc)
    nll_list += [nll]

nll_list = np.array(nll_list)

fig = plt.figure()
plt.plot(theta_list, nll_list, '.')
plt.ylabel('NLL')
plt.xlabel('theta')
plt.show()

# 1D minimisor
indices = (nll_list.argsort()[:6])  # extra indices of the three points around each minimum (total two minima)
indices = np.sort(indices)
print(indices)
