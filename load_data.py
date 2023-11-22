"""
Load data function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot = True

# load data
df = pd.read_csv('data_cz2421.txt', sep=" ", header=None)

data_osc = np.array(df[0][1:201].astype(float))  # number of oscillated data in each bin
data_unosc = np.array(df[0][202:].astype(float))  # ... non-oscillated data ...

width = 0.05  # bin width
energy = np.arange(0.025, 10, width)  # bin centre energy in GeV

# plot
if plot:
    fig = plt.figure()
    plt.bar(energy, data_osc, width, label='oscillated')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, label='unoscillated', color='C1')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    fig = plt.figure()
    plt.bar(energy, data_osc, width, label='oscillated')
    plt.bar(energy, data_unosc, width, label='unoscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    plt.show()
