"""
Load data function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LoadData():

    def __init__(self):
        pass

    def data_energy(self):

        # load data
        df = pd.read_csv('data_cz2421.txt', sep=" ", header=None)

        self.data_osc = np.array(df[0][1:201].astype(int))  # number of oscillated data in each bin
        self.data_unosc = np.array(df[0][202:].astype(float))  # ... non-oscillated data ...

        self.width = 0.05  # bin width
        self.energy = np.arange(0.025, 10, self.width)  # bin centre energy in GeV

    def simple_plot(self, plot=False):
        
        # plot
        self.plot = plot

        if self.plot:
            fig = plt.figure()
            plt.bar(self.energy, self.data_osc, self.width, label='oscillated')
            plt.ylabel('# of entries')
            plt.xlabel('energy (GeV)')
            plt.legend()

            fig = plt.figure()
            plt.bar(self.energy, self.data_unosc, self.width, label='unoscillated', color='C1')
            plt.ylabel('# of entries')
            plt.xlabel('energy (GeV)')
            plt.legend()

            fig = plt.figure()
            plt.bar(self.energy, self.data_unosc, self.width, label='unoscillated', color='C1')
            plt.bar(self.energy, self.data_osc, self.width, label='oscillated', color='C0')
            plt.ylabel('# of entries')
            plt.xlabel('energy (GeV)')
            plt.legend()

            plt.show()

    def get_data(self):
        self.data_energy()
        return self.data_osc, self.data_unosc, self.energy, self.width
    