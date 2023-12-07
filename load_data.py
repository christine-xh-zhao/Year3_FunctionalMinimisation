"""
Load data function
"""

import pandas as pd
import numpy as np


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

    def get_data(self):
        self.data_energy()
        return self.data_osc, self.data_unosc, self.energy, self.width
    