"""
Minimisers
"""

import numpy as np
import copy

import load_data as ld
import function as fc


class Univariate():

    def __init__(self):
        data_osc, data_unosc, energy, width = ld.LoadData().get_data()  # energy (GeV) in each bin
        
        self.data_osc = data_osc
        self.data_unosc = data_unosc
        self.energy = energy
    
    def min_indices(self, y_list, num):
        """
        Calculate the indices of the smallest num points
        """
        return y_list.argsort()[:num]

    def cal_x3(self, x0, x1, x2, y0, y1, y2):
        """
        Calculate second order Lagrange polynomial to estimate x3
        """

        numer = (x2**2-x1**2)*y0 + (x0**2-x2**2)*y1 + (x1**2-x0**2)*y2
        denom = (x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2
        x3 = (numer/denom)/2

        return x3

    def cal_y3(self, x3):
        """
        Calculate NLL from given theta x3
        """

        prob = fc.neutrino_prob(E=self.energy, theta=x3)
        data_unosc_prob = self.data_unosc*prob
        y3 = fc.NLL(lamb=data_unosc_prob, m=self.data_osc)

        return y3


    def parabolic_1d(self, x_list, y_list):
        """
        1D parabolic minimiser for neutrino_prob function
        """

        # inital values of the three points around first minimum
        ind_min = self.min_indices(y_list, 1)
        x0 = x_list[ind_min[0]]
        y0 = y_list[ind_min[0]]
        x1, x2 = x0 + x0/10, x0 - x0/10
        y1, y2 = y0 + y0/10, y0 - y0/10

        # iterate
        stop_cond = 1e-10
        num_max = 100
        num = 1
        y3 = 0
        while True:
            x3 = self.cal_x3(x0, x1, x2, y0, y1, y2)
            y3_new = self.cal_y3(x3)

            if (y3_new - y3) <= stop_cond:
                print(f'Stopping condition {stop_cond} reached at {num} iterations')
                print(f'Minimum is x = {x3} y = {y3_new}')
                break

            x_list_new = np.array([x0, x1, x2, x3])
            y_list_new = np.array([y0, y1, y2, y3_new])

            ind_min3 = self.min_indices(y_list_new, 3)  # indices for min three points

            # update
            x0, x1, x2 = x_list_new[ind_min3[0]], x_list_new[ind_min3[1]], x_list_new[ind_min3[2]]
            y0, y1, y2 = y_list_new[ind_min3[0]], y_list_new[ind_min3[1]], y_list_new[ind_min3[2]]

            if num >= num_max:
                print(f'Max iterations {num_max} reached')
                break

            num += 1
            y3 = copy.deepcopy(y3_new)

        return x3, y3_new
    