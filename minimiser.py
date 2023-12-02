"""
Minimisers
"""

import copy
import random
import numpy as np

import load_data as ld
import function as fc


class Minimiser():

    def __init__(self):
        data_osc, data_unosc, energy, width = ld.LoadData().get_data()  # energy (GeV) in each bin
        
        self.data_osc = data_osc
        self.data_unosc = data_unosc
        self.energy = energy


    def cal_x3(self, x_list, y_list):
        """
        Use second order Lagrange polynomial to estimate x3
        """

        x0, x1, x2 = x_list[0], x_list[1], x_list[2]
        y0, y1, y2 = y_list[0], y_list[1], y_list[2]

        numer = (x2**2-x1**2)*y0 + (x0**2-x2**2)*y1 + (x1**2-x0**2)*y2
        denom = (x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2
        x3 = (numer/denom)/2

        return x3

    def cal_nll(self, theta):
        """
        Calculate NLL from given theta
        """

        prob = fc.neutrino_prob(E=self.energy, theta=theta)
        data_unosc_prob = self.data_unosc*prob
        nll = fc.NLL(lamb=data_unosc_prob, m=self.data_osc)

        return nll


    def parabolic_1d(self, x_list, y_list, num_max=100, stop_cond=5e-6):
        """
        1D parabolic minimiser for neutrino_prob function
        """

        # inital values of the three points around first minimum
        ind_min = np.argmin(y_list)
        x0 = x_list[ind_min]
        x1 = x0 + x0/100; x2 = x0 - x0/100

        x_iter = [x0, x1, x2]
        y_iter = [self.cal_nll(x) for x in x_iter]  

        # iterate
        num = 1
        while True:
            
            x3 = self.cal_x3(x_iter, y_iter)
            y3 = self.cal_nll(x3)

            if abs(x_iter[-1] - x3) <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum is x = {x3} y = {y3}')
                break

            x_iter.append(x3)
            y_iter.append(y3)

            # shuffle two lists with same order
            temp = list(zip(x_iter, y_iter))
            random.shuffle(temp)
            x_iter, y_iter = zip(*temp)  # returns tuples
            x_iter, y_iter = list(x_iter), list(y_iter)

            # remove max value and update array
            ind_max = np.argmax(y_iter)
            x_iter.pop(ind_max)
            y_iter.pop(ind_max)

            if num == num_max:
                print(f'Max iterations {num_max} reached')
                break

            num += 1

        return x3, y3
    