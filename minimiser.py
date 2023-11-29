"""
Minimisers
"""

import numpy as np

import load_data as ld
import function as fc


class Univariate():

    def __init__(self):
        pass
    
    def min_indices(self, y_list):
        """
        Calculate the indices of the smallest three points around the minimum
        """

        indices = (y_list.argsort()[:3])  # extra indices of the three points around the minimum

        return indices

    def cal_x3(self, x0, x1, x2, y0, y1, y2):
        """
        Calculate second order Lagrange polynomial minimum
        """

        numer = (x2**2-x1**2)*y0 + (x0**2-x2**2)*y1 + (x1**2-x0**2)*y2
        denom = (x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2
        x3 = (1/2) * (numer/denom)

        return x3

    def parabolic_1d(self, x_list, y_list):
        """
        1D parabolic minimiser
        """

        # inital values of the three points around first minimum
        indices = self.min_indices(y_list)
        x0, x1, x2 = x_list[indices[0]], x_list[indices[1]], x_list[indices[2]]
        y0, y1, y2 = y_list[indices[0]], y_list[indices[1]], y_list[indices[2]]

        # iterate
        flag = True
        stop_cond = 1e-10
        max_num = 100
        num = 1
        while flag:
            x3 = self.cal_x3(x0, x1, x2, y0, y1, y2)
            list = []
        return 0