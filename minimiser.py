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

    def cal_nll(self, params):
        """
        Calculate NLL from given theta
        """

        if hasattr(params, '__len__'):
            if len(params) > 1:
                try:
                    theta = params[0]; dm2 = params[1]
                    prob = fc.neutrino_prob(E=self.energy, theta=theta, dm2=dm2)
                    data_unosc_prob = self.data_unosc*prob
                except:
                    pass
                    # theta = params[0]; dm2 = params[1]
                    # prob = fc.neutrino_prob(E=self.energy, theta=theta, dm2=dm2)
                    # data_unosc_prob = self.data_unosc*prob
            else:
                theta = params[0]
                prob = fc.neutrino_prob(E=self.energy, theta=theta)
                data_unosc_prob = self.data_unosc*prob
        else:
            theta = params
            prob = fc.neutrino_prob(E=self.energy, theta=theta)
            data_unosc_prob = self.data_unosc*prob  

        nll = fc.NLL(lamb=data_unosc_prob, m=self.data_osc)

        return nll

    def shuffle(self, x_iter, y_iter):
        """
        Shuffle two lists with same order
        """

        temp = list(zip(x_iter, y_iter))
        random.shuffle(temp)
        x_iter, y_iter = zip(*temp)  # returns tuples
        x_iter, y_iter = list(x_iter), list(y_iter)

        return x_iter, y_iter

    def parabolic_1d(self, x_list, y_list, num_max=100, stop_cond=1e-10):
        """
        1D parabolic minimiser for neutrino_prob function
        """

        # inital values of the three points around first minimum
        ind_min = np.argmin(y_list)
        x0 = x_list[ind_min]
        x1 = x0 + x0/50; x2 = x0 - x0/50

        x_iter = [x0, x1, x2]
        y_iter = [self.cal_nll(x) for x in x_iter]  

        # iterate
        num = 1
        y_old = 0
        err_list = []
        while True:
            # estimate another theta
            x3 = self.cal_x3(x_iter, y_iter)
            y3 = self.cal_nll(x3)

            x_iter.append(x3)
            y_iter.append(y3)

            # shuffle two lists with same order
            x_iter, y_iter = self.shuffle(x_iter, y_iter)

            # remove max value and update array
            ind_max = np.argmax(y_iter)
            x_iter.pop(ind_max)
            y_iter.pop(ind_max)

            # get min theta
            ind_min = np.argmin(y_iter)
            ymin = y_iter[ind_min]
            xmin = x_iter[ind_min]

            # error
            err = abs(ymin - y_old)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum is x = {xmin} y = {ymin}')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            num += 1
            y_old = 1. * ymin

        return xmin, ymin, err_list

    def univariate(
            self,
            theta_guess, dm2_guess,
            gue_step_the=50, gue_step_dm2=150,
            num_max=100, stop_cond=1e-10
            ):
        """
        Univariate method
        """

        # inital values of the three points around first minimum
        x0 = theta_guess
        x1 = x0 + x0/gue_step_the; x2 = x0 - x0/gue_step_the
        y0 = dm2_guess
        y1 = y0 + y0/gue_step_dm2; y2 = y0 - y0/gue_step_dm2

        x_iter = [x0, x1, x2]
        y_iter = [y0, y1, y2]
        nll_x = [self.cal_nll([x, dm2_guess]) for x in x_iter]  
        nll_y = [self.cal_nll([theta_guess, y]) for y in y_iter]  

        # list for storing values for plotting
        xmin = theta_guess
        ymin = dm2_guess
        x_min = []
        y_min = []
        x_min += [xmin]
        y_min += [ymin]
        x_all = []
        y_all = []
        x_all += x_iter
        y_all += y_iter
        err_list = []

        # initalise iterate values
        num = 1
        nll_old = 0
        while True:
            # estimate another theta
            nll_x = [self.cal_nll([x, ymin]) for x in x_iter]
            x3 = self.cal_x3(x_iter, nll_x)
            nll = self.cal_nll([x3, ymin])

            x_iter.append(x3)
            nll_x.append(nll)
            
            # shuffle two lists with same order
            x_iter, nll_x = self.shuffle(x_iter, nll_x)

            # remove max value and update array
            ind_max = np.argmax(nll_x)
            x_iter.pop(ind_max)
            nll_x.pop(ind_max)
            x_all += x_iter
            y_all += y_iter

            # get min theta to do parabolic along dm2
            ind_min = np.argmin(nll_x)
            nll_min = nll_x[ind_min]
            xmin = x_iter[ind_min]
            x_min += [xmin]
            y_min += [ymin]

            # error
            err = abs(nll_old - nll_min)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum of nll = {nll_min} is at\ntheta = {xmin}\ndm2 = {ymin} e-3')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            num += 1
            nll_old = 1. * nll_min

            # estimate another dm2
            nll_y = [self.cal_nll([xmin, y]) for y in y_iter]
            y3 = self.cal_x3(y_iter, nll_y)
            nll = self.cal_nll([xmin, y3])

            y_iter.append(y3)
            nll_y.append(nll)

            # shuffle two lists with same order
            y_iter, nll_y = self.shuffle(y_iter, nll_y)

            # remove max value and update array
            ind_max = np.argmax(nll_y)
            y_iter.pop(ind_max)
            nll_y.pop(ind_max)
            x_all += x_iter
            y_all += y_iter

            # get min theta to do parabolic along dm2
            ind_min = np.argmin(nll_y)
            nll_min = nll_y[ind_min]
            ymin = y_iter[ind_min]
            x_min += [xmin]
            y_min += [ymin]

            # error
            err = abs(nll_old - nll_min)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum of nll = {nll_min} is at\ntheta = {xmin}\ndm2 = {ymin} e-3')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached')
                break

            num += 1
            nll_old = 1. * nll_min

        return xmin, ymin, nll_min, err_list, np.array(x_min), np.array(y_min), np.array(x_all), np.array(y_all)
    