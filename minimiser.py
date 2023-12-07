"""
Minimisers
"""

import random
import numpy as np

import load_data as ld
import function as fc
import uncertainty as un


class Minimiser():

    def __init__(self):
        data_osc, data_unosc, energy, _ = ld.LoadData().get_data()  # energy (GeV) in each bin
        
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
                    theta = params[0]; dm2 = params[1], alpha = params[2]
                    prob = fc.neutrino_prob(E=self.energy, theta=theta, dm2=dm2)
                    data_unosc_prob = self.data_unosc*prob
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
                print(f'Minimum of nll = {nll_min} is at\ntheta = {xmin}\ndm2 = {ymin}')
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
                print(f'Minimum of nll = {nll_min} is at\ntheta = {xmin}\ndm2 = {ymin}')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached')
                break

            num += 1
            nll_old = 1. * nll_min

        return xmin, ymin, nll_min, err_list, x_min, y_min, x_all, y_all

    def Newtons(self, theta0, dm20, num_max=100, stop_cond=1e-10):
        """
        Newton's method
        """

        # initalise
        params = np.array([theta0, dm20])
        nll = self.cal_nll(params)

        theta_list = []
        dm2_list = []
        theta_list += [theta0]
        dm2_list += [dm20]

        err_list = []

        # iterate
        num = 1
        while True:
            # grad of function
            grad = un.gradient(f=self.cal_nll, x=params)

            # inverse hessian
            hes = un.hessian(func=self.cal_nll, x=params)
            det_hes = np.linalg.det(hes)  # ensure Hessian is positive definite
            if det_hes <= 0:
                print('Determinant of Hessian <= 0')
            hes_inv = np.linalg.inv(hes)

            # update parameters and nll
            params_new = params - np.dot(hes_inv, grad)
            nll_new = self.cal_nll(params_new)
            theta_list += [params_new[0]]
            dm2_list += [params_new[1]]

            # error
            err = abs(nll_new - nll)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum of nll = {nll_new} is at\ntheta = {params_new[0]}\ndm2 = {params_new[1]}')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            num += 1
            params = 1. * params_new
            nll = 1. * nll_new

        return params_new[0], params_new[1], nll_new, err_list, theta_list, dm2_list

    def quasi_Newton(self, theta0, dm20, alpha, num_max=100, stop_cond=1e-10):
        """
        Quasi-Newton method
        """

        # initalise
        params = np.array([theta0, dm20])
        nll = self.cal_nll(params)

        N = len(params)
        G = np.identity(N)
        grad = un.gradient(f=self.cal_nll, x=params) * alpha

        theta_list = []
        dm2_list = []
        theta_list += [theta0]
        dm2_list += [dm20]

        err_list = []

        # iterate
        num = 1
        while True:
            # inverse hessian
            hes_inv = 1. * G

            # update parameters and nll
            params_new = params - np.dot(hes_inv, grad)
            nll_new = self.cal_nll(params_new)
            theta_list += [params_new[0]]
            dm2_list += [params_new[1]]

            # error
            err = abs(nll_new - nll)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum of nll = {nll_new} is at\ntheta = {params_new[0]}\ndm2 = {params_new[1]}')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            # update G
            grad_new = un.gradient(f=self.cal_nll, x=params_new)

            gamma = grad_new - grad
            delta = params_new - params

            gamma_out = np.outer(gamma, gamma)
            delta_out = np.outer(delta, delta)
            denom1 = np.dot(gamma, delta)
            numer2 = np.dot(G, np.dot(gamma_out, G))
            denom2 = np.dot(gamma, np.dot(G, gamma))

            G = G + delta_out/denom1 - numer2/denom2
            
            # update other parameters
            num += 1
            params = 1. * params_new
            nll = 1. * nll_new
            grad = 1. * grad_new

        return params_new[0], params_new[1], nll_new, err_list, theta_list, dm2_list

    def gradient_descent(self, theta0, dm20, alpha, num_max=100, stop_cond=1e-10):
        """
        Gradient descent
        """

        # initalise
        params = np.array([theta0, dm20])
        nll = self.cal_nll(params)

        theta_list = []
        dm2_list = []
        theta_list += [theta0]
        dm2_list += [dm20]

        err_list = []

        # iterate
        num = 1
        while True:
            # gradient of function
            grad = un.gradient(f=self.cal_nll, x=params)

            # scale the alpha of dm2 gradient to similar magnitude as theta gradient
            alpha_mat = [[alpha, 0], [0, alpha*1e-5]]

            # update parameters and nll
            params_new = params - np.dot(alpha_mat, grad)
            nll_new = self.cal_nll(params_new)
            theta_list += [params_new[0]]
            dm2_list += [params_new[1]]

            # error
            err = abs(nll_new - nll)
            err_list += [err]

            if err <= stop_cond:
                print(f'Stopping condition {stop_cond} reached after {num} iterations')
                print(f'Minimum of nll = {nll_new} is at\ntheta = {params_new[0]}\ndm2 = {params_new[1]}')
                break

            if num == num_max:
                print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            num += 1
            params = 1. * params_new
            nll = 1. * nll_new

        return params_new[0], params_new[1], nll_new, err_list, theta_list, dm2_list

    def Monte_Carlo(
            self,
            theta_guess, dm2_guess,
            T0, step, rho=0.9,
            num_max=1000, stop_cond=1e-10,
            method='',
            printout=True
            ):
        """
        Monte-Carlo method using classical or fast simulated annealing with k_b = 1

        Select method between 'CSA' and 'FSA'
        """

        # initialise
        theta = theta_guess
        dm2 = dm2_guess
        nll = self.cal_nll([theta, dm2])
        T = T0

        theta_list = []
        dm2_list = []
        nll_list = []
        theta_list += [theta_guess]
        dm2_list += [dm2_guess]
        nll_list += [nll]

        err_list = []

        # iterate
        num = 1

        if method == '':
            raise Exception('method should be specified between \'CSA\' or \'FSA\'')
            
        while True:
            # random float following Gaussian with average 0 and variance of 1
            if method == 'CSA':
                rand1 = np.random.randn()
                rand2 = np.random.randn()
            elif method == 'FSA':
                rand1 = np.random.default_rng().standard_cauchy()
                rand2 = np.random.default_rng().standard_cauchy()       
            
            # update
            theta_new = theta + theta * rand1 * step
            dm2_new = dm2 + dm2 * rand2 * step
            nll_new = self.cal_nll([theta_new, dm2_new])

            # change in energy
            nll_diff = nll_new - nll

            # lowering temperature when iterating more
            if method == 'CSA':
                T = (rho**num) * T0
            elif method == 'FSA':
                T = T0 / num    

            # probability
            if T <= 1e-9:  # T too small will raise division to zero
                if nll_diff <= 0:
                    prob = 1
                else:
                    T = 1e-9
            else: 
                expo = -nll_diff / T
                if expo >= 709:  # max exponent to aviod overflow in np.exp()
                    expo = 709
            prob = np.exp(expo)

            # error
            err = abs(nll_new - nll)
            err_list += [err]

            if err <= stop_cond:
                theta_min, dm2_min, nll_min = theta_list[-1], dm2_list[-1], nll_list[-1]
                
                if printout:
                    print(f'Stopping condition {stop_cond} reached after {num} iterations')
                    print(f'Minimum of nll = {nll_min} is at\ntheta = {theta_min}\ndm2 = {dm2_min}')
                break

            if num == num_max:
                if printout:
                    print(f'Max iterations {num_max} reached with stopping condition {stop_cond}')
                break

            # acceptance condition
            if nll_diff <= 0 or np.random.rand() < prob:
                theta = 1. * theta_new
                dm2 = 1. * dm2_new
                nll = 1. * nll_new
                
                theta_list += [theta]
                dm2_list += [dm2]
                nll_list += [nll]

            num += 1

        theta_min, dm2_min, nll_min = theta_list[-1], dm2_list[-1], nll_list[-1]

        return theta_min, dm2_min, nll_min, err_list, theta_list, dm2_list
    