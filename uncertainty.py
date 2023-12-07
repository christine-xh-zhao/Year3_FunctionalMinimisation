"""
Calculate uncertainties for estimated parameters
"""

import numpy as np


def list(min_func, theta_min, theta_plus, theta_minus, nll_change):
    """
    Select the closest NLL value from a list

    Args:
    - min_func -- the object contains the function for calculating NLL from theta as its instance
    - theta_min -- estimated theta value, here is the theta giving min NLL
    - theta_plus -- upper bound of possible theta
    - theta_minus -- lower bound of possible theta
    - nll_change -- nll_min + 1, the expected NLL value
    """

    # create lists
    num = 500  # number of values to generate
    theta_plus_list = np.linspace(theta_min, theta_plus, num)  # list of theta values above theta min
    theta_minus_list = np.linspace(theta_minus, theta_min-theta_min/100, num)  # list of theta values below theta min

    for i in range(num):  # calculate NLL for each theta above theta min
        theta = theta_plus_list[i]
        nll = min_func.cal_nll(theta)
        if np.isclose(nll, nll_change):
            print(f'\nUpper theta = {theta} gives roughly nll_min + 1 = {nll_change}')
            print(f'which the % difference from expected nll_change is {(nll - nll_change)*100/nll_change}')
            theta_stdup = theta
            break

    for i in range(num):  # calculate NLL for each theta above theta min
        theta = theta_minus_list[-i]
        nll = min_func.cal_nll(theta)
        if np.isclose(nll, nll_change):
            print(f'\nLower theta = {theta} gives roughly nll_min + 1 = {nll_change}')
            print(f'which the % difference from expected nll_change is {(nll - nll_change)*100/nll_change}')
            theta_stddown = theta
            break

    return theta_stdup, theta_stddown


def secant(min_func, x1, x2, sol_true, stop_err=1e-6, iter_max=100):
    """
    Secant method for root finding to solve theta giving expected NLL

    Args:
    - min_func -- the object contains the function for calculating NLL from theta as its instance
    - x1 -- one bound of possible theta
    - x2 -- the other bound of possible theta
    - sol_true -- nll_min + 1, the expected NLL value
    """
    n = 0
    
    while True: 
        # calculate the intermediate value
        sol1 = min_func.cal_nll(x1)  # use the function for calculating nll from theta
        sol2 = min_func.cal_nll(x2)

        gradient = (sol1 - sol2) / (x1 - x2)
        x_new = x1 + ((sol_true - sol1) / gradient)

        # update the value of interval
        x1 = x2
        x2 = x_new

        # update number of iteration 
        n += 1 

        # compare with true value and stopping condition
        sol_new = min_func.cal_nll(x_new)
        if abs(sol_new - sol_true) < stop_err:
            print("\nRoot of the given equation =", round(x_new, 6)) 
            print(f'i.e. theta = {x_new} giving nll + 1 roughly = {sol_new}')
            print(f'which the % difference from expected nll_change is {(sol_new - sol_true)*100/sol_true}')
            print("No. of iterations =", n) 

            return x_new, sol_new

        # if reaches max iteration number
        if n == iter_max:
            print('Reaches max', iter_max, 'iterations')
            break


def gradient(f, x):
    """
    First order derivative using central difference scheme
    """

    N = x.shape[0]
    gradient = []

    for i in range(N):
        h = abs(x[i]) *  np.finfo(np.float32).eps  # abs(x[i]) * difference between 1.0 and the next smallest representable float larger than 1.0
        xx = 1. * x[i]  # store x[i] but avoid shallow copy

        # lower bound
        x[i] = xx - h
        f1 = f(x)

        # upper bound
        x[i] = xx + h
        f2 = f(x)

        # calculate gradient
        gradient.append((f2 - f1)/(2*h))

        x[i] = xx  # restore x[i]

    return np.array(gradient)


def hessian(func, x):
    """
    Calculate Hessian by calculate the derivative of a derivative using central difference scheme
    """

    N = x.shape[0]
    hessian = np.zeros((N, N))  # initialise

    for i in range(N):
        h = abs(x[i]) *  np.finfo(np.float32).eps
        xx = 1. * x[i]  # store x[i] but avoid shallow copy

        # lower bound
        x[i] = xx - h
        grad1 = gradient(func, x)

        # upper bound
        x[i] = xx + h
        grad2 = gradient(func, x)

        # calculate gradient
        hessian[i, :] = ((grad2 - grad1)/(2*h))

        x[i] = xx  # restore x[i]

    return hessian


def std_2(min_func, theta_min, dm2_min):
    """
    Calculate standard deviation for minimised theta and dm2
    """
    
    hes = hessian(func=min_func.cal_nll, x=np.array([theta_min, dm2_min]))

    hes_inv = np.linalg.inv(hes)  # inverse hessian to get covariance
    sig_theta = np.sqrt(2) * np.sqrt(hes_inv[0][0])  # std is sqrt of covariance diagonal
    sig_dm2 = np.sqrt(2) * np.sqrt(hes_inv[1][1])

    print(f'\ntheta_min = {theta_min:.4f} +/- {sig_theta:.4f}')
    print(f'with % error +/- {(sig_theta*100/theta_min):.2f}')

    print(f'\ndm2_min = {dm2_min:.4f} +/- {sig_dm2:.4f}')
    print(f'with % error +/- {(sig_dm2*100/dm2_min):.2f}')
