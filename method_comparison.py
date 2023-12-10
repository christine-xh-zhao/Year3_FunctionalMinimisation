"""
Method comparison with known function
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


import function as fc
import load_data as ld
import minimiser as mi
import uncertainty as un
import plot as pl_func

plot = False

# minimisation class
min_func = mi.Minimiser()


def function(params):
    X, Y = params[0], params[1]
    Z = (-np.cos(2*np.pi*(X)) * np.cos(np.pi*(Y))) / (1 + np.power(X, 2) + np.power(Y, 2))
    # Z = np.sin(np.pi*X)*np.sin(np.pi*Y)
    return Z


def visualise(X, Y, Z, x_min, y_min, x_plot, y_plot):
    # plot colours
    fig, ax1 = plt.subplots(figsize=(6, 4))
    cntr1 = ax1.contourf(X, Y, Z, levels=500, cmap=cm.jet)

    ax1.plot(x_min, y_min, 'x', color='red', label='Minimum')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # plot contours
    cntr1.levels = cntr1.levels.tolist()
    ax1.contour(cntr1, levels=cntr1.levels[1:-1:60], colors='k', alpha=0.5)

    # plot path
    X, Y = x_plot[:-1], y_plot[:-1]
    U = np.subtract(x_plot[1:], x_plot[:-1])
    V = np.subtract(y_plot[1:], y_plot[:-1])
    ax1.quiver(X, Y, U, V, color="white", angles='xy', scale_units='xy', scale=1, label='Step')

    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

    fig.colorbar(cntr1, ax=ax1, label=r"$f \/ (x,y)$")

    ax1.set_xlim(-1.75, 1.75)
    ax1.set_ylim(-1.75, 1.75)
    ax1.legend()
    plt.show()


# data for plotting
delta = 0.05
x = np.arange(-1.75, 1.8, delta)
y = np.arange(-1.75, 1.8, delta)
X, Y = np.meshgrid(x, y)


Z = function([X, Y])


# plot function
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.view_init(elev=20., azim=-45)

    # plot surface
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, 
        cmap=cm.jet, linewidth=0.1, edgecolor='k'
        )

    # plot contours 
    ax.contour(X, Y, Z, 7, offset=-1.2, linewidth=0.5, cmap=cm.jet, linestyles="solid")


    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label=r"$f \/ (x,y)$")

    plt.xticks(np.arange(min(x), max(x), 0.74))
    plt.yticks(np.arange(min(y), max(y), 0.74))
    plt.xlabel('x')
    plt.ylabel('y')

    ax.set_zlim(-1.2, 0.85)

    plt.show()



'''
Monte-Carlo method
- MC is a random process
- If the plots produced are not as expected, run the script few more times
- Especially for CSA, since T decreases too slow, which means the distribution might be significantly messed up
'''
print()
print('-'*26)
print('--- Monte-Carlo method ---')

run_once = False


print('\n- Classical simulated annealing -\n')

# inital guess same as before
theta_guess = -1.5
dm2_guess = 1.5
T0 = 50
rho = 0.5
step = 5e-1
stop_cond = 1e-7

if True:
    # Monte-Carlo classical simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step, rho,
        num_max=5000, stop_cond=stop_cond,
        function=function,
        method='CSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

    # estimate error
    print('\nUncertainties from Hessian')
    un.std_2(function, theta_min, dm2_min)

    # plot
    if True:
        visualise(X, Y, Z, theta_min, dm2_min, theta_plot, dm2_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$",
            stop=stop_cond
            )


print('\n\n- Fast simulated annealing -\n')

# inital guess
theta_guess = -1.5
dm2_guess = 1.5
T0 = 150
step = 1e-2
stop_cond = 1e-7

if True:
    # Monte-Carlo fast simulated annealing
    params, nll_min, err_list, params_list = min_func.Monte_Carlo(
        [theta_guess, dm2_guess],
        T0, step,
        num_max=2e4, stop_cond=stop_cond,
        function=function,
        method='FSA'
        )

    theta_min, dm2_min = params[0], params[1]
    theta_plot, dm2_plot = params_list[0], params_list[1]

    # estimate error
    print('\nUncertainties from Hessian')
    un.std_2(function, theta_min, dm2_min)

    # plot
    if True:
        visualise(X, Y, Z, theta_min, dm2_min, theta_plot, dm2_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$",
            stop=stop_cond
            )


'''
Gradient descent
'''
print()
print('-'*24)
print('--- Gradient descent ---\n')

# gradient descent
theta_guess = -0.25
dm2_guess = 0.25

params, nll_min, err_list, params_list = min_func.gradient_descent(
    [theta_guess, dm2_guess],
    alpha=1e-1,
    function=function,
    num_max=100, stop_cond=1e-10
    )

theta_min, dm2_min = params[0], params[1]
theta_plot, dm2_plot = params_list[0], params_list[1]

# estimate error
print('\nUncertainties from Hessian')
un.std_2(min_func.cal_nll, theta_min, dm2_min)

# plot
if True:
    if True:
        visualise(X, Y, Z, theta_min, dm2_min, theta_plot, dm2_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$\theta_{23}$" + ' & ' + r"$\Delta m_{23}^2$",
            stop=stop_cond
            )