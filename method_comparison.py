"""
Method comparison with known function

Note on parameter names:
- theta = x
- dm2 = y
- nll = z
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import minimiser as mi
import uncertainty as un
import plot as pl_func

plot = False

# minimisation class
min_func = mi.Minimiser()


def function(params):
    X, Y = params[0], params[1]
    Z = (-np.cos(2*np.pi*(X)) * np.cos(np.pi*(Y))) / (1 + np.power(X, 2) + np.power(Y, 2))
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
if True:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.view_init(elev=20., azim=-45)

    # plot surface
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, 
        cmap=cm.jet, linewidth=0.1, edgecolor='k'
        )

    # plot contours 
    ax.contour(X, Y, Z, 7, offset=-1.2, cmap=cm.jet, linestyles="solid")

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
x_guess = -1.5
y_guess = 1.5
T0 = 50
rho = 0.5
step = 5e-1
stop_cond = 1e-10

if True:
    # Monte-Carlo classical simulated annealing
    params, z_min, err_list, params_list = min_func.Monte_Carlo(
        [x_guess, y_guess],
        T0, step, rho,
        num_max=5000, stop_cond=stop_cond,
        function=function,
        method='CSA'
        )

    x_min, y_min = params[0], params[1]
    x_plot, y_plot = params_list[0], params_list[1]

    # plot
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )


print('\n\n- Fast simulated annealing -\n')

# inital guess
x_guess = -1.5
y_guess = 1.5
T0 = 1000
step = 1e-2
stop_cond = 1e-16

if True:
    # Monte-Carlo fast simulated annealing
    params, z_min, err_list, params_list = min_func.Monte_Carlo(
        [x_guess, y_guess],
        T0, step,
        num_max=5e4, stop_cond=stop_cond,
        function=function,
        method='FSA'
        )

    x_min, y_min = params[0], params[1]
    x_plot, y_plot = params_list[0], params_list[1]

    # plot
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )


'''
Gradient descent
'''
print()
print('-'*24)
print('--- Gradient descent ---\n')

# gradient descent
x_guess = -0.25
y_guess = 0.45
stop_cond = 1e-10

params, z_min, err_list, params_list = min_func.gradient_descent(
    [x_guess, y_guess],
    alpha=6e-2,
    alpha_frac=5,
    function=function,
    num_max=100, stop_cond=stop_cond
    )

x_min, y_min = params[0], params[1]
x_plot, y_plot = params_list[0], params_list[1]

# plot
if True:
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )


'''
Quasi-Newton method
'''
print()
print('-'*27)
print('--- Quasi-Newton method ---\n')

# Quasi-Newton
x_guess = -0.15
y_guess = 0.15
stop_cond = 1e-10

params, z_min, err_list, params_list = min_func.quasi_Newton(
    [x_guess, y_guess],
    alpha=1,
    alpha_frac=1.1,
    function=function,
    num_max=100, stop_cond=stop_cond
    )

x_min, y_min = params[0], params[1]
x_plot, y_plot = params_list[0], params_list[1]

# plot
if True:
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )
        

'''
Newton's method
'''
print()
print('-'*23)
print('--- Newton\'s method ---\n')

# Newtons
x_guess = -0.06
y_guess = 0.06
stop_cond = 5e-8

params, z_min, err_list, params_list = min_func.Newtons(
    [x_guess, y_guess],
    function=function,
    num_max=100, stop_cond=stop_cond
    )

x_min, y_min = params[0], params[1]
x_plot, y_plot = params_list[0], params_list[1]

# plot
if True:
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )
        

'''
Univariate method
'''
print()
print('-'*25)
print('--- Univariate method ---\n')

# univariate
x_guess = -0.15
y_guess = 0.2
stop_cond = 1e-10

(x_min, y_min, z_min,
 err_list,
 x_plot, y_plot,
 x_all, y_all) = min_func.univariate(
    [x_guess, y_guess],
    [150, 150],
    function=function,
    num_max=100, stop_cond=stop_cond
    )

# plot
if True:
    if True:
        visualise(X, Y, Z, x_min, y_min, x_plot, y_plot)

    if True:
        pl_func.change_nll(
            err_list,
            label=r"$x$" + ' & ' + r"$y$",
            stop=stop_cond
            )


print('The global minimum known to be at x = 0, y = 0, with z = -1') 
