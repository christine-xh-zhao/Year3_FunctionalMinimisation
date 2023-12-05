"""
Plot functions
"""

import numpy as np
import matplotlib.pyplot as plt

import function as fc


def data(energy, data_osc, data_unosc, width):
    fig = plt.figure()
    plt.bar(energy, data_osc, width, label='oscillated')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, label='unoscillated', color='C1')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, label='unoscillated', color='C1')
    plt.bar(energy, data_osc, width, label='oscillated', color='C0')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()

    plt.show()


def data_aligned(energy, data_osc, data_unosc, width, plot=True):
    if plot:
        col_ax = 'C2'
        col_ax2 = 'C4'

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.bar(energy, data_osc, width, align="edge", color=col_ax, alpha=0.9,
               label="Observed (oscillated)")
        ax.bar(0, 0, color=col_ax2, label="Simulated (not osicllated)")
        
        ax2.bar(energy, data_unosc, width, align="edge", color=col_ax2, alpha=0.75,
                label="Simulated (not osicllated)")

        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"# of entries (oscillated $\nu_\mu$)", color=col_ax)
        ax2.set_ylabel(r"# of entries (non-oscillated)", color=col_ax2)

        ax.tick_params(axis="y", colors=col_ax)
        ax2.tick_params(axis="y", colors=col_ax2)
        ax.yaxis.label.set_color(col_ax)
        ax2.yaxis.label.set_color(col_ax2)

        ax.legend()
        ax.grid(lw=0.4)
        plt.show()


def neutrino_prob_sing(energy, plot=True):
    if plot:
        fig = plt.figure()
        p = plt.plot(energy, fc.neutrino_prob(E=energy))

        props = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor=p[0].get_color())
        text = r"$\theta_{23}$ = $\pi/4$" + "\n" + \
               r"$\Delta m^2_{23}$" + \
               r"= 2.4" + r" $10^{-3} eV^2$"
        plt.annotate (text, (0.7, 0.1), xycoords="axes fraction", size=15, bbox=props)

        plt.xlabel("Energy [GeV]")
        plt.ylabel(r"Survival Probability $\nu_\mu\rightarrow\nu_\mu$")

        plt.grid(lw=0.4)
        plt.show()


def neutrino_prob_mul(energy):
    fig = plt.figure()
    plt.plot(energy, fc.neutrino_prob(E=energy), label='default params')
    plt.plot(energy, fc.neutrino_prob(E=energy, theta=np.pi/8), '-.', label='halve theta')
    plt.plot(energy, fc.neutrino_prob(E=energy, dm2=2.4/2), ':', label='halve del_m')
    plt.plot(energy, fc.neutrino_prob(E=energy, L=295/2), '--', label='halve L')
    plt.ylabel('prob')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()


def data_with_prob(energy, data_unosc_prob, data_unosc, data_osc, width):
    fig = plt.figure()
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, color='C1', label='unoscillated')
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob')
    plt.bar(energy, data_osc, width, color='C4', label='oscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob')
    plt.bar(energy, data_osc, width, color='C4', label='oscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()


def nll_1d_theta(theta_list, nll_list, dm2=2.4):
    fig = plt.figure()
    plt.plot(theta_list, nll_list, label=f'dm2 = {dm2} e-3')
    plt.ylabel('NLL')
    plt.xlabel('theta')
    plt.legend()
    plt.show()
    

def nll_1d_dm2(
        min_func,
        dm2_min, dm2_max,
        num,
        theta_min
        ):
    
    dm2_list = np.linspace(dm2_min, dm2_max, num)  # list of theta values

    nll_list = []
    for dm2 in dm2_list:  # calculate NLL for each dm2
        nll = min_func.cal_nll([theta_min, dm2])
        nll_list += [nll]

    nll_list = np.array(nll_list)

    fig = plt.figure()
    plt.plot(dm2_list, nll_list, label=f'theta = {theta_min}')
    plt.ylabel('NLL')
    plt.xlabel('dm2')
    plt.legend()
    plt.show()


def change_nll(err_list, label):
    fig = plt.figure()
    plt.semilogy(np.arange(1, len(err_list)+1), err_list, label='Optimise ' + label)
    plt.plot([1, len(err_list)], [1e-10, 1e-10], label='Stopping condition')
    plt.ylabel('Change in NLL\n(current - previous iteration)')
    plt.xlabel('Iteration number')
    plt.legend()
    plt.show()


def nll_2d_theta_dm2(min_func, N, theta_list, dm2_list):
    
    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # possible colors: "YlGnBu", "Reds", "viridis", "bone", "nipy_spectral", "gist_ncar", "jet"
    for i in ["nipy_spectral"]:

        fig, axes = plt.subplots(2, 1, figsize=(7, 9), gridspec_kw={'height_ratios': [2, 1]})
        ax0 = axes[0]
        ax1 = axes[1]
        
        cntr0 = ax0.contourf(theta_list, dm2_list, nll_list, 300, cmap=i)
        ax0.set_xlabel(r"$\theta_{23}$ $[rad]$")
        ax0.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3}\/ \/eV^2]$")
        ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
        plt.subplot(2, 1, 2)

        cntr1 = ax1.contourf(theta_list, dm2_list, nll_list, 300, cmap=i)
        ax1.set_xlabel(r"$\theta_{23}$ $[rad]$")
        ax1.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3}\/ \/eV^2]$")
        ax1.annotate ("b)", (-0.15, 1.00), xycoords="axes fraction")
                    
        ax1.set_xlim(0.65, 0.9)
        ax1.set_ylim(1.5, 3.5)
        
        plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

        fig.colorbar(cntr0, ax=axes, label="Negative Log Likelihood")

        plt.show()


def visual_one_method(
        min_func,
        N,
        theta_low, theta_high,
        dm2_low, dm2_high,
        theta_min, dm2_min,
        theta_plot, dm2_plot,
        theta_all=[], dm2_all=[],
        plot_points=False
        ):

    theta_list = np.linspace(theta_low, theta_high, N)
    dm2_list = np.linspace(dm2_low, dm2_high, N)

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    cntr1 = ax1.contourf(theta_list, dm2_list, nll_list, levels=500, cmap='nipy_spectral')

    if plot_points:
        ax1.plot(theta_all, dm2_all, '.', color='cyan', label='Points of parabola')
    ax1.plot(theta_min, dm2_min, 'x', color='red', label='Minimum')

    ax1.set_xlabel(r"$\theta_{23}$ $[rad]$")
    ax1.set_ylabel(r"$\Delta m_{23}^2$ $[10^{-3}\/ \/eV^2]$")

    cntr1.levels = cntr1.levels.tolist()
    ax1.contour(cntr1, levels=cntr1.levels[1:30:8], colors='w', alpha=0.5)
    ax1.contour(cntr1, levels=cntr1.levels[40:-1:42], colors='w', alpha=0.5)

    # plot path
    X, Y = theta_plot[:-1], dm2_plot[:-1]
    U = np.subtract(theta_plot[1:], theta_plot[:-1])
    V = np.subtract(dm2_plot[1:], dm2_plot[:-1])
    ax1.quiver(X, Y, U, V, color="white", angles='xy', scale_units='xy', scale=1, label='Step')

    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

    fig.colorbar(cntr1, ax=ax1, label="Negative Log Likelihood")
    
    ax1.legend()
    plt.show()
