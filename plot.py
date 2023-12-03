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
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.bar(energy, data_osc, width, align="edge", color="C1", alpha=0.9,
               label="Observed (oscillated)")
        ax.bar(0, 0, color="C4", label="Simulated (not osicllated)")
        
        ax2.bar(energy, data_unosc, width, align="edge", color="C4", alpha=0.75,
                label="Simulated (not osicllated)")

        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel(r"# of entries (oscillated $\nu_\mu$)", color="C1")
        ax2.set_ylabel(r"# of entries (non-oscillated)", color="C4")

        ax.tick_params(axis="y", colors="C1")
        ax2.tick_params(axis="y", colors="C4")
        ax.yaxis.label.set_color('C1')
        ax2.yaxis.label.set_color("C4")

        ax.legend()
        ax.grid(lw=0.4)
        plt.show()


def neutrino_prob(energy):
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
    plt.plot(theta_list, nll_list, '.', label=f'dm2 = {dm2} e-3')
    plt.ylabel('NLL')
    plt.xlabel('theta')
    plt.legend()
    plt.show()
    

def nll_1d_dm2(dm2_list, nll_list, theta):
    fig = plt.figure()
    plt.plot(dm2_list, nll_list, '.', label=f'theta = {theta}')
    plt.ylabel('NLL')
    plt.xlabel('dm2')
    plt.legend()
    plt.show()


def nll_2d_theta_dm2(theta_list, dm2_list, nll_list, plot=True):
    if plot:
        # possible colors: "Reds", "viridis", "bone", "nipy_spectral", "gist_ncar", "jet"
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

            fig.show()
