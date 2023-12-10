"""
Plot functions
"""

import io
import numpy as np
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


import function as fc
import load_data as ld


def data(energy, data_osc, data_unosc, width):

    # plot energy against raw data, oscillated and unoscillated
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


def data_aligned(energy, data_osc, data_unosc, width, dir_folder, plot=True):

    # plot energy against raw data, oscillated and unoscillated, and align their peaks
    if plot:
        col_ax = 'C2'
        col_ax2 = 'C4'

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.bar(energy, data_osc, width, align="edge", color=col_ax, alpha=0.9,
               label="Observed (oscillated)")
        ax.bar(1, 0, color=col_ax2, label="Simulated (unoscillated)")
        
        ax2.bar(energy, data_unosc, width, align="edge", color=col_ax2, alpha=0.75)

        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"# of entries (oscillated $\nu_\mu$)", color=col_ax)
        ax2.set_ylabel(r"# of entries (non-oscillated)", color=col_ax2)

        ax.tick_params(axis="y", colors=col_ax)
        ax2.tick_params(axis="y", colors=col_ax2)
        ax.yaxis.label.set_color(col_ax)
        ax2.yaxis.label.set_color(col_ax2)

        ax.legend()
        ax.grid(lw=0.4)

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/data_aligned_1D.tiff")
        png1.close()

        plt.show()


def data_aligned_2D(theta, dm2, dir_folder, plot=True):

    # plot energy against raw data, oscillated and unoscillated, and align their peaks
    if plot:
        # load data
        data = ld.LoadData()
        data_osc, data_unosc, energy, width = data.get_data()  # energy (GeV) in each bin

        # colour
        col_ax = 'C2'
        col_ax2 = 'C4'

        fig, ax = plt.subplots()

        ax.bar(energy, data_osc, width, align="edge", color=col_ax, alpha=0.9,
               label="Observed")

        prob = fc.neutrino_prob(E=energy, theta=theta, dm2=dm2)
        data_unosc_prob = data_unosc*prob

        ax.bar(energy, data_unosc_prob, width, align="edge", color=col_ax2, alpha=0.75,
               label="Expected")

        ax.bar(10, 10, color='w',
               label=r"$\theta_{23}$ = " + f'{theta:.3f}' + r" $[rad]$" + "\n" + \
                r"$\Delta m^2_{23}$ = " + f'{dm2*1e3:.3f}' + r" $[10^{-3} \/ \/ eV^2]$")

        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"# of $\nu_\mu$ entries")

        ax.legend()
        ax.grid(lw=0.4)

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/data_aligned_2D.tiff")
        png1.close()

        plt.show()

        # Goodness of fit test
        doff = len(data_osc) - 2  # degree of freedom
        chi2 = fc.chi2(data_osc, data_unosc_prob, doff)  # chi squared value
        p = 1 - stats.chi2.cdf(chi2, doff)  # p value
        print('\nGoodness of fit test:')
        print(f'chi2 = {chi2} with {doff} doff and p = {p}')
        print(f'reduced chi2 = {chi2 / doff}')
        print(f'p < 0.05: {p < 0.05}')


def data_aligned_3D(theta, dm2, alpha, dir_folder, plot=True):

    # plot energy against raw data, oscillated and unoscillated, and align their peaks
    if plot:
        # load data
        data = ld.LoadData()
        data_osc, data_unosc, energy, width = data.get_data()  # energy (GeV) in each bin

        # colour
        col_ax = 'C2'
        col_ax2 = 'C4'

        fig, ax = plt.subplots()

        ax.bar(energy, data_osc, width, align="edge", color=col_ax, alpha=0.9,
               label="Observed")

        prob = fc.neutrino_prob(E=energy, theta=theta, dm2=dm2)
        data_unosc_prob = data_unosc * energy * prob * alpha

        ax.bar(energy, data_unosc_prob, width, align="edge", color=col_ax2, alpha=0.75,
               label="Expected")

        ax.bar(10, 10, color='w',
               label=r"$\theta_{23}$ = " + f'{theta:.3f}' + r" $[rad]$" + "\n" + \
                r"$\Delta m^2_{23}$ = " + f'{dm2*1e3:.3f}' + r" $[10^{-3} \/ \/ eV^2]$" + "\n" + \
                r"$\alpha$ = " + f'{alpha:.3f}' + r" $[GeV^{-1}]$")

        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"# of $\nu_\mu$ entries")

        ax.legend()
        ax.grid(lw=0.4)

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/data_aligned_3D.tiff")
        png1.close()

        plt.show()

        # Goodness of fit test
        doff = len(data_osc) - 2  # degree of freedom
        chi2 = fc.chi2(data_osc, data_unosc_prob, doff)  # chi squared value
        p = 1 - stats.chi2.cdf(chi2, doff)  # p value
        print('\nGoodness of fit test:')
        print(f'chi2 = {chi2} with {doff} doff and p = {p}')
        print(f'reduced chi2 = {chi2 / doff}')
        print(f'p < 0.05: {p < 0.05}')


def neutrino_prob_sing(energy, dir_folder, plot=True):

    # plot neutrino probability against energy
    if plot:
        fig = plt.figure()
        p = plt.plot(energy, fc.neutrino_prob(E=energy))

        props = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor=p[0].get_color())
        text = r"$\theta_{23}$ = $\pi/4$" + r" $[rad]$" + "\n" + \
               r"$\Delta m^2_{23}$" + \
               r"= 2.4" + r" $[10^{-3} \/ \/ eV^2]$"
        plt.annotate(text, (0.6, 0.09), xycoords="axes fraction", size=12, bbox=props)

        plt.xlabel("Energy [GeV]")
        plt.ylabel(r"Survival Probability $\nu_\mu\rightarrow\nu_\mu$")

        plt.grid(lw=0.4)

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/neutrino_probability.tiff")
        png1.close()

        plt.show()


def neutrino_prob_mul(energy):

    # plot neutrino probability against energy with different theta, dm2, L values
    fig = plt.figure()
    plt.plot(energy, fc.neutrino_prob(E=energy), label='default params')
    plt.plot(energy, fc.neutrino_prob(E=energy, theta=np.pi/8), '-.', label='halve theta')
    plt.plot(energy, fc.neutrino_prob(E=energy, dm2=2.4e-3/2), ':', label='halve del_m')
    plt.plot(energy, fc.neutrino_prob(E=energy, L=295/2), '--', label='halve L')
    plt.ylabel('prob')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()


def data_with_prob(energy, data_unosc_prob, data_unosc, data_osc, width):

    # plot number of entries for oscillated, unoscillated, and unoscillated x neutrino_prob against energy
    fig = plt.figure()
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob\n(default params)')
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(energy, data_unosc, width, color='C1', label='unoscillated')
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob\n(default params)')
    plt.bar(energy, data_osc, width, color='C4', label='oscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(energy, data_unosc_prob, width, color='C2', label='unoscillated x prob\n(default params)')
    plt.bar(energy, data_osc, width, color='C4', label='oscillated', alpha=0.7)
    plt.ylabel('# of entries')
    plt.xlabel('energy (GeV)')
    plt.legend()
    plt.show()


def nll_1d_theta(
        min_func,
        theta_min, theta_max,
        num,
        dm2=2.4
        ):

    # generate data to plot
    theta_list = np.linspace(theta_min, theta_max, num)  # list of theta values

    nll_list = []
    for theta in theta_list:  # calculate NLL for each theta
        nll = min_func.cal_nll(theta)
        nll_list += [nll]

    nll_list = np.array(nll_list)

    # plot theta against nll with fixted theta
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
    
    # generate data to plot
    dm2_list = np.linspace(dm2_min, dm2_max, num)  # list of theta values

    nll_list = []
    for dm2 in dm2_list:  # calculate NLL for each dm2
        nll = min_func.cal_nll([theta_min, dm2])
        nll_list += [nll]

    nll_list = np.array(nll_list)

    # plot dm2 against nll with fixed dm2
    fig = plt.figure()
    plt.plot(dm2_list, nll_list, label=f'theta = {theta_min}')
    plt.ylabel('NLL')
    plt.xlabel('dm2')
    plt.legend()
    plt.show()


def nll_1d_alpha(
        min_func,
        alpha_min, alpha_max,
        num,
        dir_folder,        
        theta=np.pi/4, dm2=2.4e-3,
        plot=True
        ):
    
    if plot:
        # generate data to plot
        alpha_list = np.linspace(alpha_min, alpha_max, num)  # list of theta values

        nll_list = []
        for alpha in alpha_list:  # calculate NLL for each dm2
            nll = min_func.cal_nll([theta, dm2, alpha])
            nll_list += [nll]

        nll_list = np.array(nll_list)

        # plot alpha against nll with fixed alpha
        fig = plt.figure()
        p = plt.plot(alpha_list, nll_list)

        props = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor=p[0].get_color())
        text = r"$\theta_{23}$ = $\pi/4$" + r" $[rad]$" + "\n" + \
               r"$\Delta m^2_{23}$" + \
               r"= 2.4" + r" $[10^{-3} \/ \/ eV^2]$"
        plt.annotate(text, (0.6, 0.09), xycoords="axes fraction", size=12, bbox=props)

        plt.ylabel('Negative Log Likelihood')
        plt.xlabel(r'$\alpha$' + r" $[GeV^{-1}]$")

        plt.grid(lw=0.4)

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/nll_vs_alpha.tiff")
        png1.close()

        plt.show()


def change_nll(err_list, label, stop=1e-10):
    
    # plot the abs change in NLL (current - previous iteration) against iteration number
    fig = plt.figure()
    plt.semilogy(np.arange(1, len(err_list)+1), err_list, label='Optimise ' + label)
    plt.plot([1, len(err_list)], [stop, stop], label='Stopping condition')
    plt.ylabel('Absolute change in NLL\n(current - previous iteration)')
    plt.xlabel('Iteration number')
    plt.legend()
    plt.show()


def nll_2d_theta_dm2(min_func, N, theta_list, dm2_list, dir_folder):
    
    # generate data to plot
    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # possible colors: "YlGnBu", "Reds", "viridis", "bone", "nipy_spectral", "gist_ncar", "jet"
    for i in ["nipy_spectral"]:
        # plot colours
        fig, axes = plt.subplots(2, 1, figsize=(7, 9), gridspec_kw={'height_ratios': [2, 1]})
        ax0 = axes[0]
        ax1 = axes[1]
        
        # big plot for large range
        cntr0 = ax0.contourf(theta_list, dm2_list, nll_list, 300, cmap=i)
        ax0.set_xlabel(r"$\theta_{23}$ $[rad]$")
        ax0.set_ylabel(r"$\Delta m_{23}^2$ $[eV^2]$")
        ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
        # zoomed in small plot around global minima
        plt.subplot(2, 1, 2)
        ax1.contourf(theta_list, dm2_list, nll_list, 300, cmap=i)
        ax1.set_xlabel(r"$\theta_{23}$ $[rad]$")
        ax1.set_ylabel(r"$\Delta m_{23}^2$ $[eV^2]$")
        ax1.annotate ("b)", (-0.15, 1.10), xycoords="axes fraction")
                    
        ax1.set_xlim(0.6, 0.95)
        ax1.set_ylim(1.5e-3, 3.5e-3)
        
        plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)

        fig.colorbar(cntr0, ax=axes, label="Negative Log Likelihood")

        png1 = io.BytesIO()
        plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                    bbox_inches='tight')
        png2 = Image.open(png1)
        png2.save(dir_folder + f"/nll_vs_theta-dm2.tiff")
        png1.close()

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

    # generate data to plot
    theta_list = np.linspace(theta_low, theta_high, N)
    dm2_list = np.linspace(dm2_low, dm2_high, N)

    nll_list = np.zeros((N, N))
    for i in range(len(theta_list)):
        for j in range(len(dm2_list)):
            nll = min_func.cal_nll([theta_list[i], dm2_list[j]])
            nll1 = 1. * nll
            nll_list[j][i] = nll1

    nll_list = np.array(nll_list)

    # plot colours
    fig, ax1 = plt.subplots(figsize=(6, 4))
    cntr1 = ax1.contourf(theta_list, dm2_list, nll_list, levels=500, cmap='nipy_spectral')

    if plot_points:
        ax1.plot(theta_all, dm2_all, '.', color='cyan', label='Points of parabola')
    ax1.plot(theta_min, dm2_min, 'x', color='red', label='Minimum')

    ax1.set_xlabel(r"$\theta_{23}$ $[rad]$")
    ax1.set_ylabel(r"$\Delta m_{23}^2$ $[eV^2]$")

    # plot contours
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


def fit_MC(var_list, var, N, dir_folder=None, var_string='', std_ratio=0.005, Lorentz=False, FSA=False, plot=False):

    # parameters for histogram
    number, edges = np.histogram(var_list, bins=100)
    num = number/number.sum()  # normalised number
    bincenters = 0.5 * (edges[1:] + edges[:-1])
    widths = edges[1:] - edges[:-1]

    # fit Gaussian or Lorentzian
    ind_max = np.argmax(num)
    centre_guess = bincenters[ind_max]
    std_guess = centre_guess * std_ratio
    x_fit = np.linspace(edges[0], edges[-1], 1000)  # generate data to plot
    if Lorentz and FSA:
        print('Lorentzian fit')
        fit, cov = curve_fit(fc.Lorentzian, bincenters, num, [max(num), centre_guess, std_guess])
        y_fit = fc.Lorentzian(x_fit, *fit)  # generate data to plot
    else:
        print('Gaussian fit')
        fit, cov = curve_fit(fc.Gaussian, bincenters, num, [max(num), centre_guess, std_guess])
        y_fit = fc.Gaussian(x_fit, *fit)  # generate data to plot

    # print fit results
    centre = fit[1]
    std = np.sqrt(cov[1, 1])
    print(f'Fit centre = {centre:.7f} +/- {std:.7f}')
    print(f'with % error +/- {(std*100/centre):.3f}')

    # plot
    if plot:
        plt.figure(1, figsize=(4,3))

        if Lorentz and FSA:
            label1 = 'FSA'
            label2 = 'Lorentzian fit'
        elif FSA:
            label1 = 'FSA'
            label2 = 'Gaussian fit'
        else:
            label1 = 'CSA'
            label2 = 'Gaussian fit'

        plt.bar(bincenters, num, widths, label='Run ' + label1 + f' {N} times')

        plt.plot(x_fit, y_fit, color='C1', label=label2)

        totalweight = num.sum()
        print('\nTotal weight =', totalweight)

        plt.xlabel(var)
        plt.ylabel('Frequency of events')
        plt.xticks(rotation=25)
        plt.legend()
    
        if dir_folder is not None:
            png1 = io.BytesIO()
            plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                        bbox_inches='tight')
            png2 = Image.open(png1)
            png2.save(dir_folder + f"/MC-{label1}-{var_string}.tiff")
            png1.close()

        plt.show()

    return centre, std
