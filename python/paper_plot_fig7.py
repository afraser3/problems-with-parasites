import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
plt.style.use('style_file.mplstyle')


log_x = False
log_y = True

plot_hydro_withTC = False
skip_WD = False
also_plot_heat_flux = False

# N = 17  # number of Fourier modes to include in parasite EVP (positive, negative, and zero mode included)
N = 21
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
# ks = np.append(np.geomspace(1e-8, 0.1, num=100, endpoint=False), np.linspace(0.1, 4.0, num=100))

# RGB parameters
tau1 = 1e-7
Pr1 = 1e-6
Pm1 = 0.1
DB1 = Pr1 / Pm1
HBs1 = [1e-7, 1e-5]

# WD parameters
tau2 = 1e-3
Pr2 = 1e-3
Pm2 = 1.0
DB2 = Pr2 / Pm2
HBs2 = [1e-3, 1e-1]

delta = 0.0  # from KH analysis -- leave at 0, corresponds to finding parasites with x-periodicity matching lambda_f
C1 = 0.62  # for with_TC model
C2 = 0.33  # for with_TC model
# C1 = 0.33
# C2 = 0.62
# C1 = 1.24  # for no_TC model
# C2 = 1/1.66  # for no_TC model
kb = 1.24  # value of C1 to use whenever using eq32 (i.e. the HG19 model and/or the Brown model)

# Set up the array of rs (reduced density ratios) to solve for, and thus the R0s
rs = np.linspace(1/49, 1, num=49, endpoint=False)
# rs = np.linspace(1/49, 1, num=20, endpoint=False)
# r = (R0 - 1.0) / ((1.0 / tau) - 1)
R0s1 = rs*((1/tau1) - 1) + 1
R0s2 = rs*((1/tau2) - 1) + 1

lamhats1, l2hats1 = np.transpose([fingering_modes.gaml2max(Pr1, tau1, R0) for R0 in R0s1])
lhats1 = np.sqrt(l2hats1)
lamhats2, l2hats2 = np.transpose([fingering_modes.gaml2max(Pr2, tau2, R0) for R0 in R0s2])
lhats2 = np.sqrt(l2hats2)

# get results of parasite models with T and C

if plot_hydro_withTC:
    print("--- starting results_hydro_withTC for RGB ---")
    results_hydro_withTC1 = parasite_model.results_vs_r0(R0s1, 0.0, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, withTC=True, Sam=True, C1=C1, C2=C2)
    # results_hydro_withTC1 = parasite_model.results_vs_r0(R0s1, 0.0, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, withTC=False, C1=C1, C2=C2, badks_exception=False)
print("--- starting results_withTC for RGB ---")
results_withTC1 = [parasite_model.results_vs_r0(R0s1, hb, Pr1, tau1, DB1, ks, N, lamhats1, l2hats1, withTC=True, Sam=True, C1=C1, C2=C2) for hb in HBs1]
# results_withTC1 = [parasite_model.results_vs_r0(R0s1, hb, Pr1, tau1, DB1, ks, N, lamhats1, l2hats1, withTC=False, C1=C1, C2=C2) for hb in HBs1]
# results of old parasite models
results_HG191 = [parasite_model.results_vs_r0(R0s1, hb, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, eq32=True, C1=kb) for hb in HBs1]
results_brown1 = parasite_model.results_vs_r0(R0s1, 0.0, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, eq32=True, C1=kb)

# repeat for WD case
if not skip_WD:
    print("--- starting results_withTC for WD ---")
    results_withTC2 = [parasite_model.results_vs_r0(R0s2, hb, Pr2, tau2, DB2, ks, N, lamhats2, l2hats2, withTC=True, Sam=True, C1=C1, C2=C2) for hb in HBs2]
    # results_withTC2 = [parasite_model.results_vs_r0(R0s2, hb, Pr2, tau2, DB2, ks, N, lamhats2, l2hats2, withTC=False, C1=C1, C2=C2) for hb in HBs2]
    if plot_hydro_withTC:
        print("--- starting results_hydro_withTC for WD ---")
        results_hydro_withTC2 = parasite_model.results_vs_r0(R0s2, 0.0, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, withTC=True, Sam=True, C1=C1, C2=C2)

    results_HG192 = [parasite_model.results_vs_r0(R0s2, hb, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, eq32=True, C1=kb) for hb in HBs2]
    results_brown2 = parasite_model.results_vs_r0(R0s2, 0.0, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, eq32=True, C1=kb)

field = "NuC"
if not skip_WD:
    scale = 0.8
    plt.figure(figsize=(12.8 * scale, 4.8 * scale))
    plt.subplot(1, 2, 1)
plt.plot(R0s1, results_brown1[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='grey', label=r'HG19')
if plot_hydro_withTC:
    plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='grey', label=r'New Model')
    plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='k', label=r'$B = 0G$')
else:
    plt.plot(R0s1, results_withTC1[0][field] - 1, 'o-', c='grey', label=r'New Model')
plt.plot(R0s1, results_withTC1[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
plt.plot(R0s1, results_withTC1[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='C2')
plt.plot(R0s1, results_HG191[1][field] - 1, ':', c='C0')
plt.title(r'\begin{center} RGB\\($\tau = 10^{-7}$, $\mathrm{Pr} = 10^{-6}$, $\mathrm{Pm} = 0.1$) \end{center}')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau1))
# plt.ylim(ymin=1e-3)
plt.xlabel(r'$R_0$')
# plt.ylabel(r'$|\hat{F}_C|$')
plt.ylabel(r'$D_C/\kappa_C$')
plt.legend(fontsize='x-small', ncol=2)

if not skip_WD:
    plt.subplot(1, 2, 2)
    plt.plot(R0s2, results_brown2[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
    if plot_hydro_withTC:
        plt.plot(R0s2, results_hydro_withTC2[field] - 1, 'o-', c='k', label=r'$B = 0G$')
    plt.plot(R0s2, results_withTC2[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
    plt.plot(R0s2, results_withTC2[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
    plt.plot(R0s2, results_HG192[0][field] - 1, ':', c='C2')
    plt.plot(R0s2, results_HG192[1][field] - 1, ':', c='C0')
    plt.title(r'\begin{center} WD\\($\tau = 10^{-3}$, $\mathrm{Pr} = 10^{-3}$, $\mathrm{Pm} = 1.0$) \end{center}')
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlim((1.0, 1.0/tau2))
    # plt.ylim(ymin=1e-3)
    plt.xlabel(r'$R_0$')
    # plt.ylabel(r'$|\hat{F}_C|$')
    plt.ylabel(r'$D_C/\kappa_C$')

plt.tight_layout()
# plt.savefig('figures/Fig8_NuC_N21.pdf')
# plt.savefig('figures/Fig7_switch-Cs_longer-ks.pdf')
plt.savefig('figures/Fig7.pdf')

if also_plot_heat_flux:
    plt.close()
    plt.figure(figsize=(12.8 * scale, 4.8 * scale))

    field = "NuT"

    plt.subplot(1, 2, 1)
    # plt.plot(R0s1, results_brown1[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
    plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='grey', label=r'HG19')
    # plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='grey', label=r'New Model')
    # plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='k', label=r'$B = 0G$')
    plt.plot(R0s1, results_withTC1[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
    plt.plot(R0s1, results_withTC1[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
    plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='C2')
    plt.plot(R0s1, results_HG191[1][field] - 1, ':', c='C0')
    plt.title(r'RGB\\($\tau = 10^{-7}$, $\mathrm{Pr} = 10^{-6}$, $\mathrm{Pm} = 0.1$)')
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlim((1.0, 1.0/tau1))
    # plt.ylim(ymin=1e-3)
    plt.xlabel(r'$R_0$')
    # plt.ylabel(r'$|\hat{F}_C|$')
    plt.ylabel(r'$D_T/\kappa_T$')
    plt.legend(fontsize='small', ncol=2)

    plt.subplot(1, 2, 2)
    # plt.plot(R0s2, results_brown2[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
    # plt.plot(R0s2, results_hydro_withTC2[field] - 1, 'o-', c='k', label=r'$B = 0G$')
    plt.plot(R0s2, results_withTC2[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
    plt.plot(R0s2, results_withTC2[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
    plt.plot(R0s2, results_HG192[0][field] - 1, ':', c='C2')
    plt.plot(R0s2, results_HG192[1][field] - 1, ':', c='C0')
    plt.title(r'WD\\($\tau = 10^{-3}$, $\mathrm{Pr} = 10^{-3}$, $\mathrm{Pm} = 1.0$)')
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlim((1.0, 1.0/tau2))
    # plt.ylim(ymin=1e-3)
    plt.xlabel(r'$R_0$')
    # plt.ylabel(r'$|\hat{F}_C|$')
    plt.ylabel(r'$D_T/\kappa_T$')

    plt.tight_layout()
    plt.savefig('figures/Fig7_NuT_N17.pdf')