import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
plt.style.use('style_file.mplstyle')


log_x = False
log_y = True

N = 17
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

delta = 0.0  # from KH analysis -- leave at 0
C1 = 0.62
C2 = 0.33
kb = 1.24  # value of C1 to use whenever using eq32
ch = 1.66  # value of 1/C2 to use sometimes?

# ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
#                          np.linspace(0.05, 0.275, num=50, endpoint=False)),
#                np.linspace(0.275, 0.6, num=50))
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
# ks = np.append(np.geomspace(1e-6, 0.1, num=20, endpoint=False), np.linspace(0.1, 2.0, num=20))

# Set up the array of rs to solve for, and thus the R0s
rs = np.linspace(1/49, 1, num=49, endpoint=False)
# r = (R0 - 1.0) / ((1.0 / tau) - 1)
R0s1 = rs*((1/tau1) - 1) + 1
R0s2 = rs*((1/tau2) - 1) + 1

lamhats1, l2hats1 = np.transpose([fingering_modes.gaml2max(Pr1, tau1, R0) for R0 in R0s1])
lhats1 = np.sqrt(l2hats1)
lamhats2, l2hats2 = np.transpose([fingering_modes.gaml2max(Pr2, tau2, R0) for R0 in R0s2])
lhats2 = np.sqrt(l2hats2)

# get results of parasite models with T and C
print("--- starting results_withTC for RGB ---")
results_withTC1 = [parasite_model.results_vs_r0(R0s1, hb, Pr1, tau1, DB1, ks, N, lamhats1, l2hats1, withTC=True, Sam=True, C1=C1, C2=C2) for hb in HBs1]
print("--- starting results_hydro_withTC for RGB ---")
results_hydro_withTC1 = parasite_model.results_vs_r0(R0s1, 0.0, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, withTC=True, Sam=True, C1=C1, C2=C2)
# results of old parasite models
results_HG191 = [parasite_model.results_vs_r0(R0s1, hb, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, eq32=True, C1=kb) for hb in HBs1]
results_brown1 = parasite_model.results_vs_r0(R0s1, 0.0, Pr1, tau1, 1.0, ks, N, lamhats1, l2hats1, eq32=True, C1=kb)

# repeat for WD case
print("--- starting results_withTC for WD ---")
results_withTC2 = [parasite_model.results_vs_r0(R0s2, hb, Pr2, tau2, DB2, ks, N, lamhats2, l2hats2, withTC=True, Sam=True, C1=C1, C2=C2) for hb in HBs2]
print("--- starting results_hydro_withTC for WD ---")
results_hydro_withTC2 = parasite_model.results_vs_r0(R0s2, 0.0, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, withTC=True, Sam=True, C1=C1, C2=C2)

results_HG192 = [parasite_model.results_vs_r0(R0s2, hb, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, eq32=True, C1=kb) for hb in HBs2]
results_brown2 = parasite_model.results_vs_r0(R0s2, 0.0, Pr2, tau2, 1.0, ks, N, lamhats2, l2hats2, eq32=True, C1=kb)

scale = 0.8
plt.figure(figsize=(12.8 * scale, 4.8 * scale))

field = "NuC"

plt.subplot(1, 2, 1)
plt.plot(R0s1, results_brown1[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='grey', label=r'HG19')
plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='grey', label=r'New Model')
plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='k', label=r'$B = 0G$')
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
plt.ylabel(r'$D_C/\kappa_C$')
plt.legend(fontsize='small', ncol=2)

plt.subplot(1, 2, 2)
plt.plot(R0s2, results_brown2[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
plt.plot(R0s2, results_hydro_withTC2[field] - 1, 'o-', c='k', label=r'$B = 0G$')
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
plt.ylabel(r'$D_C/\kappa_C$')

plt.tight_layout()
plt.savefig('figures/Fig8_NuC_N17.pdf')

# Uncomment the following if you want to make a heat flux figure too
# plt.close()
# plt.figure(figsize=(12.8 * scale, 4.8 * scale))

# field = "NuT"

# plt.subplot(1, 2, 1)
# # plt.plot(R0s1, results_brown1[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
# plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='grey', label=r'HG19')
# # plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='grey', label=r'New Model')
# # plt.plot(R0s1, results_hydro_withTC1[field] - 1, 'o-', c='k', label=r'$B = 0G$')
# plt.plot(R0s1, results_withTC1[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
# plt.plot(R0s1, results_withTC1[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
# plt.plot(R0s1, results_HG191[0][field] - 1, ':', c='C2')
# plt.plot(R0s1, results_HG191[1][field] - 1, ':', c='C0')
# plt.title(r'RGB\\($\tau = 10^{-7}$, $\mathrm{Pr} = 10^{-6}$, $\mathrm{Pm} = 0.1$)')
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau1))
# # plt.ylim(ymin=1e-3)
# plt.xlabel(r'$R_0$')
# # plt.ylabel(r'$|\hat{F}_C|$')
# plt.ylabel(r'$D_T/\kappa_T$')
# plt.legend(fontsize='small', ncol=2)

# plt.subplot(1, 2, 2)
# # plt.plot(R0s2, results_brown2[field] - 1, '--', c='grey', label=r'Brown et al.~2013')
# # plt.plot(R0s2, results_hydro_withTC2[field] - 1, 'o-', c='k', label=r'$B = 0G$')
# plt.plot(R0s2, results_withTC2[0][field] - 1, 'o-', c='C2', label=r'$B = 100G$')
# plt.plot(R0s2, results_withTC2[1][field] - 1, 'o-', c='C0', label=r'$B = 1000G$')
# plt.plot(R0s2, results_HG192[0][field] - 1, ':', c='C2')
# plt.plot(R0s2, results_HG192[1][field] - 1, ':', c='C0')
# plt.title(r'WD\\($\tau = 10^{-3}$, $\mathrm{Pr} = 10^{-3}$, $\mathrm{Pm} = 1.0$)')
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau2))
# # plt.ylim(ymin=1e-3)
# plt.xlabel(r'$R_0$')
# # plt.ylabel(r'$|\hat{F}_C|$')
# plt.ylabel(r'$D_T/\kappa_T$')

# plt.tight_layout()
# plt.savefig('figures/Fig8_NuT_N17_nohydro.pdf')