import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import OUTfile_reader
plt.style.use('style_file.mplstyle')


log_x = False
log_y = True

N_noTC = 9
N_withTC = 21
# N_noTC = 9
# N_withTC = 9
Pr = 1e-1
tau = 1e-1
delta = 0.0  # from KH analysis -- leave at 0
# Pms = np.array([0.01, 0.1, 1.0])  # magnetic Prandtl number
# DBs = Pr / Pms
# HBs = [[0.1], [0.01, 0.1], [0.01, 0.1, 1.0]]  # for each Pm, these are the HBs I want to plot
Pm1 = 0.01
Pm2 = 0.1
Pm3 = 1.0
DB1 = Pr / Pm1
DB2 = Pr / Pm2
DB3 = Pr / Pm3
# for each Pm, these are the HBs I want to plot
HB1 = 0.1
HBs2 = [0.01, 0.1]
HBs3 = [0.01, 0.1, 1.0]
C1 = 0.62
C2 = 0.33
C1_c = 3.29  # value of C1 for panel c
C1_d = 0.27
C2_c = 0.8  # value of C2 for panel c
C2_d = 0.2

ks_noTC = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                              np.linspace(0.05, 0.275, num=50, endpoint=False)),
                    np.linspace(0.275, 0.6, num=50))
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
# ks_noTC = np.append(np.append(np.linspace(0.0025, 0.05, num=10, endpoint=False),
#                               np.linspace(0.05, 0.275, num=20, endpoint=False)),
#                     np.linspace(0.275, 0.6, num=20))
# ks = np.append(np.geomspace(1e-6, 0.1, num=20, endpoint=False), np.linspace(0.1, 2.0, num=20))

# Set up the array of R0s to solve for
R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)
# R0s = np.linspace(1.45, 9.9, num=5, endpoint=True)
R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])
# R0s_DNS_hydro = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
FCs_DNS1 = np.zeros((len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var1 = np.zeros_like(FCs_DNS1)
FCs_DNS2 = np.zeros((len(HBs2), len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var2 = np.zeros_like(FCs_DNS2)
FCs_DNS3 = np.zeros((len(HBs3), len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var3 = np.zeros_like(FCs_DNS3)
# FCs_DNS_hydro = np.zeros_like(R0s_DNS_hydro)
# FCs_DNS_hydro_var = np.zeros_like(FCs_DNS_hydro)
# for pmi, pm in enumerate(Pms):
#     for ri, r0 in enumerate(R0s_DNS[pmi]):
#         FCs_DNS[pmi, ri], FCs_DNS_var[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, pm, "flux_Chem", with_variance=True)
for ri in range(len(R0s_DNS[0])):
    FCs_DNS1[ri], FCs_DNS_var1[ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[0, ri], HB1, Pm1, "flux_Chem", with_variance=True)
    for hbi, hb in enumerate(HBs2):
        FCs_DNS2[hbi, ri], FCs_DNS_var2[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[1, ri], hb, Pm2, "flux_Chem", with_variance=True)
    for hbi, hb in enumerate(HBs3):
        if ri < 4 or hbi < 2:  # skip R0=9 (for HB=1) because we don't have great data for that run
            FCs_DNS3[hbi, ri], FCs_DNS_var3[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[2, ri], hb, Pm3, "flux_Chem", with_variance=True)

R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
FCs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
FCs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
for ri, r0 in enumerate(R0s_hydro_DNS):
    FCs_hydro_DNS[ri], FCs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "flux_Chem", with_variance=True)

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
# get results of parasite models without T and C
print("--- starting results_noTC ---")
print("Pm1")
results_noTC1 = parasite_model.results_vs_r0(R0s, HB1, Pr, tau, DB1, ks_noTC, N_noTC, lamhats, l2hats, C1=C1, C2=C2)
print("Pm2")
results_noTC2 = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB2, ks_noTC, N_noTC, lamhats, l2hats, C1=C1, C2=C2) for hb in HBs2]
print("Pm2")
results_noTC3 = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB3, ks_noTC, N_noTC, lamhats, l2hats, C1=C1, C2=C2) for hb in HBs3]
print("--- starting results_hydro_noTC ---")
results_hydro_noTC = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks_noTC, N_noTC, lamhats, l2hats, C1=C1, C2=C2)

# get results of parasite models with T and C
print("--- starting results_withTC, panel b ---")
print("Pm1")
results_withTC1 = parasite_model.results_vs_r0(R0s, HB1, Pr, tau, DB1, ks, N_withTC, lamhats, l2hats, C1=C1, C2=C2, withTC=True, Sam=True)
print("Pm2")
results_withTC2 = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB2, ks, N_withTC, lamhats, l2hats, C1=C1, C2=C2, withTC=True, Sam=True) for hb in HBs2]
print("Pm2")
results_withTC3 = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB3, ks, N_withTC, lamhats, l2hats, C1=C1, C2=C2, withTC=True, Sam=True) for hb in HBs3]
print("--- starting results_hydro_withTC ---")
results_hydro_withTC = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N_withTC, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2)

print("--- starting results_withTC, panel c ---")
print("Pm1")
results_withTC1_c = parasite_model.results_vs_r0(R0s, HB1, Pr, tau, DB1, ks, N_withTC, lamhats, l2hats, C1=C1_c, C2=C2_c, withTC=True, Sam=True)
print("Pm2")
results_withTC2_c = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB2, ks, N_withTC, lamhats, l2hats, C1=C1_c, C2=C2_c, withTC=True, Sam=True) for hb in HBs2]
print("Pm2")
results_withTC3_c = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB3, ks, N_withTC, lamhats, l2hats, C1=C1_c, C2=C2_c, withTC=True, Sam=True) for hb in HBs3]
print("--- starting results_hydro_withTC ---")
results_hydro_withTC_c = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N_withTC, lamhats, l2hats, withTC=True, Sam=True, C1=C1_c, C2=C2_c)

print("--- starting results_withTC, panel d ---")
print("Pm1")
results_withTC1_d = parasite_model.results_vs_r0(R0s, HB1, Pr, tau, DB1, ks, N_withTC, lamhats, l2hats, C1=C1_d, C2=C2_d, withTC=True, Sam=True)
print("Pm2")
results_withTC2_d = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB2, ks, N_withTC, lamhats, l2hats, C1=C1_d, C2=C2_d, withTC=True, Sam=True) for hb in HBs2]
print("Pm2")
results_withTC3_d = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB3, ks, N_withTC, lamhats, l2hats, C1=C1_d, C2=C2_d, withTC=True, Sam=True) for hb in HBs3]
print("--- starting results_hydro_withTC ---")
results_hydro_withTC_d = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N_withTC, lamhats, l2hats, withTC=True, Sam=True, C1=C1_d, C2=C2_d)

# lhats_DNS = np.zeros_like(R0s_DNS)
# for pmi, pm in enumerate(Pms):
#     for ri, r0 in enumerate(R0s_DNS[pmi]):
#         lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, r0)
#         lhats_DNS[pmi, ri] = np.sqrt(l2hat)

# colors = np.array(['saddlebrown', 'firebrick', 'C1'])
scale = 0.8
# plt.figure(figsize=(12.8 * scale, 4.8 * scale))
plt.figure(figsize=(15 * scale, 12 * scale))
# color1 = 'saddlebrown'
# colors2 = ['darkblue', 'firebrick']
# colors3 = ['C0', 'C1', 'C2']
color1 = 'saddlebrown'
colors2 = ['green', 'firebrick']
colors3 = ['C0', 'C1', 'purple']

plt.subplot(2, 2, 1)
plt.plot(R0s, results_hydro_noTC['FC'], c='k', label=r'$H_B = 0$')
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
plt.plot(R0s, results_noTC1['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
for hbi, hb in enumerate(HBs2):
    plt.plot(R0s, results_noTC2[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
    plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
for hbi, hb in enumerate(HBs3):
    plt.plot(R0s, results_noTC3[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = 1$, $H_B = {}$'.format(hb))
    if hbi == 2:  # skip R0=9 for HB=1
        plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
    else:
        plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=3e-4)
# plt.xlabel(r'$R_0$')
plt.ylabel(r'$|\hat{F}_C|$')
# plt.legend(fontsize='small', ncol=2, columnspacing=0.5)
plt.legend(fontsize='small')
plt.title(r'No $T$ or $C$ fields, $C_2 = 0.33$')

plt.subplot(2, 2, 2)
plt.plot(R0s, results_hydro_withTC['FC'], c='k', label=r'$H_B = 0$')
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
plt.plot(R0s, results_withTC1['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
for hbi, hb in enumerate(HBs2):
    plt.plot(R0s, results_withTC2[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
    plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
for hbi, hb in enumerate(HBs3):
    plt.plot(R0s, results_withTC3[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
    if hbi == 2:  # skip R0=9 for HB=1
        plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
    else:
        plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=3e-4)
# plt.xlabel(r'$R_0$')
# plt.ylabel(r'$|\hat{F}_C|$')
# plt.legend(fontsize='small')
plt.title(r'With $T$ and $C$ fields, $C_2 = 0.33$')

plt.subplot(2, 2, 3)
plt.plot(R0s, results_hydro_withTC_c['FC'], c='k', label=r'$H_B = 0$')
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
plt.plot(R0s, results_withTC1_c['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
for hbi, hb in enumerate(HBs2):
    plt.plot(R0s, results_withTC2_c[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
    plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
for hbi, hb in enumerate(HBs3):
    plt.plot(R0s, results_withTC3_c[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
    if hbi == 2:  # skip R0=9 for HB=1
        plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
    else:
        plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=3e-4)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$|\hat{F}_C|$')
# plt.legend(fontsize='small')
plt.title(r'With $T$ and $C$ fields, $C_2 = 0.8$')

plt.subplot(2, 2, 4)
plt.plot(R0s, results_hydro_withTC_d['FC'], c='k', label=r'$H_B = 0$')
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
plt.plot(R0s, results_withTC1_d['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
for hbi, hb in enumerate(HBs2):
    plt.plot(R0s, results_withTC2_d[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
    plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
for hbi, hb in enumerate(HBs3):
    plt.plot(R0s, results_withTC3_d[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
    if hbi == 2:  # skip R0=9 for HB=1
        plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
    else:
        plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=3e-4)
plt.xlabel(r'$R_0$')
# plt.ylabel(r'$|\hat{F}_C|$')
# plt.legend(fontsize='small')
plt.title(r'With $T$ and $C$ fields, $C_2 = 0.2$')

plt.tight_layout()
plt.savefig('figures/Fig5.pdf')
# plt.show()
