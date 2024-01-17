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
Pm4 = 1.0
Pm5 = 1.0
Pm6 = 1.0
DB1 = Pr / Pm1
DB2 = Pr / Pm2
DB3 = Pr / Pm3
DB4 = Pr / Pm4
DB5 = Pr / Pm5
DB6 = Pr / Pm6
# for each Pm, these are the HBs I want to plot
HB1 = 0.1
HBs2 = [0.01, 0.1]
# HBs3 = [0.01, 0.1, 1.0]
HBs3 = [0.01, 0.1]
HB4 = 1.0
HB5 = 10.0
HB6 = 100.0
C1 = 0.62
C2 = 0.33

ks_noTC = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                              np.linspace(0.05, 0.275, num=50, endpoint=False)),
                    np.linspace(0.275, 0.6, num=50))
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))
# ks_noTC = np.append(np.append(np.linspace(0.0025, 0.05, num=10, endpoint=False),
#                               np.linspace(0.05, 0.275, num=20, endpoint=False)),
#                     np.linspace(0.275, 0.6, num=20))
# ks = np.append(np.geomspace(1e-6, 0.1, num=20, endpoint=False), np.linspace(0.1, 2.0, num=20))

# Set up the array of R0s to solve for
# R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)
# R0s = np.linspace(1.45, 9.9, num=5, endpoint=True)
R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])
R0s4 = np.array([1.45, 3.0, 5.0, 7.0])
R05 = 1.45
R06 = 1.45
# R0s_DNS_hydro = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
FCs_DNS1 = np.zeros((len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var1 = np.zeros_like(FCs_DNS1)
FCs_DNS2 = np.zeros((len(HBs2), len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var2 = np.zeros_like(FCs_DNS2)
FCs_DNS3 = np.zeros((len(HBs3), len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var3 = np.zeros_like(FCs_DNS3)
FCs_DNS4 = np.zeros((len(R0s4)), dtype=np.float64)
FCs_DNS_var4 = np.zeros_like(FCs_DNS4)

FTs_DNS1 = np.zeros((len(R0s_DNS[0])), dtype=np.float64)
FTs_DNS_var1 = np.zeros_like(FCs_DNS1)
FTs_DNS2 = np.zeros((len(HBs2), len(R0s_DNS[0])), dtype=np.float64)
FTs_DNS_var2 = np.zeros_like(FCs_DNS2)
FTs_DNS3 = np.zeros((len(HBs3), len(R0s_DNS[0])), dtype=np.float64)
FTs_DNS_var3 = np.zeros_like(FCs_DNS3)
FTs_DNS4 = np.zeros((len(R0s4)), dtype=np.float64)
FTs_DNS_var4 = np.zeros_like(FCs_DNS4)

for ri in range(len(R0s_DNS[0])):
    FCs_DNS1[ri], FCs_DNS_var1[ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[0, ri], HB1, Pm1, "flux_Chem", with_variance=True)
    FTs_DNS1[ri], FTs_DNS_var1[ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[0, ri], HB1, Pm1, "flux_Temp", with_variance=True)
    for hbi, hb in enumerate(HBs2):
        FCs_DNS2[hbi, ri], FCs_DNS_var2[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[1, ri], hb, Pm2, "flux_Chem", with_variance=True)
        FTs_DNS2[hbi, ri], FTs_DNS_var2[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[1, ri], hb, Pm2, "flux_Temp", with_variance=True)
    for hbi, hb in enumerate(HBs3):
        # if ri < 4:  # skip R0=9 because we don't have great data for that run
        if True:
            FCs_DNS3[hbi, ri], FCs_DNS_var3[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[2, ri], hb, Pm3, "flux_Chem", with_variance=True)
            FTs_DNS3[hbi, ri], FTs_DNS_var3[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s_DNS[2, ri], hb, Pm3, "flux_Temp", with_variance=True)
for ri in range(len(R0s4)):
    FCs_DNS4[ri], FCs_DNS_var4[ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s4[ri], HB4, Pm4, "flux_Chem", with_variance=True)
    FTs_DNS4[ri], FTs_DNS_var4[ri] = OUTfile_reader.get_avg_from_DNS(Pr, R0s4[ri], HB4, Pm4, "flux_Temp", with_variance=True)
FCs_DNS5, FCs_DNS_var5 = OUTfile_reader.get_avg_from_DNS(Pr, R05, HB5, Pm5, "flux_Chem", with_variance=True)
FTs_DNS5, FTs_DNS_var5 = OUTfile_reader.get_avg_from_DNS(Pr, R05, HB5, Pm5, "flux_Temp", with_variance=True)
FCs_DNS6, FCs_DNS_var6 = OUTfile_reader.get_avg_from_DNS(Pr, R06, HB6, Pm6, "flux_Chem", with_variance=True)
FTs_DNS6, FTs_DNS_var6 = OUTfile_reader.get_avg_from_DNS(Pr, R06, HB6, Pm6, "flux_Temp", with_variance=True)

R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
FCs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
FCs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
FTs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
FTs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
for ri, r0 in enumerate(R0s_hydro_DNS):
    FCs_hydro_DNS[ri], FCs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "flux_Chem", with_variance=True)
    FTs_hydro_DNS[ri], FTs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "flux_Temp", with_variance=True)

lamhats1, l2hats1 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s_DNS[0]])
lhats1 = np.sqrt(l2hats1)
lamhats2, l2hats2 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s_DNS[1]])
lhats2 = np.sqrt(l2hats2)
lamhats3, l2hats3 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s_DNS[2]])
lhats3 = np.sqrt(l2hats3)
lamhats4, l2hats4 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s4])
lhats4 = np.sqrt(l2hats4)
lamhat5, l2hat5 = fingering_modes.gaml2max(Pr, tau, R05)
lhat5 = np.sqrt(l2hat5)
lamhat6, l2hat6 = fingering_modes.gaml2max(Pr, tau, R06)
lhat6 = np.sqrt(l2hat6)

# get results of parasite models with T and C
print("Pm1")
results_withTC1 = parasite_model.results_vs_r0(R0s_DNS[0], HB1, Pr, tau, DB1, ks, N_withTC, lamhats1, l2hats1, C1=C1, C2=C2, withTC=True, Sam=True)
print("Pm2")
results_withTC2 = [parasite_model.results_vs_r0(R0s_DNS[1], hb, Pr, tau, DB2, ks, N_withTC, lamhats2, l2hats2, C1=C1, C2=C2, withTC=True, Sam=True) for hb in HBs2]
print("Pm3")
results_withTC3 = [parasite_model.results_vs_r0(R0s_DNS[2], hb, Pr, tau, DB3, ks, N_withTC, lamhats3, l2hats3, C1=C1, C2=C2, withTC=True, Sam=True) for hb in HBs3]
print("Pm4")
results_withTC4 = parasite_model.results_vs_r0(R0s4, HB4, Pr, tau, DB4, ks, N_withTC, lamhats4, l2hats4, C1=C1, C2=C2, withTC=True, Sam=True)
# parasite_results(R0, HB, Pr, tau, DB, ks, N, lamhat, l2hat,
#                      eq32=False, double_N=False, delta=0.0, ideal=False, badks_exception=True,
#                      withTC=False, Sam=False, C1=1.24, C2=1/1.66):
results_withTC5 = parasite_model.parasite_results(R05, HB5, Pr, tau, DB5, ks, N_withTC, lamhat5, l2hat5, withTC=True, Sam=True, C1=C1, C2=C2)
results_withTC6 = parasite_model.parasite_results(R06, HB6, Pr, tau, DB6, ks, N_withTC, lamhat6, l2hat6, withTC=True, Sam=True, C1=C1, C2=C2)
print("--- starting results_hydro_withTC ---")
results_hydro_withTC = parasite_model.results_vs_r0(R0s_DNS[0], 0.0, Pr, tau, 1.0, ks, N_withTC, lamhats1, l2hats1, withTC=True, Sam=True, C1=C1, C2=C2)

FCmin = min(FCs_hydro_DNS[-1], results_hydro_withTC['FC'][-1])
FTmin = min(FTs_hydro_DNS[-1], results_hydro_withTC['FT'][-1])
FCmax = max(FCs_DNS6, results_withTC6['FC'])
FTmax = max(FTs_DNS6, results_withTC6['FT'])

# colors = np.array(['saddlebrown', 'firebrick', 'C1'])
scale = 0.8
plt.figure(figsize=(12.8 * scale, 4.8 * scale))
# plt.figure(figsize=(15 * scale, 12 * scale))
color1 = 'saddlebrown'
colors2 = ['darkblue', 'firebrick']
colors3 = ['C0', 'C1', 'C2']

plt.subplot(1, 2, 1)
plt.plot([FCmin, FCmax], [FCmin, FCmax], '--', c='k')
plt.plot([FCmin, FCmax], [2*FCmin, 2*FCmax], '--', c='grey')
plt.plot([FCmin, FCmax], [FCmin/2, FCmax/2], '--', c='grey')
plt.scatter(FCs_DNS1, results_withTC1['FC'])
for hbi in range(len(HBs2)):
    plt.scatter(FCs_DNS2[hbi], results_withTC2[hbi]['FC'])
for hbi in range(len(HBs3)):
    plt.scatter(FCs_DNS3[hbi], results_withTC3[hbi]['FC'])
plt.scatter(FCs_DNS4, results_withTC4['FC'])
plt.scatter(FCs_hydro_DNS, results_hydro_withTC['FC'])
plt.scatter(FCs_DNS5, results_withTC5['FC'])
plt.scatter(FCs_DNS6, results_withTC6['FC'])
plt.xscale("log")
plt.yscale("log")

plt.subplot(1, 2, 2)
plt.plot([FTmin, FTmax], [FTmin, FTmax], '--', c='k')
plt.plot([FTmin, FTmax], [2*FTmin, 2*FTmax], '--', c='grey')
plt.plot([FTmin, FTmax], [FTmin/2, FTmax/2], '--', c='grey')
plt.scatter(FTs_DNS1, results_withTC1['FT'])
for hbi in range(len(HBs2)):
    plt.scatter(FTs_DNS2[hbi], results_withTC2[hbi]['FT'])
for hbi in range(len(HBs3)):
    plt.scatter(FTs_DNS3[hbi], results_withTC3[hbi]['FT'])
plt.scatter(FTs_DNS4, results_withTC4['FT'])
plt.scatter(FTs_hydro_DNS, results_hydro_withTC['FT'])
plt.scatter(FTs_DNS5, results_withTC5['FT'])
plt.scatter(FTs_DNS6, results_withTC6['FT'])
plt.xscale("log")
plt.yscale("log")

plt.savefig("figures/test_fig6.pdf")
plt.show()

# plt.subplot(2, 2, 1)
# plt.plot(R0s, results_hydro_noTC['FC'], c='k', label=r'$H_B = 0$')
# plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
# plt.plot(R0s, results_noTC1['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
# plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
# for hbi, hb in enumerate(HBs2):
#     plt.plot(R0s, results_noTC2[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
#     plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
# for hbi, hb in enumerate(HBs3):
#     plt.plot(R0s, results_noTC3[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = 1$, $H_B = {}$'.format(hb))
#     if hbi < 2:  # skip R0=9 for HB=1
#         plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
#     else:
#         plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau))
# plt.ylim(ymin=3e-4)
# # plt.xlabel(r'$R_0$')
# plt.ylabel(r'$|\hat{F}_C|$')
# # plt.legend(fontsize='small', ncol=2, columnspacing=0.5)
# plt.legend(fontsize='small')
# plt.title(r'No $T$ or $C$ fields, $C_2 = 0.33$')
#
# plt.subplot(2, 2, 2)
# plt.plot(R0s, results_hydro_withTC['FC'], c='k', label=r'$H_B = 0$')
# plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
# plt.plot(R0s, results_withTC1['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
# plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
# for hbi, hb in enumerate(HBs2):
#     plt.plot(R0s, results_withTC2[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
#     plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
# for hbi, hb in enumerate(HBs3):
#     plt.plot(R0s, results_withTC3[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
#     if hbi < 2:  # skip R0=9 for HB=1
#         plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
#     else:
#         plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau))
# plt.ylim(ymin=3e-4)
# # plt.xlabel(r'$R_0$')
# # plt.ylabel(r'$|\hat{F}_C|$')
# # plt.legend(fontsize='small')
# plt.title(r'With $T$ and $C$ fields, $C_2 = 0.33$')
#
# plt.subplot(2, 2, 3)
# plt.plot(R0s, results_hydro_withTC_c['FC'], c='k', label=r'$H_B = 0$')
# plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
# plt.plot(R0s, results_withTC1_c['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
# plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
# for hbi, hb in enumerate(HBs2):
#     plt.plot(R0s, results_withTC2_c[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
#     plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
# for hbi, hb in enumerate(HBs3):
#     plt.plot(R0s, results_withTC3_c[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
#     if hbi < 2:  # skip R0=9 for HB=1
#         plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
#     else:
#         plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau))
# plt.ylim(ymin=3e-4)
# plt.xlabel(r'$R_0$')
# plt.ylabel(r'$|\hat{F}_C|$')
# # plt.legend(fontsize='small')
# plt.title(r'With $T$ and $C$ fields, $C_2 = 0.8$')
#
# plt.subplot(2, 2, 4)
# plt.plot(R0s, results_hydro_withTC_d['FC'], c='k', label=r'$H_B = 0$')
# plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
# plt.plot(R0s, results_withTC1_d['FC'], c=color1, label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm1, HB1))
# plt.errorbar(R0s_DNS[0], FCs_DNS1, fmt='x', yerr=FCs_DNS_var1, c=color1)
# for hbi, hb in enumerate(HBs2):
#     plt.plot(R0s, results_withTC2_d[hbi]['FC'], c=colors2[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm2, hb))
#     plt.errorbar(R0s_DNS[1], FCs_DNS2[hbi], fmt='x', yerr=FCs_DNS_var2[hbi], c=colors2[hbi])
# for hbi, hb in enumerate(HBs3):
#     plt.plot(R0s, results_withTC3_d[hbi]['FC'], c=colors3[hbi], label=r'$\mathrm{{Pm}} = {}$, $H_B = {}$'.format(Pm3, hb))
#     if hbi < 2:  # skip R0=9 for HB=1
#         plt.errorbar(R0s_DNS[2, :-1], FCs_DNS3[hbi, :-1], fmt='x', yerr=FCs_DNS_var3[hbi, :-1], c=colors3[hbi])
#     else:
#         plt.errorbar(R0s_DNS[2], FCs_DNS3[hbi], fmt='x', yerr=FCs_DNS_var3[hbi], c=colors3[hbi])
# if log_x:
#     plt.xscale("log")
# if log_y:
#     plt.yscale("log")
# plt.xlim((1.0, 1.0/tau))
# plt.ylim(ymin=3e-4)
# plt.xlabel(r'$R_0$')
# # plt.ylabel(r'$|\hat{F}_C|$')
# # plt.legend(fontsize='small')
# plt.title(r'With $T$ and $C$ fields, $C_2 = 0.2$')
#
# plt.tight_layout()
# plt.savefig('figures/Fig5.pdf')
# plt.show()
