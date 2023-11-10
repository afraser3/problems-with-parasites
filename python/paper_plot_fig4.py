"""
This is just plot_parasites_r0_scan.py but with certain things hardcoded to specifically craft
one of the figures in the paper
"""
import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import OUTfile_reader
plt.style.use('style_file.mplstyle')

# choose from ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star"]
# plot_quantity = "NuC"
# DNS_name = "flux_Chem"
# hydro_DNS_entry = 2  # 0, 1, 2, 3, 4 for FC, FT, NuC, NuT, or wrms
# if plot_quantity == "NuC":
    # DNS_name = "flux_Chem"
# if plot_quantity == "NuT":
    # DNS_name = "flux_Temp"
log_x = False
log_y = True
compare_modified_HG19 = False
compare_eq32 = True
compare_hydro = True
compare_DNS = True
compare_hydro_DNS = True


N = 17  # need to check this often. 33 is plenty for Pr=0.01, but not for 0.001 w/ HB=10
Pr = 1e-1
tau = 1e-1
HB = 0.1
delta = 0.0  # from KH analysis -- probably leave at 0
Pms = np.array([0.01, 0.1, 1.0])  # magnetic Prandtl number
DBs = Pr / Pms
ch = 1.66

ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))

# Set up the array of R0s or rs to solve for
# R0s = np.linspace(1.45, 9.5, num=50, endpoint=True)
R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)

if compare_DNS:  # get results of DNS
    R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])  # Pm = 0.1 and Pm = 1
    FCs_DNS = np.zeros((len(Pms), len(R0s_DNS[0])), dtype=np.float64)
    FCs_DNS_var = np.zeros_like(FCs_DNS)
    wfs_DNS = np.zeros_like(FCs_DNS)
    wfs_DNS_var = np.zeros_like(wfs_DNS)
    for pmi, pm in enumerate(Pms):
        for ri, r0 in enumerate(R0s_DNS[pmi]):
            FCs_DNS[pmi, ri], FCs_DNS_var[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, pm, "flux_Chem", with_variance=True)
            wfs_DNS[pmi, ri], wfs_DNS_var[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, pm, "uzrms", with_variance=True)
if compare_hydro_DNS:
    R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    FCs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
    FCs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
    wfs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
    wfs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
    # FCs_hydro_DNS = OUTfile_reader.fluxes_nusselts_wrms_hydr_DNS()[0]
    # wfs_hydro_DNS = OUTfile_reader.fluxes_nusselts_wrms_hydr_DNS()[4]
    for ri, r0 in enumerate(R0s_hydro_DNS):
        FCs_hydro_DNS[ri], FCs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "flux_Chem", with_variance=True)
        wfs_hydro_DNS[ri], wfs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "uzrms", with_variance=True)

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
if compare_modified_HG19:
    # get results of parasite models
    results = [parasite_model.results_vs_r0(R0s, HB, Pr, tau, db, ks, N, lamhats, l2hats, CH=ch) for db in DBs]
if compare_eq32:
    results_eq32 = parasite_model.results_vs_r0(R0s, HB, Pr, tau, 1.0, ks, N, lamhats, l2hats, eq32=True)
if compare_hydro:
    results_hydro = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N, lamhats, l2hats, eq32=True)

lhats_DNS = np.zeros_like(R0s_DNS)
for pmi, pm in enumerate(Pms):
    for ri, r0 in enumerate(R0s_DNS[pmi]):
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, r0)
        lhats_DNS[pmi, ri] = np.sqrt(l2hat)

colors = np.array([['midnightblue', 'saddlebrown'], ['darkblue', 'firebrick'], ['C0', 'C1']])
scale = 0.8
plt.figure(figsize=(12.8 * scale, 4.8 * scale))#, constrained_layout=True)

plt.subplot(1, 2, 1)
if compare_hydro:
    # plt.plot(R0s, results_hydro["FC"], '--', c='grey', label='Parasite model')
    plt.plot(R0s, results_hydro["FC"], '--', c='k', label='Brown et al.~2013')
if compare_hydro_DNS:
    plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='.', yerr=FCs_hydro_DNS_var, c='k', label=r'Hydro')
for pmi, pm in enumerate(Pms):
    if compare_modified_HG19:
        plt.plot(R0s, results[pmi]["FC"], '-', c=colors[pmi, 1], label=r'$H_B = {}, \mathrm{{Pm}} = {}$'.format(HB, pm))
    if compare_DNS:
        plt.errorbar(R0s_DNS[pmi], FCs_DNS[pmi], fmt='X', yerr=FCs_DNS_var[pmi], c=colors[pmi, 1], label=r'$\mathrm{{Pm}} = {}$'.format(pm))
if compare_eq32:
    plt.plot(R0s, results_eq32["FC"], '--', c='grey', label='HG19')  # , label=r'HG19, $H_B = {}$'.format(hb))
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='.', yerr=FCs_hydro_DNS_var, c='k')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
plt.ylabel(r'$|\hat{F}_C|$')
plt.legend(ncol=2, columnspacing=0.25, fontsize='small')

plt.subplot(1, 2, 2)
for pmi, pm in enumerate(Pms):
    #if compare_modified_HG19:
        #plt.plot(R0s, results[pmi]["wf"]**2.0, '-', c=colors[pmi, 1], label=r'$H_B = {}, \mathrm{{Pm}} = {}$'.format(HB, pm))
    if compare_DNS:
        plt.errorbar(R0s_DNS[pmi], pm*wfs_DNS[pmi]/(Pr*lhats_DNS[pmi]), fmt='X', yerr=pm*wfs_DNS_var[pmi]/(Pr*lhats_DNS[pmi]), c=colors[pmi, 1])
# if compare_eq32:
#     for pmi, pm in enumerate(Pms):
#         # plt.plot(R0s, pm*results_eq32["wf"]/(Pr*lhats), '--', c=colors[pmi, 1])  # , label=r'HG19, $H_B = {}$'.format(hb))
#         plt.plot(R0s, pm * results_hydro["wf"] / (Pr * lhats), '--', c=colors[pmi, 1])
#if compare_hydro:
    #plt.plot(R0s, results_hydro["wf"]**2.0, '--', c='k', label='Hydro')
#if compare_hydro_DNS:
    #plt.plot(R0s_hydro_DNS, wfs_hydro_DNS**2.0, '.', c='k')
#if log_x:
    #plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
# plt.ylabel(r'$\mathrm{Pm} \hat{u}_{z, \mathrm{rms}}/(\mathrm{Pr} \hat{l}_f)$')
plt.ylabel(r'$\mathrm{Rm}$')
plt.axhline(1.0, c='k')
# plt.legend()

plt.tight_layout()
# plt.show()
# plt.savefig('figures/HG19_vs_DNS_FC_Rm_Pmscan.pdf')
plt.savefig('figures/Fig4.pdf')
