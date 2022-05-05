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
plot_quantity = "NuC"
DNS_name = "flux_Chem"
hydro_DNS_entry = 2  # 0, 1, 2, 3, 4 for FC, FT, NuC, NuT, or wrms
if plot_quantity == "NuC":
    DNS_name = "flux_Chem"
if plot_quantity == "NuT":
    DNS_name = "flux_Temp"
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
HBs = [0.01, 0.1]
delta = 0.0  # from KH analysis -- probably leave at 0
Pm = 1.0  # magnetic Prandtl number
DB = Pr / Pm

ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))

# Set up the array of R0s or rs to solve for
R0s = np.linspace(1.45, 9.8, num=22, endpoint=True)

if compare_DNS:  # get results of DNS
    if Pm == 1.0:
        R0s_DNS = np.array([1.45, 3.0, 5.0, 7.0, 9.0])
    else:
        R0s_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    results_DNS = np.zeros((len(HBs), len(R0s_DNS)), dtype=np.float64)
    for hbi, hb in enumerate(HBs):
        for ri, r0 in enumerate(R0s_DNS):
            results_DNS[hbi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, hb, Pm, DNS_name)
            if plot_quantity == "NuC":
                results_DNS[hbi, ri] = results_DNS[hbi, ri] * (r0 / tau) + 1.0
            if plot_quantity == "NuT":
                results_DNS[hbi, ri] = results_DNS[hbi, ri] + 1.0
if compare_hydro_DNS:
    R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    results_hydro_DNS = OUTfile_reader.fluxes_nusselts_wrms_hydr_DNS()[hydro_DNS_entry]

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
if compare_modified_HG19:
    # get results of parasite models
    results = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB, ks, N, lamhats, l2hats, CH=1.66) for hb in HBs]
if compare_eq32:
    results_eq32 = [parasite_model.results_vs_r0(R0s, hb, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=True) for hb in HBs]
if compare_hydro:
    results_hydro = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=True)


scale = 0.8
plt.figure(figsize=(6.4 * scale, 4.8 * scale))
for hbi, hb in enumerate(HBs):
    if compare_modified_HG19:
        plt.plot(R0s, results[hbi][plot_quantity], '-', c='C{}'.format(hbi), label=r'$H_B = {}$'.format(hb))
    if compare_DNS:
        plt.plot(R0s_DNS, results_DNS[hbi], 'X', c='C{}'.format(hbi))
    if compare_eq32:
        plt.plot(R0s, results_eq32[hbi][plot_quantity], '--', c='C{}'.format(hbi), label=r'HG19, $H_B = {}$'.format(hb))
if compare_hydro:
    plt.plot(R0s, results_hydro[plot_quantity], '-', c='k', label='B13 (hydro)')
if compare_hydro_DNS:
    plt.plot(R0s_hydro_DNS, results_hydro_DNS, '.', c='k')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\mathrm{Nu}_C$')
plt.legend()

# plt.tight_layout()
# plt.show()
plt.savefig('figures/HG19_vs_DNS_Pm1.pdf', bbox_inches='tight')
