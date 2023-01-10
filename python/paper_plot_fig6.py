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
compare_modified_HG19 = True
compare_eq32 = False
compare_hydro = False
compare_DNS = True
compare_hydro_DNS = False


N = 17  # need to check this often. 33 is plenty for Pr=0.01, but not for 0.001 w/ HB=10
Pr = 1e-1
tau = 1e-1
HB = 0.1
delta = 0.0  # from KH analysis -- probably leave at 0
# Pms = np.array([0.01, 0.1, 1.0])  # magnetic Prandtl number
Pm = 0.1
DB = Pr / Pm
chs = np.array([0.75, 1.66, 2.5])  # TODO: adjust this to be C2 = 1/Ch to be more consistent with the text

ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))
ks = np.append(np.linspace(0.0005, 0.0025, num=10, endpoint=False), ks)

# Set up the array of R0s or rs to solve for
# R0s = np.linspace(1.45, 9.5, num=50, endpoint=True)
R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)

if compare_DNS:  # get results of DNS
    # R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])  # Pm = 0.1 and Pm = 1
    R0s_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    FCs_DNS = np.zeros((len(R0s_DNS)), dtype=np.float64)
    FCs_DNS_var = np.zeros_like(FCs_DNS)
    wfs_DNS = np.zeros_like(FCs_DNS)
    wfs_DNS_var = np.zeros_like(wfs_DNS)
    # for pmi, pm in enumerate(Pms):
    for ri, r0 in enumerate(R0s_DNS):
        FCs_DNS[ri], FCs_DNS_var[ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, Pm, "flux_Chem", with_variance=True)
        wfs_DNS[ri], wfs_DNS_var[ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, Pm, "uzrms", with_variance=True)

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
if compare_modified_HG19:
    # get results of parasite models
    results = [parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, CH=ch) for ch in chs]

lhats_DNS = np.zeros_like(R0s_DNS)
# for pmi, pm in enumerate(Pms):
for ri, r0 in enumerate(R0s_DNS):
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, r0)
    lhats_DNS[ri] = np.sqrt(l2hat)

# colors = np.array([['midnightblue', 'saddlebrown'], ['darkblue', 'peru'], ['C0', 'C1']])
colors = np.array(['C0', 'firebrick', 'C4', 'C5'])
scale = 0.8
# plt.figure(figsize=(12.8 * scale, 4.8 * scale))#, constrained_layout=True)
plt.figure(figsize=(6.4 * scale, 4.8 * scale))

# plt.subplot(1, 2, 1)
for chi, ch in enumerate(chs):
    if compare_modified_HG19:
        plt.plot(R0s, results[chi]["FC"], '-', c=colors[chi], label=r'Model, $C_2 = {:3.1f}$'.format(1.0/ch))
if compare_DNS:
    plt.errorbar(R0s_DNS, FCs_DNS, fmt='X', yerr=FCs_DNS_var, c='firebrick', label=r'DNS')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=1e-3)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$\hat{F}_C$')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('figures/Fig6.pdf')
