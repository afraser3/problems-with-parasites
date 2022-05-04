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
HB = 0.1
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
    results_DNS = np.zeros_like(R0s_DNS)
    for ri, r0 in enumerate(R0s_DNS):
        results_DNS[ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, Pm, DNS_name)
        if plot_quantity == "NuC":
            results_DNS[ri] = results_DNS[ri] * (r0 / tau) + 1.0
        if plot_quantity == "NuT":
            results_DNS[ri] = results_DNS[ri] + 1.0
if compare_hydro_DNS:
    R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    results_hydro_DNS = OUTfile_reader.fluxes_nusselts_wrms_hydr_DNS()[hydro_DNS_entry]

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
if compare_modified_HG19:
    # get results of parasite models
    results = parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, CH=1.66)
if compare_eq32:
    results_eq32 = parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=True)
if compare_hydro:
    results_hydro = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=True)


scale = 0.8
plt.figure(figsize=(6.4 * scale, 4.8 * scale))
if compare_modified_HG19:
    plt.plot(R0s, results[plot_quantity], '-', c='C0', label=r'$H_B = {}$'.format(HB))
if compare_DNS:
    plt.plot(R0s_DNS, results_DNS, 'x', c='C0', label='DNS')
if compare_eq32:
    plt.plot(R0s, results_eq32[plot_quantity], '--', c='C0', label=r'HG19 eq 32')
if compare_hydro:
    plt.plot(R0s, results_hydro[plot_quantity], '-', c='k', label='hydro')
if compare_hydro_DNS:
    plt.plot(R0s_hydro_DNS, results_hydro_DNS, '.', c='k')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.xlabel(r'$R_0$')
plt.ylabel(plot_quantity)
plt.legend()

plt.tight_layout()
plt.show()
