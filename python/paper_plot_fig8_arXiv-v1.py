"""
Compares FC vs wrms via equation 27 of HG19, each saved to a .txt file using extract_and_save_data_from_DNS.py
"""
import numpy as np
import fingering_modes
import parasite_model
import OUTfile_reader
from matplotlib import pyplot as plt
plt.style.use('style_file.mplstyle')

Pr = 1e-1
tau = 1e-1

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

symbols = ['X', 'X']
# colors = ['C0', 'C1']
colors = np.array([['limegreen', 'firebrick'], ['C0', 'C1']])
scale = 0.8
plt.figure(figsize=(12.8 * scale, 4.8 * scale))#, constrained_layout=True)

plt.subplot(1, 2, 1)
for hbi, HB in enumerate([0.01, 0.1]):
    for pmi, Pm in enumerate([0.1, 1.0]):
        fname = 'extracted_data/Pr{}_HB{}_Pm{}_R0scan_data.txt'.format(Pr, HB, Pm)

        data = np.loadtxt(fname)
        R0s = data[:, 0]
        FC = data[:, 6]
        wrms = data[:, 10]

        lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])

        Eq27_coeff = 1.24 / (R0s * (lamhats + tau*l2hats))

        plt.loglog(Eq27_coeff * wrms**2.0, FC, symbols[pmi], c=colors[pmi, hbi], label=r'$H_B = {}, \mathrm{{Pm}} = {}$'.format(HB, Pm))
plt.loglog(Eq27_coeff * wfs_hydro_DNS**2.0, FCs_hydro_DNS, '.', c='k', label=r'hydro')
plt.loglog([5e-3, 10], [5e-3, 10], '--', c='k')
# plt.xlim(xmin=0)
# plt.ylim(ymin=0)
# plt.legend()
plt.xlabel(r'$-C_1 \hat{u}_{z, \mathrm{rms}}^2/[R_0(\hat{\lambda}_f + \tau \hat{l}_f^2)]$')
plt.ylabel(r'$\hat{F}_C$')


########
log_x = False
log_y = True
compare_modified_HG19 = True
compare_eq32 = False
compare_hydro = True
compare_DNS = True
compare_hydro_DNS = True


N = 17  # need to check this often. 33 is plenty for Pr=0.01, but not for 0.001 w/ HB=10
HB1 = 0.01
HB2 = 0.1
delta = 0.0  # from KH analysis -- probably leave at 0
Pms = np.array([0.1, 1.0])  # magnetic Prandtl number
DBs = Pr / Pms
ch = 1.66

ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=50))

# Set up the array of R0s or rs to solve for
# R0s = np.linspace(1.45, 9.5, num=50, endpoint=True)
# R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)

if compare_DNS:  # get results of DNS
    # R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])  # Pm = 0.1 and Pm = 1
    R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])  # Pm = 0.1 and Pm = 1
    wfs_DNS1 = np.zeros((len(Pms), len(R0s_DNS[0])), dtype=np.float64)
    wfs_DNS_var1 = np.zeros_like(wfs_DNS1)
    wfs_DNS2 = np.zeros((len(Pms), len(R0s_DNS[0])), dtype=np.float64)
    wfs_DNS_var2 = np.zeros_like(wfs_DNS2)
    for pmi, pm in enumerate(Pms):
        for ri, r0 in enumerate(R0s_DNS[pmi]):
            wfs_DNS1[pmi, ri], wfs_DNS_var1[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB1, pm, "uzrms", with_variance=True)
            wfs_DNS2[pmi, ri], wfs_DNS_var2[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB2, pm, "uzrms", with_variance=True)

# if compare_hydro_DNS:
    # R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
    # wfs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
    # wfs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
    # for ri, r0 in enumerate(R0s_hydro_DNS):
        # wfs_hydro_DNS[ri], wfs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "uzrms", with_variance=True)

lamhats1, l2hats1 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s_DNS[0]])
lhats1 = np.sqrt(l2hats1)
lamhats2, l2hats2 = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s_DNS[1]])
lhats2 = np.sqrt(l2hats2)
if compare_modified_HG19:
    # get results of parasite models
    results1 = [parasite_model.results_vs_r0(R0s_DNS[0], HB1, Pr, tau, db, ks, N, lamhats1, l2hats1, CH=ch) for db in DBs]
    results2 = [parasite_model.results_vs_r0(R0s_DNS[1], HB2, Pr, tau, db, ks, N, lamhats2, l2hats2, CH=ch) for db in DBs]
if compare_eq32:
    results_eq32_1 = parasite_model.results_vs_r0(R0s_DNS[0], HB1, Pr, tau, 1.0, ks, N, lamhats1, l2hats1, eq32=True)
    results_eq32_2 = parasite_model.results_vs_r0(R0s_DNS[1], HB2, Pr, tau, 1.0, ks, N, lamhats2, l2hats2, eq32=True)
if compare_hydro:
    results_hydro = parasite_model.results_vs_r0(R0s_DNS[0], 0.0, Pr, tau, 1.0, ks, N, lamhats1, l2hats1, eq32=True)

lhats_DNS = np.zeros_like(R0s_DNS)
for pmi, pm in enumerate(Pms):
    for ri, r0 in enumerate(R0s_DNS[pmi]):
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, r0)
        lhats_DNS[pmi, ri] = np.sqrt(l2hat)

plt.subplot(1, 2, 2)
for pmi, pm in enumerate(Pms):
    plt.loglog(results1[pmi]["wf"]**2.0, wfs_DNS1[pmi]**2.0, symbols[pmi], c=colors[pmi, 0])
    plt.loglog(results2[pmi]["wf"]**2.0, wfs_DNS2[pmi]**2.0, symbols[pmi], c=colors[pmi, 1])
plt.loglog(results_hydro["wf"]**2.0, wfs_hydro_DNS**2.0, '.', c='k')
# plt.loglog([3e-2, 3e0], [3e-2, 3e0], '--', c='k')
plt.loglog([1e-3, 3e0], [1e-3, 3e0], '--', c='k')
plt.xlabel(r'$\hat{w}_f^2$')
plt.ylabel(r'$\hat{u}^2_{z, \mathrm{rms}}$')
# plt.legend()

########
plt.tight_layout()
# plt.show()
# plt.savefig('HG19_Eq27_verification.pdf')
plt.savefig('figures/fig8b.pdf')
