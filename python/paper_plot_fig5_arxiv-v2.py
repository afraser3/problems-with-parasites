import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import parasite_model
import OUTfile_reader
plt.style.use('style_file.mplstyle')


log_x = False
log_y = True

N = 17  # need to check this often. 33 is plenty for Pr=0.01, but not for 0.001 w/ HB=10
Pr = 1e-1
tau = 1e-1
HB = 0.1
delta = 0.0  # from KH analysis -- leave at 0
Pms = np.array([0.01, 0.1, 1.0])  # magnetic Prandtl number
DBs = Pr / Pms
C1 = 0.62
C2 = 0.33
kb = 1.24  # value of C1 to use whenever using eq32
ch = 1.66  # value of 1/C2 to use sometimes?

# ks = np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
#                          np.linspace(0.05, 0.275, num=50, endpoint=False)),
#                np.linspace(0.275, 0.6, num=50))
ks = np.append(np.geomspace(1e-6, 0.1, num=50, endpoint=False), np.linspace(0.1, 2.0))

# Set up the array of R0s to solve for
R0s = np.linspace(1.45, 9.9, num=25, endpoint=True)
R0s_DNS = np.array([[1.5, 3.0, 5.0, 7.0, 9.0], [1.5, 3.0, 5.0, 7.0, 9.0], [1.45, 3.0, 5.0, 7.0, 9.0]])
FCs_DNS = np.zeros((len(Pms), len(R0s_DNS[0])), dtype=np.float64)
FCs_DNS_var = np.zeros_like(FCs_DNS)
for pmi, pm in enumerate(Pms):
    for ri, r0 in enumerate(R0s_DNS[pmi]):
        FCs_DNS[pmi, ri], FCs_DNS_var[pmi, ri] = OUTfile_reader.get_avg_from_DNS(Pr, r0, HB, pm, "flux_Chem", with_variance=True)

R0s_hydro_DNS = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
FCs_hydro_DNS = np.zeros_like(R0s_hydro_DNS)
FCs_hydro_DNS_var = np.zeros_like(R0s_hydro_DNS)
for ri, r0 in enumerate(R0s_hydro_DNS):
    FCs_hydro_DNS[ri], FCs_hydro_DNS_var[ri] = OUTfile_reader.get_avg_from_hydr_DNS(r0, "flux_Chem", with_variance=True)

lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)
# get results of parasite models without T and C
print("--- starting results_noTC ---")
results_noTC = [parasite_model.results_vs_r0(R0s, HB, Pr, tau, db, ks, N, lamhats, l2hats, C1=C1, C2=C2) for db in DBs]
print("--- starting results_hydro_noTC ---")
results_hydro_noTC = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N, lamhats, l2hats, C1=C1, C2=C2)
# get results of parasite models with T and C
print("--- starting results_withTC ---")
results_withTC = [parasite_model.results_vs_r0(R0s, HB, Pr, tau, db, ks, N, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2) for db in DBs]
print("--- starting results_hydro_withTC ---")
results_hydro_withTC = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2)

results_HG19 = parasite_model.results_vs_r0(R0s, HB, Pr, tau, 1.0, ks, N, lamhats, l2hats, eq32=True, C1=kb)
results_brown = parasite_model.results_vs_r0(R0s, 0.0, Pr, tau, 1.0, ks, N, lamhats, l2hats, eq32=True, C1=kb)

lhats_DNS = np.zeros_like(R0s_DNS)
for pmi, pm in enumerate(Pms):
    for ri, r0 in enumerate(R0s_DNS[pmi]):
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, r0)
        lhats_DNS[pmi, ri] = np.sqrt(l2hat)

colors = np.array(['saddlebrown', 'firebrick', 'C1'])
scale = 0.8
plt.figure(figsize=(12.8 * scale, 4.8 * scale))

plt.subplot(1, 2, 1)
plt.plot(R0s, results_brown["FC"], '--', c='grey', label=r'Brown et al.~2013')
plt.plot(R0s, results_hydro_noTC["FC"], ':', c='k', label=r'no $T$, $C$')
plt.plot(R0s, results_hydro_withTC["FC"], '-', c='k', label=r'with $T$, $C$')
plt.errorbar(R0s_hydro_DNS, FCs_hydro_DNS, fmt='x', yerr=FCs_hydro_DNS_var, c='k')
plt.title(r'$H_B = 0$')
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=1e-3)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$|\hat{F}_C|$')
plt.legend(fontsize='small')

plt.subplot(1, 2, 2)
plt.plot(R0s, results_HG19["FC"], '--', c='grey', label=r'HG19')
for pmi, pm in enumerate(Pms):
    plt.plot(R0s, results_noTC[pmi]["FC"], ':', c=colors[pmi])
    plt.plot(R0s, results_withTC[pmi]["FC"], '-', c=colors[pmi], label=r'$\mathrm{{Pm}} = {}$'.format(pm))
    plt.errorbar(R0s_DNS[pmi], FCs_DNS[pmi], fmt='x', yerr=FCs_DNS_var[pmi], c=colors[pmi])
plt.title(r'$H_B = {}$'.format(HB))
if log_x:
    plt.xscale("log")
if log_y:
    plt.yscale("log")
plt.xlim((1.0, 1.0/tau))
plt.ylim(ymin=1e-3)
plt.xlabel(r'$R_0$')
plt.ylabel(r'$|\hat{F}_C|$')
plt.legend(fontsize='small')

plt.tight_layout()
plt.savefig('figures/Fig5.pdf')
