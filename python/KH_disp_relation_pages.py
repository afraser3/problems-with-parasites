import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import kolmogorov_EVP
import parasite_model
import fingering_modes

Pr = 1e-6
tau = 1e-7
# Pm = Pr / DB
Pm = 0.1
DB = Pr / Pm
### r = (R0-1)/(tau_inv - 1)
### R0 = 1 + r*(tau_inv - 1)
r = 0.9
R0 = 1 + r*(-1 + 1.0/tau)

# Three different resolutions to compare
N1 = 5
N2 = 9
N3 = 17
# HB = 1.0
HB = 1e-7
# screwing around with different ks to scan over
# ks = np.append(np.geomspace(1e-10, 0.025, num=40, endpoint=False), np.append(np.append(np.linspace(0.0025, 0.05, num=20, endpoint=False),
#                          np.linspace(0.05, 0.275, num=50, endpoint=False)),
#                np.linspace(0.275, 1.0, num=100)))
ks = np.append(np.geomspace(1e-10, 0.0025, num=20, endpoint=False), np.append(np.append(np.linspace(0.0025, 0.05, num=10, endpoint=False),
                         np.linspace(0.05, 0.275, num=25, endpoint=False)),
               np.linspace(0.275, 2.0, num=50)))
lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
lhat = np.sqrt(l2hat)

# seems the wf predicted by the Brown and HG19 models are good brackets over which to search
w_Brown = parasite_model.w_f_HG19(Pr, tau, R0, 0.0).root
w_HG19 = parasite_model.w_f_HG19(Pr, tau, R0, HB).root
wfs = np.linspace(w_Brown, w_HG19, num=20)

with PdfPages('KH_growth_scan_Pr{:0.2e}_tau{:0.2e}_HB{:0.2e}_Pm{:0.2e}_R0{:0.2e}.pdf'.format(Pr, tau, HB, Pm, R0)) as pdf:
    for wi, wf in enumerate(wfs):
        gammas1 = kolmogorov_EVP.gamma_over_k_withTC(0.0, wf, HB, DB, Pr, tau, R0, ks, N1, Sam=True)
        gammas2 = kolmogorov_EVP.gamma_over_k_withTC(0.0, wf, HB, DB, Pr, tau, R0, ks, N2, Sam=True)
        gammas3 = kolmogorov_EVP.gamma_over_k_withTC(0.0, wf, HB, DB, Pr, tau, R0, ks, N3, Sam=True)
        # The following commented-out stuff was to test Rich's idea of using the Gershgorin circle theorem to bound the
        # parasite growth rates. Leaving it in for now in case we want to revisit it.
        # N_Sam1 = int((N1 - 1) / 2)
        # N_Sam2 = int((N2 - 1) / 2)
        # N_Sam3 = int((N3 - 1) / 2)
        # A_psi = wf / (2 * lhat)
        # A_T = -lhat * A_psi / (lamhat + l2hat)
        # A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
        # bounds1 = np.zeros_like(gammas1)
        # bounds2 = np.zeros_like(gammas2)
        # bounds3 = np.zeros_like(gammas3)
        # for kzi, kz in enumerate(ks):
        #     Lmat1 = kolmogorov_EVP.Sams_Lmat(N_Sam1, 0, lhat, kz*lhat, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
        #     disk_extents = np.zeros(len(Lmat1[0]), dtype=np.float64)
        #     for i in range(len(Lmat1[0])): # this can be vectorized
        #         disk_extents[i] = np.real(Lmat1[i, i]) + np.sum(np.abs(Lmat1[i, :])) - np.abs(Lmat1[i, i])  # note the np.real() here isn't really doing anything because the diagonals are already real, so this is just discarding the +0j
        #     bounds1[kzi] = np.max(disk_extents)
        #
        #     Lmat2 = kolmogorov_EVP.Sams_Lmat(N_Sam2, 0, lhat, kz*lhat, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
        #     disk_extents = np.zeros(len(Lmat2[0]), dtype=np.float64)
        #     for i in range(len(Lmat2[0])):
        #         disk_extents[i] = np.real(Lmat2[i, i]) + np.sum(np.abs(Lmat2[i, :])) - np.abs(Lmat2[i, i])
        #     bounds2[kzi] = np.max(disk_extents)
        #
        #     Lmat3 = kolmogorov_EVP.Sams_Lmat(N_Sam3, 0, lhat, kz * lhat, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
        #     disk_extents = np.zeros(len(Lmat3[0]), dtype=np.float64)
        #     for i in range(len(Lmat3[0])):
        #         disk_extents[i] = np.real(Lmat3[i, i]) + np.sum(np.abs(Lmat3[i, :])) - np.abs(Lmat3[i, i])
        #     bounds3[kzi] = np.max(disk_extents)
        plt.subplot(1, 2, 1)
        plt.plot(ks, gammas1)
        plt.plot(ks, gammas2, '--')
        plt.plot(ks, gammas3, ':')
        # plt.plot(ks, bounds1, '-', c='C0')
        # plt.plot(ks, bounds2, '--', c='C1')
        # plt.plot(ks, bounds3, ':', c='C2')
        plt.axhline(lamhat/0.33, c='red')
        plt.axhline(0, c='k')
        plt.xlabel(r'$k_z$')
        plt.ylabel(r'$\sigma_{KH}$')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        plt.subplot(1, 2, 2)
        plt.semilogx(ks, gammas1)
        plt.semilogx(ks, gammas2, '--')
        plt.semilogx(ks, gammas3, ':')
        # plt.plot(ks, bounds1, '-', c='C0')
        # plt.plot(ks, bounds2, '--', c='C1')
        # plt.plot(ks, bounds3, ':', c='C2')
        plt.axhline(lamhat/0.33, c='red')
        plt.axhline(0, c='k')
        plt.xlabel(r'$k_z$')
        plt.ylim(ymin=0)

        pdf.savefig()
        plt.close()
