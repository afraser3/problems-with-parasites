"""
Script for plotting the structures (in x and z) of parasite modes. Adapted from similar scripts in my
"resistive_KH_modes" repo, such as paper_plot_KH_modes.py.

Note that the vectors in Sam's matrix are laid out like:
v = [..., psi_-1, T_-1, C_-1, A_-1, psi_0, T_0, C_0, A_0, ...]^T
"""
import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import kolmogorov_EVP
from matplotlib.backends.backend_pdf import PdfPages

Pr = 1e-6  # 0.1
tau = 1e-7  # 0.1
Pm = 0.1  # 1
HB = 1e-7  # 0.1
# wf = 7e-5  # 2
wf = 0.0003806622073821217
# kz_stars = np.linspace(0.01, 0.4, 40)  # normalized to lhat
# kz_stars = np.linspace(0.001, 0.01, 50)
kz_stars = np.geomspace(1e-6, 0.01, num=50, endpoint=True)
mode_index = 0
R0 = 8e6  # 5
N1 = 3
N2 = 5
N3 = 17



def xz_from_kxkz(phi_kx_ishift, ns_ishift, kz, scalefac=1):
    # given phi(kx,kz) returns phi(x,z) where z is the direction of flow, x is the direction of shear
    # ns is the array of kx values over which the Fourier series of phi is given
    # returns phi(x,z) where the x,z grid is len(ns)*scalefac points in each of x and z, with
    # 0 <= x < 2pi and 0 <= z < 2pi/kz
    #
    # NOT SET UP FOR delta>0 MODES
    # Also, I'm pretty sure I never got scalefac working correctly
    #
    # ASSUMES phi_kxkz IS IN STANDARD FFT ORDER, i.e., STARTING WITH kx=0 PART
    if ns_ishift[0] != 0:
        raise ValueError('Need to provide arrays in standard FFT order')
    if int(scalefac*len(ns_ishift)) != scalefac*len(ns_ishift):
        raise ValueError('Need scalefac * len(ns_ishift) to be an integer')
    phi_kxkz_ishift = np.zeros((int(scalefac*len(ns_ishift)), int(scalefac*len(ns_ishift))), dtype=np.complex128)
    phi_kxkz_ishift[:, 1] = phi_kx_ishift
    phi_kx = np.fft.fftshift(phi_kx_ishift)
    # need to do some shifting around here in order to ensure
    # that phi_kxkz(-kx, -kz) = conj[phi_kxkz(kx, kz)]
    phi_kx_flip = phi_kx[::-1]
    phi_kx_ishift_flip = np.fft.ifftshift(phi_kx_flip)
    phi_kxkz_ishift[:, -1] = np.conj(phi_kx_ishift_flip)
    xs = np.linspace(0, 2.0*np.pi, num=int(scalefac*len(ns_ishift)), endpoint=False)
    zs = np.linspace(0, 2.0*np.pi/kz, num=int(scalefac*len(ns_ishift)), endpoint=False)
    phi_xz = np.fft.ifft2(phi_kxkz_ishift)*len(xs)*len(zs)
    return phi_xz, xs, zs


ns = np.array(range(-int((N3 - 1) / 2), int((N3 + 1) / 2), 1))  # these are wavenumbers in shear direction
lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
lhat = np.sqrt(l2hat)
kzs = kz_stars * lhat  # undo the lhat normalization
A_psi = wf / (2 * lhat)
A_T = -lhat * A_psi / (lamhat + l2hat)
A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
char_poly1 = fingering_modes.characteristic_polynomial(Pr, tau, R0, l2hat)
roots1 = char_poly1.roots()
char_poly2 = fingering_modes.characteristic_polynomial(Pr, tau, R0, 4 * l2hat)
roots2 = char_poly2.roots()
C1 = 0.62
C2 = 0.33
# lambdas = np.zeros((3, len(kzs)), dtype=np.complex128)
evalues_all = []  # including all N

for n_ind, N in enumerate([N1, N2, N3]):
    N_Sam = int((N-1)/2)  # Sam's definition of N is different than mine
    for kzi, kz in enumerate(kzs):
        L = kolmogorov_EVP.Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
        w, v = np.linalg.eig(L)
        if kzi == 0:
            evalues = np.zeros((len(kzs), len(w)), dtype=np.complex128)  # all eigenvalues at this N
        w_argsort = np.argsort(-np.real(w))  # 0th element is fastest-growing mode
        evalues[kzi] = w[w_argsort]
        if n_ind == 2:
            mode_kzi = v[:, w_argsort[mode_index]]  # grab nth (where n is mode_index) fastest-growing mode
            if kzi == 0:
                modes = np.zeros((len(kzs), len(mode_kzi)), dtype=np.complex128)
            modes[kzi] = mode_kzi
    evalues_all.append(evalues)

with PdfPages('figures/parasite_structures/vs_kz/Parasite_structures_vs_kz_Pr{:0.2e}_tau{:0.2e}_HB{:0.2e}_Pm{:0.2e}_R0{:0.2e}_wf{:0.2e}_ind{}.pdf'.format(Pr, tau, HB, Pm, R0, wf, mode_index)) as pdf:
    # for i, evalue in enumerate(w_sorted[:-10]):
    for kzi, kz in enumerate(kzs):
        kz_star = kz_stars[kzi]
        plt.figure(figsize=(5, 10))
        plt.subplot(3, 1, 1)
        # for i in range(len(evalues_all[0][0])):  # for each mode
        #     plt.plot(kz_stars, np.real(evalues_all[0][:, i]), 'x', c='C0', label=r'$N = {}$'.format(N1))  # plot that mode's growth rate vs kz
        # for i in range(len(evalues_all[1][0])):  # for each mode
        #     plt.plot(kz_stars, np.real(evalues_all[1][:, i]), '+', c='C1', label=r'$N = {}$'.format(N2))
        # for i in range(len(evalues_all[2][0])):  # for each mode
        #     plt.plot(kz_stars, np.real(evalues_all[2][:, i]), '.', c='C2', label=r'$N = {}$'.format(N3))
        for i in range(8):
            # evalues_N1_sorted = evalues_all[0]
            if i == 0:
                plt.plot(kz_stars, np.real(evalues_all[0][:, i]) / lamhat, '-', c='C0', label=r'$N = {}$'.format(N1))
                plt.plot(kz_stars, np.real(evalues_all[1][:, i]) / lamhat, '--', c='C1', label=r'$N = {}$'.format(N2))
                plt.plot(kz_stars, np.real(evalues_all[2][:, i]) / lamhat, ':', c='C2', label=r'$N = {}$'.format(N3))
            else:
                plt.plot(kz_stars, np.real(evalues_all[0][:, i]) / lamhat, '-', c='C0')
                plt.plot(kz_stars, np.real(evalues_all[1][:, i]) / lamhat, '--', c='C1')
                plt.plot(kz_stars, np.real(evalues_all[2][:, i]) / lamhat, ':', c='C2')
        plt.axvline(kz_star, c='red')
        plt.xscale('log')
        plt.ylim((2 * roots2[2] / lamhat, 1 / C2))
        # plt.xlim(xmin=0)
        # plt.ylim(ymin=-0.5)
        plt.xlabel(r'$k_z/\hat{l}_f$')
        plt.ylabel(r'$Re[\lambda]$')
        plt.legend()

        # mode = v[:, w_argsort[i]]  # loop through the eigenmodes
        mode = modes[kzi]
        norm = mode[4]
        mode = mode/norm
        # extract the individual fields from the eigenvector
        mode_psi = mode[::4]
        mode_T = mode[1::4]
        mode_C = mode[2::4]
        mode_A = mode[3::4]

        # Now let's iFFT these bad boys. First, put them into standard FFT format
        psi_ishift = np.fft.ifftshift(mode_psi)
        T_ishift = np.fft.ifftshift(mode_T)
        C_ishift = np.fft.ifftshift(mode_C)
        A_ishift = np.fft.ifftshift(mode_A)

        # then pass them to the iFFT helper function above
        psi_xz, xs, zs = xz_from_kxkz(psi_ishift, np.fft.ifftshift(ns), kz_star, scalefac=1)
        T_xz = xz_from_kxkz(T_ishift, np.fft.ifftshift(ns), kz_star, scalefac=1)[0]
        C_xz = xz_from_kxkz(C_ishift, np.fft.ifftshift(ns), kz_star, scalefac=1)[0]
        A_xz = xz_from_kxkz(A_ishift, np.fft.ifftshift(ns), kz_star, scalefac=1)[0]

        plt.subplot(3, 2, 3)
        plt.contourf(xs, zs, psi_xz.T)
        plt.colorbar()
        plt.ylabel(r'$z$')
        # plt.xlabel(r'$x$')
        plt.title(r'$\psi$')

        plt.subplot(3, 2, 4)
        plt.contourf(xs, zs, T_xz.T)
        plt.colorbar()
        # plt.ylabel(r'$z$')
        # plt.xlabel(r'$x$')
        plt.title(r'$T$')

        plt.subplot(3, 2, 5)
        plt.contourf(xs, zs, C_xz.T)
        plt.colorbar()
        plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        plt.title(r'$C$')

        plt.subplot(3, 2, 6)
        plt.contourf(xs, zs, A_xz.T)
        plt.colorbar()
        # plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        plt.title(r'$A$')

        plt.tight_layout()
        pdf.savefig()
        plt.close()