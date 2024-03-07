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
import parasite_model
from matplotlib.backends.backend_pdf import PdfPages

Pr = 1e-6
tau = 1e-7
Pm = 1e-1
DB = Pr / Pm
HB = 1e-7
wf = -1  # 4.2e-4  # 0.000425
kz_star = 1e-4  # normalized to lhat
R0 = 1.2e6
N1 = 3
N2 = 5
N3 = 17

lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
lhat = np.sqrt(l2hat)
if wf < 0:
    C1 = 0.62  # for with_TC model
    C2 = 0.33  # for with_TC model
    kz_stars = np.append(np.geomspace(1e-6, 0.1, num=100, endpoint=False), np.linspace(0.1, 1.0, num=50))
    result = parasite_model.parasite_results(R0, HB, Pr, tau, DB, kz_stars, N3, lamhat, l2hat, withTC=True, Sam=True, C1=C1, C2=C2)
    wf = result['wf']
# v_Alfven = np.sqrt(HB/(wf**2))  # units aren't right


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
    if np.all(np.isreal(phi_xz)):
        phi_xz = np.real(phi_xz)
    return phi_xz, xs, zs


ns = np.array(range(-int((N3 - 1) / 2), int((N3 + 1) / 2), 1))  # these are wavenumbers in shear direction
kz = kz_star * lhat  # undo the lhat normalization
A_psi = wf / (2 * lhat)
A_T = -lhat * A_psi / (lamhat + l2hat)
A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
evalues = []

for n_ind, N in enumerate([N1, N2, N3]):
    N_Sam = int((N-1)/2)  # Sam's definition of N is different than mine
    L = kolmogorov_EVP.Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
    w, v = np.linalg.eig(L)
    evalues.append(w)
    if n_ind == 2:  # only bother with the N=N3 stuff for the following
        w_argsort = np.argsort(-np.real(w))
        w_sorted = w[w_argsort]  # first element is fastest growing mode, descends from there

char_poly1 = fingering_modes.characteristic_polynomial(Pr, tau, R0, l2hat)
roots1 = char_poly1.roots()
char_poly2 = fingering_modes.characteristic_polynomial(Pr, tau, R0, 4 * l2hat)
roots2 = char_poly2.roots()

with PdfPages('figures/parasite_structures/vs_ind/Parasite_mode_structures_Pr{:0.2e}_tau{:0.2e}_HB{:0.2e}_Pm{:0.2e}_R0{:0.2e}_wf{:0.2e}_kz{:0.2e}.pdf'.format(Pr, tau, HB, Pm, R0, wf, kz_star)) as pdf:
    for i, evalue in enumerate(w_sorted[:-10]):
        plt.subplot(3, 1, 1)
        plt.plot(np.real(evalues[0]/lamhat), np.imag(evalues[0]/lamhat), 'x', label=r'$N = {}$'.format(N1))
        plt.plot(np.real(evalues[1]/lamhat), np.imag(evalues[1]/lamhat), '+', label=r'$N = {}$'.format(N2))
        plt.plot(np.real(evalues[2]/lamhat), np.imag(evalues[2]/lamhat), '.', label=r'$N = {}$'.format(N3))
        plt.plot(np.real(evalue/lamhat), np.imag(evalue/lamhat), '.', c='red')
        plt.title(r'$\lambda/\lambda_f = {}$'.format(evalue/lamhat))
        # plt.xlim(xmin=np.real(w_sorted[-11]/lamhat))
        for ri in range(3):  # these lines correspond to elevator mode solutions, for comparison
            plt.axvline(np.real(roots1[ri]) / lamhat, c='C0')
            plt.axvline(np.real(roots2[ri]) / lamhat, c='C1')
        plt.xlim((-5, 1.25))
        plt.xlabel(r'$\Re[\lambda]$')
        plt.ylabel(r'$\Im[\lambda]$')
        plt.legend()

        mode = v[:, w_argsort[i]]  # loop through the eigenmodes
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