"""
Script for plotting the structures (in x and z) of parasite modes. Adapted from similar scripts in my
"resistive_KH_modes" repo, such as paper_plot_KH_modes.py.

Note that the vectors in Sam's matrix are laid out like:
v = [..., psi_-1, T_-1, C_-1, A_-1, psi_0, T_0, C_0, A_0, ...]^T
"""
import numpy as np
from matplotlib import pyplot as plt
import parasite_model
import fingering_modes
import kolmogorov_EVP
from matplotlib.backends.backend_pdf import PdfPages

Pr = 1e-6  # 0.1
tau = 1e-7  # 0.1
Pm = 0.1  # 1
DB = Pr / Pm
HB = 1e-7  # 0.1
# kz_stars = np.linspace(0.01, 0.4, 40)  # normalized to lhat
# kz_stars = np.linspace(0.001, 0.01, 50)
kz_stars = np.append(np.geomspace(1e-6, 0.1, num=100, endpoint=False), np.linspace(0.1, 1.0, num=50))
mode_index = 0
# R0 = 9e6  # 5
# rs = np.linspace(1/49, 1, num=40, endpoint=False)
rs = np.linspace(0.79, 0.875, num=40, endpoint=True)
R0s = rs*((1/tau) - 1) + 1

N = 17
N_Sam = int((N-1)/2)  # Sam's definition of N is different than mine
C1 = 0.62  # for with_TC model
C2 = 0.33  # for with_TC model


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


ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
lamhats, l2hats = np.transpose([fingering_modes.gaml2max(Pr, tau, R0) for R0 in R0s])
lhats = np.sqrt(l2hats)

# evalues_all = []
results = parasite_model.results_vs_r0(R0s, HB, Pr, tau, DB, kz_stars, N, lamhats, l2hats, withTC=True, Sam=True, C1=C1, C2=C2)
print("done calculating results array")

with PdfPages('Parasite_structures_vs_R0_Pr{:0.2e}_tau{:0.2e}_HB{:0.2e}_Pm{:0.2e}_N{}-3.pdf'.format(Pr, tau, HB, Pm, N)) as pdf:
    for ri, r0 in enumerate(R0s):
        l2hat = l2hats[ri]
        lhat = lhats[ri]
        lamhat = lamhats[ri]
        kzs = kz_stars * lhat  # undo the lhat normalization

        char_poly1 = fingering_modes.characteristic_polynomial(Pr, tau, r0, l2hat)
        roots1 = char_poly1.roots()
        char_poly2 = fingering_modes.characteristic_polynomial(Pr, tau, r0, 4 * l2hat)
        roots2 = char_poly2.roots()
        # char_poly3 = fingering_modes.characteristic_polynomial(Pr, tau, R0, 9 * l_f ** 2)
        # roots3 = char_poly3.roots()

        # note that kmax below is in terms of kz_stars, not kzs
        # wf, kmax_star = parasite_model.w_f_withTC(Pr, tau, r0, HB, DB, kz_stars, N, get_kmax=True, C2=C2, lamhat=lamhat, l2hat=l2hat, Sam=True)
        wf = results["wf"][ri]
        kmax_star = results["kmax-star"][ri]
        kmax_ind = np.where(kz_stars == kmax_star)[0][0]  # the index in kz_stars where kmax_star occurs
        A_psi = wf / (2 * lhat)
        A_T = -lhat * A_psi / (lamhat + l2hat)
        A_C = -lhat * A_psi / (r0 * (lamhat + tau * l2hat))
        for kzi, kz in enumerate(kzs):
            L = kolmogorov_EVP.Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, r0, Pm, HB)
            w, v = np.linalg.eig(L)
            if kzi == 0:
                evalues = np.zeros((len(kzs), len(w)), dtype=np.complex128)  # all eigenvalues at this r0
            w_argsort = np.argsort(-np.real(w))  # 0th element is fastest-growing mode
            evalues[kzi] = w[w_argsort]
            if kzi == kmax_ind:
                # mode_kzi = v[:, w_argsort[mode_index]]  # grab nth (where n is mode_index) fastest-growing mode
                # if kzi == 0:
                #     modes = np.zeros((len(kzs), len(mode_kzi)), dtype=np.complex128)
                # modes[kzi] = mode_kzi
                mode = v[:, w_argsort[mode_index]]  # grab nth (where n is mode_index) fastest-growing mode
        # evalues_all.append(evalues)

        plt.figure(figsize=(10, 13))
        plt.subplot(4, 2, 1)
        plt.plot(R0s, results['wf'], 'o-')
        plt.ylabel(r'$w_f$')
        plt.xlabel(r'$R_0$')
        # plt.xlim(xmin=0)
        plt.xlim((0, 1e7))
        plt.ylim((0, 0.00045))
        plt.axvline(r0, c='red')

        plt.subplot(4, 2, 3)
        for i in range(3):  # these lines correspond to elevator mode solutions, for comparison
            plt.axhline(np.real(roots1[i]) / lamhat, c='C0')
            plt.axhline(np.real(roots2[i]) / lamhat, c='C1')

        d = np.shape(evalues)[1]  # rank of the matrix
        for j in range(d):  # loop over every mode branch
            i = np.where(np.abs(evalues[:, j].imag) > 1e-12)[0]
            if len(i) > 1:
                plt.scatter(kz_stars[i], evalues[i, j].real / lamhat, 1, color='r')  # /(N**2*l_fs**2+k_z**2))
            i = np.where(np.abs(evalues[:, j].imag) < 1e-12)[0]
            if len(i) > 1:
                plt.scatter(kz_stars[i], evalues[i, j].real / lamhat, 1, color='k')  # /(N**2*l_fs**2+k_z**2))
        plt.axvline(kmax_star, c='red')
        # plt.ylim((-10, 5))
        plt.ylim((roots2[2]/lamhat, 1/C2))
        plt.xlabel(r'$k_z/\hat{l}_f$')
        plt.ylabel(r'$Re[\lambda]/\hat{\lambda}_f$')
        plt.xscale("log")

        plt.subplot(4, 2, 4)
        for i in range(3):  # these lines correspond to elevator mode solutions, for comparison
            plt.axhline(np.real(roots1[i]) / lamhat, c='C0')
            plt.axhline(np.real(roots2[i]) / lamhat, c='C1')

        d = np.shape(evalues)[1]  # rank of the matrix
        for j in range(d):  # loop over every mode branch
            i = np.where(np.abs(evalues[:, j].imag) > 1e-12)[0]
            if len(i) > 1:
                plt.scatter(kz_stars[i], evalues[i, j].real / lamhat, 1, color='r')  # /(N**2*l_fs**2+k_z**2))
            i = np.where(np.abs(evalues[:, j].imag) < 1e-12)[0]
            if len(i) > 1:
                plt.scatter(kz_stars[i], evalues[i, j].real / lamhat, 1, color='k')  # /(N**2*l_fs**2+k_z**2))
        plt.axvline(kmax_star, c='red')
        # plt.ylim((-10, 5))
        plt.ylim((-5, 1/C2))
        plt.xlabel(r'$k_z/\hat{l}_f$')
        # plt.ylabel(r'$Re[\lambda]/\hat{\lambda}_f$')
        plt.xscale("log")

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
        psi_xz, xs, zs = xz_from_kxkz(psi_ishift, np.fft.ifftshift(ns), kmax_star, scalefac=1)
        T_xz = xz_from_kxkz(T_ishift, np.fft.ifftshift(ns), kmax_star, scalefac=1)[0]
        C_xz = xz_from_kxkz(C_ishift, np.fft.ifftshift(ns), kmax_star, scalefac=1)[0]
        A_xz = xz_from_kxkz(A_ishift, np.fft.ifftshift(ns), kmax_star, scalefac=1)[0]

        plt.subplot(4, 2, 5)
        plt.contourf(xs, zs, psi_xz.T)
        plt.colorbar()
        plt.ylabel(r'$z$')
        # plt.xlabel(r'$x$')
        plt.title(r'$\psi$')

        plt.subplot(4, 2, 6)
        plt.contourf(xs, zs, T_xz.T)
        plt.colorbar()
        # plt.ylabel(r'$z$')
        # plt.xlabel(r'$x$')
        plt.title(r'$T$')

        plt.subplot(4, 2, 7)
        plt.contourf(xs, zs, C_xz.T)
        plt.colorbar()
        plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        plt.title(r'$C$')

        plt.subplot(4, 2, 8)
        plt.contourf(xs, zs, A_xz.T)
        plt.colorbar()
        # plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        plt.title(r'$A$')

        plt.tight_layout()
        pdf.savefig()
        plt.close()
