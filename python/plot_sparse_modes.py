import numpy as np
from matplotlib import pyplot as plt
import fingering_modes
import kolmogorov_EVP
import parasite_model
import time
from matplotlib.backends.backend_pdf import PdfPages

Pr = 1e-6
tau = 1e-7
Pm = 1e-1
DB = Pr / Pm
HB = 1e-7
wf = -1
wf = 0.0003806622073821217  # this is for R0=8e6
R0 = 8e6
N = 17
N_Sam = int((N-1)/2)  # Sam's definition of N is different than mine
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction

kz_stars = np.append(np.geomspace(1e-6, 0.1, num=2*100, endpoint=False), np.linspace(0.1, 1.0, num=50))
lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
lhat = np.sqrt(l2hat)
kzs = kz_stars * lhat  # undo the lhat normalization
C1 = 0.62  # for with_TC model
C2 = 0.33  # for with_TC model
if wf < 0:
    result = parasite_model.parasite_results(R0, HB, Pr, tau, DB, kz_stars, N, lamhat, l2hat, withTC=True, Sam=True, C1=C1, C2=C2)
    wf = result['wf']


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


A_psi = wf / (2 * lhat)
A_T = -lhat * A_psi / (lamhat + l2hat)
A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
char_poly1 = fingering_modes.characteristic_polynomial(Pr, tau, R0, l2hat)
roots1 = char_poly1.roots()
char_poly2 = fingering_modes.characteristic_polynomial(Pr, tau, R0, 4 * l2hat)
roots2 = char_poly2.roots()
# test if my timing is being screwed up by the import statements in kolmogorov_EVP.py
# Ltest = kolmogorov_EVP.Sams_Lmat(N_Sam, 0, lhat, kzs[0], A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
# end test
dense_start_time = time.time()
for kzi, kz in enumerate(kzs):
    L = kolmogorov_EVP.Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, R0, Pm, HB)
    w, v = np.linalg.eig(L)
    if kzi == 0:
        evalues = np.zeros((len(kzs), len(w)), dtype=np.complex128)  # all eigenvalues at this r0
    w_argsort = np.argsort(-np.real(w))  # 0th element is fastest-growing mode
    evalues[kzi] = w[w_argsort]
dense_stop_time = time.time()

sparse_start_time = time.time()
sparse_out = kolmogorov_EVP.gamma_over_k_withTC(0.0, wf, HB, DB, Pr, tau, R0, kz_stars, N, Sam=True, sparse_method=True, pass_sigma=True, sparse_matrix='csc', Richs_matrix=False, get_evecs=True)
sparse_stop_time = time.time()
# unpack sparse_out
elevator1_evalue1s, elevator1_mode1s, elevator1_evalue2s, elevator1_mode2s, elevator2_evalue1s, elevator2_mode1s, elevator2_evalue2s, elevator2_mode2s = sparse_out
# elevator1_evalue1s, elevator1_evalue2s, elevator2_evalue1s, elevator2_evalue2s = sparse_out
###
sparse_evalues = [elevator1_evalue1s, elevator1_evalue2s, elevator2_evalue1s, elevator2_evalue2s]
evecs = [elevator1_mode1s, elevator1_mode2s, elevator2_mode1s, elevator2_mode2s]
branch = 2
with PdfPages('figures/parasite_structures/test_sparse/plot_sparse_Pr{:0.2e}_tau{:0.2e}_HB{:0.2e}_Pm{:0.2e}_R0{:0.2e}-N{}-branch{}.pdf'.format(Pr, tau, HB, Pm, R0, N, branch)) as pdf:
    for kzi, kz in enumerate(kz_stars):
        if kzi % 4 == 0:
            plt.figure(figsize=(10, 16))
            plt.subplot(3, 1, 1)
            plt.axhline(np.max(np.real(roots1)) / lamhat, c='C0')
            plt.axhline(np.max(np.real(roots2)) / lamhat, c='C1')
            plt.axvline(kz, c='red')

            d = np.shape(evalues)[1]  # rank of the matrix (same as len(evalues[kzi]))
            for j in range(d):  # loop over every mode branch
                i = np.where(np.abs(evalues[:, j].imag) > 1e-12)[0]
                if len(i) > 1:
                    plt.plot(kz_stars[i], evalues[i, j].real / lamhat, '.', ms=1.5, color='r')  # TODO: no need for the i indexing here! Oops
                i = np.where(np.abs(evalues[:, j].imag) < 1e-12)[0]
                if len(i) > 1:
                    plt.plot(kz_stars[i], evalues[i, j].real / lamhat, '.', ms=1.5, color='k')
            plt.plot(kz_stars, np.real(sparse_evalues[branch]) / lamhat, '.', ms=1)
            plt.ylim((2 * roots2[2] / lamhat, 1 / C2))
            plt.xlabel(r'$k_z/\hat{l}_f$')
            plt.ylabel(r'$Re[\lambda]/\hat{\lambda}_f$')
            plt.xscale("log")

            mode = evecs[branch][kzi]
            norm = mode[4]
            mode = mode / norm
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
            psi_xz, xs, zs = xz_from_kxkz(psi_ishift, np.fft.ifftshift(ns), kz, scalefac=1)
            T_xz = xz_from_kxkz(T_ishift, np.fft.ifftshift(ns), kz, scalefac=1)[0]
            C_xz = xz_from_kxkz(C_ishift, np.fft.ifftshift(ns), kz, scalefac=1)[0]
            A_xz = xz_from_kxkz(A_ishift, np.fft.ifftshift(ns), kz, scalefac=1)[0]

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

