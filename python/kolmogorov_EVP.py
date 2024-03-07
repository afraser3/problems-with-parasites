"""Solves the eigenvalue problem for KH instability of sinusoidal flow

Sets up the eigenvalue problem corresponding to the KH instability of a
sinusoidal flow profile in MHD with a uniform, flow-aligned magnetic 
field. Includes various functions for finding just the most-unstable
growth rate, or the complex frequency, or a scan over k, or a helper
function for using root-finding algorithms for finding the flow speed
such that the growth rate matches some value.

Can be used with viscosity/resistivity (ideal=False) or without 
(ideal=True).

Note that most methods return frequencies or growth rates that are
normalized to the finger width/speed, like in Fig 3 of
Harrington & Garaud. TODO: clarify which methods

Exceptions
----------
KrangeError(Exception)
    Used for handling the situation where you want the most-
    unstable growth rate for a given range of ks, but the most-
    unstable growth rate occurs at one of the edges of that rage,
    and thus you likely have not found a true local maximum.

Methods
-------
Deln(k,n,delta)
    Calculates Delta_n from my LaTeX notes (for setting up KH EVP)
Lmat(delta, M2, Re, Rm, k, N, ideal=False)
    Constructs the linear operator whose eigenvalues are the
    complex frequencies for the KH eigenmodes, i.e., the modes
    are taken to go like f(t) ~ exp[i omega t], and the eigenvalues
    of this matrix are the various omega for different modes.
gamfromL(L)
    Calls numpy.linalg.eig to get the eigenvalues. Returns the
    growth rate gamma = -np.imag(omega) of the most unstable mode
omegafromL(L)
    Returns the complex frequency instead (should just merge these
    two functions)
gamma_over_k(delta, M2, Re, Rm, ks, N, ideal=False)
    Calls gamfromL for each k in array ks, returns the resulting
    array of growth rates gamma[ks]
omega_over_k(...)
    Same as above but complex frequencies
gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, badks_except=False)
    Same as gamma_over_k but returns just the most unstable growth
    rate over the provided range of ks. If the result is positive
    and badks_except=True, it will check to see if the maximum
    occurred at either the highest or lowest k in ks, and throw an
    error if it did. This is so that you can make sure your local
    maximum isn't just a maximum because of the range you chose.
KHparams_from_fingering(w, lhat, HB, Pr, DB)
    Returns H_B^* (equivalent to 1/M_A^2), Re, and Rm defined in
    terms of the finger's speed, width, and fluid parameters.
gammax_minus_lambda(w, lamhat, lhat, HB, Pr, DB, delta, ks, N, 
            ideal=False, badks_exception=False, CH=1.66)
    Just a helper function around gammax_kscan. Instead of
    returning gamma, it returns gamma*w_f*l_f - C_H*lambda_f, i.e.,
    equations 30 and 31 in Harrington & Garaud except without the
    fit, and written in the form F(stuff) = 0 so that saturation is
    given by the roots of F.

"""
# TODO: rename KH growth rate from gamma to sigma for consistency w/ HG18
import numpy as np
import fingering_modes
import scipy.sparse
# import scipy


class KrangeError(Exception):
    pass


def KHparams_from_fingering(w, lhat, hb, pr, db):
    """
    Calculates the dimensionless parameters relevant to the KH stability analysis
    in terms of the relevant quantities from the DDC problem
    """
    hb_star = hb / w ** 2.0
    re = w / (pr * lhat)
    rm = w / (db * lhat)
    return [hb_star, re, rm]


def Deln(k, n, delta, finger_norm=False, k0=1.0):  # \Delta_n in my notes. So simple, probably shouldn't be a function. Note k0 is just lhat
    if finger_norm:
        return k ** 2.0 + k0**2.0 * (n + delta)**2.0
    else:
        return k ** 2.0 + (n + delta) ** 2.0


def Lmat(delta, M2, Re, Rm, k, N, ideal=False):
    """Returns the linear operator for the KH instability

    Note that the eigenvalues are the complex frequencies, and the
    eigenvectors are the streamfunction and flux function mixed together,
    with the nth entry being the streamfunction at some wavenumber, and
    the n+1th being the flux function at a wavenumber.

    The eigenvalue problem this corresponds to is normalized to the flow speed and length scale, and background field.

    Parameters
    ----------
    delta : float
        This should be in the range 0 <= delta <= 0.5 and indicates
        the periodicity of the KH mode relative to the wavelength
        of the sinusoidal shear flow. See LaTeX notes; should
        probably be left at delta=0.0
    M2 : float
        The parameter H_B^* in Harrington & Garaud
    Re : float
        Reynolds number
    Rm : float
        Magnetic Reynolds number
    k : float
        Wavenumber in direction of flow
    N : int (ODD NUMBER)
        Numerical resolution in direction of shear
    ideal : Bool, default=False
        Whether or not to set viscosity, resistivity -> 0
        (if True then Re and Rm don't matter)

    Returns
    -------
    L : 2N x 2N numpy array
        Matrix whose eigenvalues are complex frequencies of KH modes
    """
    diss = 1.0 - ideal  # =0 for ideal=True, =1 for ideal=False
    # ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # the n over which we sum the Fourier series
    ms = list(range(-N + 1, N + 1, 1))  # this is going to be twice as long so we can loop over each n twice, once for phi and once for psi

    # the following few lines just sets up arrays of Delta_n
    # delns = [Deln(k, n, delta) for n in ns]
    delns_m = np.zeros_like(ms, dtype=np.float64)
    for i, m in enumerate(ms):
        if m % 2 == 0:
            delns_m[i] = Deln(k, m / 2, delta)
        else:
            delns_m[i] = Deln(k, (m - 1) / 2, delta)

    M = 2 * N
    L = np.zeros((M, M), dtype=np.complex128)

    # first fill in the entries that aren't near the edges
    for i, m in enumerate(ms):
        deltan = delns_m[i]
        if i > 1 and i < len(ms) - 2:  # avoid entries near edges
            deltanp1 = delns_m[i + 2]
            deltanm1 = delns_m[i - 2]
            if m % 2 == 0:  # phi entries
                # n = m / 2
                # phi_n, phi_n part
                L[i, i] = (1.0j) * (diss / Re) * deltan
                # phi_n, psi_n part
                L[i, i + 1] = M2 * k
                # phi_n, phi_n+1
                L[i, i + 2] = -k * (1 - deltanp1) / (2.0j * deltan)
                # phi_n, phi_n-1
                L[i, i - 2] = k * (1 - deltanm1) / (2.0j * deltan)
            else:  # psi entries
                # psi_n, psi_n
                L[i, i] = (1.0j) * deltan * diss / Rm
                # psi_n, phi_n
                L[i, i - 1] = k
                # psi_n, psi_n+1
                L[i, i + 2] = k / (2.0j)
                # psi_n, psi_n-1
                L[i, i - 2] = -k / (2.0j)
    # now do the edges
    # first, the most negative phi
    L[0, 0] = (1.0j) * delns_m[0] * diss / Re
    L[0, 1] = M2 * k
    L[0, 2] = -k * (1 - delns_m[2]) / (2.0j * delns_m[0])
    # most negative psi
    L[1, 1] = (1.0j) * delns_m[1] * diss / Rm
    L[1, 0] = k
    L[1, 3] = k / (2.0j)
    # most positive phi
    L[-2, -2] = (1.0j) * delns_m[-2] * diss / Re
    L[-2, -1] = M2 * k
    L[-2, -4] = k * (1 - delns_m[-4]) / (2.0j * delns_m[-2])
    # most positive psi
    L[-1, -1] = (1.0j) * delns_m[-1] * diss / Rm
    L[-1, -2] = k
    L[-1, -3] = -k / (2.0j)
    return L


def Lmat_withTC(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, k0, kz, delta, N, ideal=False, zero_T_C_diss=False):
    # THIS DOES NOT WORK AND STILL NEEDS DEBUGGING
    # note k0 is just lhat
    diss = 1.0 - ideal  # =0 for ideal=True, =1 for ideal=False
    diss_TC = 1.0 - zero_T_C_diss
    M = int(4*N)  # size of matrix
    L = np.zeros((M, M), dtype=np.complex128)
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # e.g. [-4, -3, -2, ..., 4] for N=9
    ns_long, field_iter = np.meshgrid(ns, range(4))
    ns_long = ns_long.T.flatten()  # [-4, -4, -4, -4, -3, -3, -3, -3, ...] so four copies of ns
    field_iter = field_iter.T.flatten()  # [0,1,2,3,0,1,2,3,...] useful for looping over the 4 physical fields
    delns_m = Deln(kz, ns_long, delta, True, k0)  # \Delta_n from my notes, but four copies just like ns_long

    for i, n in enumerate(ns_long):
        delta_n = delns_m[i]  # \Delta_n
        if n != ns[0] and n != ns[-1]:  # avoid filling in the edges of the matrix for now
            delta_nm1 = delns_m[i-4]  # \Delta_{n-1}
            delta_np1 = delns_m[i+4]  # \Delta_{n+1}
            if field_iter[i] == 0:  # phi entries
                L[i, i] = 1.0j * Pr * delta_n * diss  # phi_n, phi_n part
                L[i, i + 1] = HB * kz  # phi_n, psi_n
                L[i, i + 2] = Pr * (n + delta) * k0 / delta_n  # phi_n, T_n
                L[i, i + 3] = -Pr * (n + delta) * k0 / delta_n  # phi_n, C_n
                #L[i, i + 4] = -A_phi * kz * k0 * (k0**2.0 - delta_np1) / (2.0j * delta_n)  # phi_n, phi_{n+1}
                #L[i, i - 4] = A_phi * kz * k0 * (k0**2.0 - delta_nm1) / (2.0j * delta_n)  # phi_n, phi_{n-1}
                L[i, i + 4] = A_phi * kz * k0 * (k0 ** 2.0 - delta_np1) / (2.0j * delta_n)  # phi_n, phi_{n+1}
                L[i, i - 4] = -A_phi * kz * k0 * (k0 ** 2.0 - delta_nm1) / (2.0j * delta_n)  # phi_n, phi_{n-1}
            if field_iter[i] == 1:  # psi entries
                L[i, i] = 1.0j * DB * delta_n * diss  # psi_n, psi_n part
                L[i, i - 1] = kz  # psi_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # psi_n, psi_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # psi_n, psi_{n-1}
            if field_iter[i] == 2:  # T entries
                L[i, i] = 1.0j * delta_n * diss_TC  # T_n, T_n part
                L[i, i - 2] = (n + delta) * k0  # T_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # T_n, T_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # T_n, T_{n-1}
                L[i, i + 2] = -A_T * kz * k0 / 2.0  # T_n, phi_{n+1}
                L[i, i - 6] = L[i, i + 2]  # T_n, phi_{n-1}
            if field_iter[i] == 3:  # C entries
                L[i, i] = 1.0j * tau * delta_n * diss_TC  # C_n, C_n part
                L[i, i - 3] = (n + delta) * k0 / R0  # C_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # C_n, C_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # C_n, C_{n-1}
                L[i, i + 1] = -A_C * kz * k0 / 2.0  # C_n, phi_{n+1}
                L[i, i - 7] = L[i, i + 1]  # C_n, phi_{n-1}
    # Now fill in the edges
    # First, most negative phi part
    L[0, 0] = 1.0j * Pr * delns_m[0] * diss  # phi_-N, phi_-N
    L[0, 1] = HB * kz  # phi_-N, psi_-N
    L[0, 2] = Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, T_-N
    L[0, 3] = -Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, C_-N
    # L[0, 4] = -A_phi * kz * k0 * (k0**2.0 - delns_m[4]) / (2.0j * delns_m[0])  # phi_-N, phi_{-N + 1}
    L[0, 4] = A_phi * kz * k0 * (k0 ** 2.0 - delns_m[4]) / (2.0j * delns_m[0])  # phi_-N, phi_{-N + 1}
    # Most positive phi part
    L[-4, -4] = 1.0j * Pr * delns_m[-4] * diss  # phi_N, phi_N
    L[-4, -3] = HB * kz  # phi_N, psi_N
    L[-4, -2] = Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, T_N
    L[-4, -1] = -Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, C_N
    # L[-4, -8] = A_phi * kz * k0 * (k0**2.0 - delns_m[-8]) / (2.0j * delns_m[-4])  # phi_N, phi_{N-1}
    L[-4, -8] = -A_phi * kz * k0 * (k0 ** 2.0 - delns_m[-8]) / (2.0j * delns_m[-4])  # phi_N, phi_{N-1}
    # Most negative psi part
    L[1, 0] = kz  # psi_-N, phi_-N
    L[1, 1] = 1.0j * DB * delns_m[1] * diss  # psi_-N, psi_-N
    L[1, 5] = A_phi * kz * k0 / 2.0j  # psi_-N, psi_{-N + 1}
    # Most positive psi part
    L[-3, -4] = kz  # psi_N, phi_N
    L[-3, -3] = 1.0j * DB * delns_m[-3] * diss  # psi_N, psi_N
    L[-3, -7] = -A_phi * kz * k0 / 2.0j  # psi_N, psi_{N-1}
    # Most negative T part
    L[2, 0] = (ns[0] + delta) * k0  # T_-N, phi_-N
    L[2, 2] = 1.0j * delns_m[2] * diss_TC  # T_-N, T_-N
    L[2, 4] = -A_T * kz * k0 / 2.0  # T_-N, phi_{-N + 1}
    L[2, 6] = A_phi * kz * k0 / 2.0j  # T_-N, T_{-N + 1}
    # Most positive T part
    L[-2, -4] = (ns[-1] + delta) * k0  # T_N, phi_N
    L[-2, -2] = 1.0j * delns_m[-2] * diss_TC  # T_N, T_N
    L[-2, -8] = -A_T * kz * k0 / 2.0  # T_N, phi_{N - 1}
    L[-2, -6] = -A_phi * kz * k0 / 2.0j  # T_N, T_{N - 1}
    # Most negative C part
    L[3, 0] = (ns[0] + delta) * k0 / R0  # C_-N, phi_-N
    L[3, 3] = 1.0j * delns_m[3] * tau * diss_TC  # C_-N, C_-N
    L[3, 4] = -A_C * kz * k0 / 2.0  # C_-N, phi_{-N + 1}
    L[3, 7] = A_phi * kz * k0 / 2.0j  # C_-N, C_{-N + 1}
    # Most positive C part
    L[-1, -4] = (ns[-1] + delta) * k0 / R0  # C_N, phi_N
    L[-1, -1] = 1.0j * delns_m[-1] * tau * diss_TC  # C_N, C_N
    L[-1, -8] = -A_T * kz * k0 / 2.0  # C_N, phi_{N - 1}
    L[-1, -5] = -A_phi * kz * k0 / 2.0j  # C_N, C_{N - 1}
    return L


def Lmat2_withTC(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, k0, kz, delta, N, ideal=False, zero_T_C_diss=False):
    # THIS DOES NOT WORK AND STILL NEEDS DEBUGGING
    # note k0 is just lhat
    diss = 1.0 - ideal  # =0 for ideal=True, =1 for ideal=False
    diss_TC = 1.0 - zero_T_C_diss
    M = int(4*N)  # size of matrix
    L = np.zeros((M, M), dtype=np.complex128)
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # e.g. [-4, -3, -2, ..., 4] for N=9
    ns_long, field_iter = np.meshgrid(ns, range(4))
    ns_long = ns_long.T.flatten()  # [-4, -4, -4, -4, -3, -3, -3, -3, ...] so four copies of ns
    field_iter = field_iter.T.flatten()  # [0,1,2,3,0,1,2,3,...] useful for looping over the 4 physical fields
    delns_m = Deln(kz, ns_long, delta, True, k0)  # \Delta_n from my notes, but four copies just like ns_long

    for i, n in enumerate(ns_long):
        delta_n = delns_m[i]  # \Delta_n
        if n != ns[0] and n != ns[-1]:  # avoid filling in the edges of the matrix for now
            delta_nm1 = delns_m[i-4]  # \Delta_{n-1}
            delta_np1 = delns_m[i+4]  # \Delta_{n+1}
            if field_iter[i] == 0:  # phi entries
                L[i, i] = 1.0j * Pr * delta_n * diss  # phi_n, phi_n part
                L[i, i + 1] = HB * kz  # phi_n, psi_n
                L[i, i + 2] = Pr * (n + delta) * k0 / delta_n  # phi_n, T_n
                L[i, i + 3] = -Pr * (n + delta) * k0 / delta_n  # phi_n, C_n
                L[i, i + 4] = -A_phi * kz * k0 * (k0**2.0 - delta_np1) / (2.0j * delta_n)  # phi_n, phi_{n+1}
                L[i, i - 4] = A_phi * kz * k0 * (k0**2.0 - delta_nm1) / (2.0j * delta_n)  # phi_n, phi_{n-1}
                # L[i, i + 4] = A_phi * kz * k0 * (k0 ** 2.0 - delta_np1) / (2.0j * delta_n)  # phi_n, phi_{n+1}
                # L[i, i - 4] = -A_phi * kz * k0 * (k0 ** 2.0 - delta_nm1) / (2.0j * delta_n)  # phi_n, phi_{n-1}
            if field_iter[i] == 1:  # psi entries
                L[i, i] = 1.0j * DB * delta_n * diss  # psi_n, psi_n part
                L[i, i - 1] = kz  # psi_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # psi_n, psi_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # psi_n, psi_{n-1}
            if field_iter[i] == 2:  # T entries
                L[i, i] = 1.0j * delta_n * diss_TC  # T_n, T_n part
                L[i, i - 2] = (n + delta) * k0  # T_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # T_n, T_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # T_n, T_{n-1}
                L[i, i + 2] = -A_T * kz * k0 / 2.0  # T_n, phi_{n+1}
                L[i, i - 6] = L[i, i + 2]  # T_n, phi_{n-1}
            if field_iter[i] == 3:  # C entries
                L[i, i] = 1.0j * tau * delta_n * diss_TC  # C_n, C_n part
                L[i, i - 3] = (n + delta) * k0 / R0  # C_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # C_n, C_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # C_n, C_{n-1}
                L[i, i + 1] = -A_C * kz * k0 / 2.0  # C_n, phi_{n+1}
                L[i, i - 7] = L[i, i + 1]  # C_n, phi_{n-1}
    # Now fill in the edges
    # First, most negative phi part
    L[0, 0] = 1.0j * Pr * delns_m[0] * diss  # phi_-N, phi_-N
    L[0, 1] = HB * kz  # phi_-N, psi_-N
    L[0, 2] = Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, T_-N
    L[0, 3] = -Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, C_-N
    L[0, 4] = -A_phi * kz * k0 * (k0**2.0 - delns_m[4]) / (2.0j * delns_m[0])  # phi_-N, phi_{-N + 1}
    # L[0, 4] = A_phi * kz * k0 * (k0 ** 2.0 - delns_m[4]) / (2.0j * delns_m[0])  # phi_-N, phi_{-N + 1}
    # Most positive phi part
    L[-4, -4] = 1.0j * Pr * delns_m[-4] * diss  # phi_N, phi_N
    L[-4, -3] = HB * kz  # phi_N, psi_N
    L[-4, -2] = Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, T_N
    L[-4, -1] = -Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, C_N
    L[-4, -8] = A_phi * kz * k0 * (k0**2.0 - delns_m[-8]) / (2.0j * delns_m[-4])  # phi_N, phi_{N-1}
    # L[-4, -8] = -A_phi * kz * k0 * (k0 ** 2.0 - delns_m[-8]) / (2.0j * delns_m[-4])  # phi_N, phi_{N-1}
    # Most negative psi part
    L[1, 0] = kz  # psi_-N, phi_-N
    L[1, 1] = 1.0j * DB * delns_m[1] * diss  # psi_-N, psi_-N
    L[1, 5] = A_phi * kz * k0 / 2.0j  # psi_-N, psi_{-N + 1}
    # Most positive psi part
    L[-3, -4] = kz  # psi_N, phi_N
    L[-3, -3] = 1.0j * DB * delns_m[-3] * diss  # psi_N, psi_N
    L[-3, -7] = -A_phi * kz * k0 / 2.0j  # psi_N, psi_{N-1}
    # Most negative T part
    L[2, 0] = (ns[0] + delta) * k0  # T_-N, phi_-N
    L[2, 2] = 1.0j * delns_m[2] * diss_TC  # T_-N, T_-N
    L[2, 4] = -A_T * kz * k0 / 2.0  # T_-N, phi_{-N + 1}
    L[2, 6] = A_phi * kz * k0 / 2.0j  # T_-N, T_{-N + 1}
    # Most positive T part
    L[-2, -4] = (ns[-1] + delta) * k0  # T_N, phi_N
    L[-2, -2] = 1.0j * delns_m[-2] * diss_TC  # T_N, T_N
    L[-2, -8] = -A_T * kz * k0 / 2.0  # T_N, phi_{N - 1}
    L[-2, -6] = -A_phi * kz * k0 / 2.0j  # T_N, T_{N - 1}
    # Most negative C part
    L[3, 0] = (ns[0] + delta) * k0 / R0  # C_-N, phi_-N
    L[3, 3] = 1.0j * delns_m[3] * tau * diss_TC  # C_-N, C_-N
    L[3, 4] = -A_C * kz * k0 / 2.0  # C_-N, phi_{-N + 1}
    L[3, 7] = A_phi * kz * k0 / 2.0j  # C_-N, C_{-N + 1}
    # Most positive C part
    L[-1, -4] = (ns[-1] + delta) * k0 / R0  # C_N, phi_N
    L[-1, -1] = 1.0j * delns_m[-1] * tau * diss_TC  # C_N, C_N
    L[-1, -8] = -A_T * kz * k0 / 2.0  # C_N, phi_{N - 1}
    L[-1, -5] = -A_phi * kz * k0 / 2.0j  # C_N, C_{N - 1}
    return L


def Sams_Lmat(N, f, k, m, A_Psi, A_T, A_C, flag, Pr, tau, R_0, Pm, H_b, no_TC=False):
    # This is Sam Reifenstein's code that I'm simply copy+pasting into mine and can verify that it works
    dim = (2 * N + 1) * 4
    D_b = Pr / Pm
    # matrix for linear system of perturbation growth
    A = np.zeros((dim, dim)) * 1j

    # loop over Fourier modes
    for i in range(-N, N + 1):
        k_mode = np.sqrt((i * k + f) ** 2 + m ** 2)  # $k_m$ in latex
        k_x = i * k + f  # $(f + (m+1)k_x)$ in latex
        k_z = m  # $k_z$ in latex

        P_k_mode = np.sqrt(((i + 1) * k + f) ** 2 + m ** 2)  # $k_{m+1}$ in latex
        N_k_mode = np.sqrt(((i - 1) * k + f) ** 2 + m ** 2)  # $k_{m-1}$ in latex

        # \psi field

        # -\lambda k_m^2 \psi_m -
        # 	  i k_x  k_z   E_{\psi} \left( k_{m+1}^2 \psi_{m+1} +  k_{m-1}^2 \psi_{m-1} \right)  +
        #  i k_x^3 k_z E_{\psi}\left( \psi_{m+1} + \psi_{m-1}\right)
        #  \end{equation}
        #  $$ =  \text{Pr} k_m^4 + i\text{Pr}(f + (m+1)k_x)(T_m -C_m) -
        # 	i H_b  k_z k_m^2 A_m  $$

        # (terms are in order that they appear in latex doc)

        PsiPsi_P = 1j * (-P_k_mode ** 2 * A_Psi * k_z * k + A_Psi * k ** 3 * k_z) / k_mode ** 2
        PsiPsi_N = 1j * (-N_k_mode ** 2 * A_Psi * k_z * k + A_Psi * k ** 3 * k_z) / k_mode ** 2

        PsiPsi = - Pr * k_mode ** 2
        PsiT = -1j * Pr * k_x / k_mode ** 2
        PsiC = 1j * Pr * k_x / k_mode ** 2

        if no_TC:
            PsiT = PsiT * flag
            PsiC = PsiC * flag

        PsiA = 1j * k_z * H_b

        # T field

        # \lambda T_m  +  i k_x  k_z E_{\psi}(T_{m+1} + T_{m-1})  + k_x k_z E_{T} (-\psi_{m+1} + \psi_{m-1}) + i(f + m k_x) \psi_m = -k_m^2 T_m

        TT_P = 1j * (-k * A_Psi * k_z)
        TT_N = 1j * (-k * A_Psi * k_z)

        TPsi_P = 1 * (1 * k * A_T * k_z)
        TPsi_N = 1 * (- 1 * k * A_T * k_z)

        TPsi = - (1j * k_x)
        TT = -k_mode ** 2

        # C filed

        # \lambda C_m  +  i k_x  k_z E_{\psi} (C_{m+1} + C_{m-1})  + k_x k_z E_{C} (-\psi_{m+1} + \psi_{m-1})  + i(f + m k_x) \frac{1}{R_0}\psi_m = -\tau k_m^2 C_m

        CC_P = 1j * (-k * A_Psi * k_z)
        CC_N = 1j * (-k * A_Psi * k_z)

        CPsi_P = 1 * (1 * k * A_C * k_z)
        CPsi_N = 1 * (- 1 * k * A_C * k_z)

        CPsi = -  (1j * k_x / R_0)
        CC = -tau * k_mode ** 2

        # A field

        # I think the second term below has a typo: the k_m^2 shouldn't be there. But it's not in the code, so it's
        # just a typo in the documentation/comment as far as I can tell.
        # \lambda A +    i k_m^2 k_x k_z E_{\psi}\left(  A_{m+1} +  A_{m-1} \right) = -D_B k_m^2 A_m + k_z i \psi_m

        AA_P = -1j * k * k_z * A_Psi
        AA_N = -1j * k * k_z * A_Psi

        AA = -D_b * k_mode ** 2

        APsi = 1j * k_z

        # interactions

        index = (N + i) * 4

        # same mode interactions

        A[index, index] = PsiPsi
        A[index, index + 1] = PsiT
        A[index, index + 2] = PsiC

        A[index + 1, index] = TPsi
        A[index + 1, index + 1] = TT

        A[index + 2, index] = CPsi
        A[index + 2, index + 2] = CC

        A[index + 3, index] = APsi
        A[index + 3, index + 3] = AA
        A[index, index + 3] = PsiA

        stride = 4

        # neighboring interactions

        # this is executed for all but the first i, as a way to handle the fact that the most-negative Fourier mode
        # doesn't have an m,m-1 interaction because we're truncating the m-1 part for that mode
        if (index > 0):
            A[index, index - stride] = PsiPsi_N
            A[index + 1, index - stride] = TPsi_N
            A[index + 2, index - stride] = CPsi_N
            A[index + 1, index - stride + 1] = TT_N
            A[index + 2, index - stride + 2] = CC_N

            A[index + 3, index - stride + 3] = AA_N

        # this is likewise executed for all but the last i
        if (index + stride + 3 < dim):
            A[index, index + stride] = PsiPsi_P
            A[index + 1, index + stride] = TPsi_P
            A[index + 2, index + stride] = CPsi_P
            A[index + 1, index + stride + 1] = TT_P
            A[index + 2, index + stride + 2] = CC_P

            A[index + 3, index + stride + 3] = AA_P

    return A


def Richs_build_matrix(N, k_z, R0, Pr, tau, l_f, E_psi, E_T, E_C, H_B, D_B):
    # Build the Fraser+(2023, F23) eigenproblem matrix given:
    #
    # N    - summation limit in eqn. (43) of F23
    # k_z  - vertical wavenumber squared
    # R0   - density ratio
    # Pr   - Prandtl number
    # tau  - inverse Lewis number
    # w_f  - vertical velocity amplitude of fastest-growing elevator mode
    # H_B  - Lorentz-force strength
    # D_B  - resistive diffusivity ratio

    # Evaluate fingering mode properties

    # lam_f, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    # l_f = np.sqrt(l2hat)

    # Evaluate the various E's

    # E_psi = w_f / (2 * l_f)  # F23, after eqn. (36)
    # E_T = -l_f * E_psi / (lam_f + l_f ** 2)  # F23, eqn. (42)
    # E_C = -l_f * E_psi / (R0 * (lam_f + tau * l_f ** 2))  # F23, eqn. (42)

    # Set up the matrix elements

    n = 4 * (2 * N + 1)

    Q = np.zeros((n, n), dtype=complex)

    for m in range(-N, N + 1):

        # Set up block indices

        i_p = 4 * (m + N)
        i_p_p = i_p + 4
        i_p_m = i_p - 4

        i_t = 4 * (m + N) + 1
        i_t_p = i_t + 4
        i_t_m = i_t - 4

        i_c = 4 * (m + N) + 2
        i_c_p = i_c + 4
        i_c_m = i_c - 4

        i_a = 4 * (m + N) + 3
        i_a_p = i_a + 4
        i_a_m = i_a - 4

        # Evaluate wavenumbers (after eqn. 39 of F23)

        k2_m = m ** 2 * l_f ** 2 + k_z ** 2

        k2_m_p = (m + 1) ** 2 * l_f ** 2 + k_z ** 2
        k2_m_m = (m - 1) ** 2 * l_f ** 2 + k_z ** 2

        # Set matrix elements

        # Eqn. (44), divided by k_2m

        Q[i_p, i_p] = -Pr * k2_m
        Q[i_p, i_t] = -1j * Pr * m * l_f / k2_m
        Q[i_p, i_c] = 1j * Pr * m * l_f / k2_m
        Q[i_p, i_a] = 1j * H_B * k_z

        if m > -N:
            Q[i_p, i_p_m] = 1j * l_f ** 3 * k_z * E_psi / k2_m - 1j * l_f * k_z * E_psi * k2_m_m / k2_m
        if m < N:
            Q[i_p, i_p_p] = 1j * l_f ** 3 * k_z * E_psi / k2_m - 1j * l_f * k_z * E_psi * k2_m_p / k2_m

        # Eqn. (45)

        Q[i_t, i_p] = -1j * m * l_f
        Q[i_t, i_t] = -k2_m

        if m > -N:
            Q[i_t, i_p_m] = -l_f * k_z * E_T
            Q[i_t, i_t_m] = -1j * l_f * k_z * E_psi
        if m < N:
            Q[i_t, i_p_p] = l_f * k_z * E_T
            Q[i_t, i_t_p] = -1j * l_f * k_z * E_psi

        # Eqn. (46)

        Q[i_c, i_p] = -1j * m * l_f / R0
        Q[i_c, i_c] = -tau * k2_m

        if m > -N:
            Q[i_c, i_p_m] = -l_f * k_z * E_C
            Q[i_c, i_c_m] = -1j * l_f * k_z * E_psi
        if m < N:
            Q[i_c, i_p_p] = l_f * k_z * E_C
            Q[i_c, i_c_p] = -1j * l_f * k_z * E_psi

        # Eqn. (47)

        Q[i_a, i_p] = 1j * k_z
        Q[i_a, i_a] = -D_B * k2_m

        if m > -N:
            Q[i_a, i_a_m] = -1j * l_f * k_z * E_psi
        if m < N:
            Q[i_a, i_a_p] = -1j * l_f * k_z * E_psi

    return Q


def Richs_build_matrix_real(N, k_z, R0, Pr, tau, l_f, E_psi, E_T, E_C, H_B, D_B):
    # Build the real version of the Fraser+(2023, F23) eigenproblem matrix

    Q = Richs_build_matrix(N, k_z, R0, Pr, tau, l_f, E_psi, E_T, E_C, H_B, D_B)

    # Transform variables

    n = 4 * (2 * N + 1)

    for m in range(-N, N + 1):
        # Set up block indices

        i_p = 4 * (m + N)
        i_t = 4 * (m + N) + 1
        i_c = 4 * (m + N) + 2
        i_a = 4 * (m + N) + 3

        # Scale columns

        Q[:, i_p] *= 1j ** m
        Q[:, i_t] *= 1j ** (m + 1)
        Q[:, i_c] *= 1j ** (m + 1)
        Q[:, i_a] *= 1j ** (m + 1)

        # Scale rows

        Q[i_p, :] /= 1j ** m
        Q[i_t, :] /= 1j ** (m + 1)
        Q[i_c, :] /= 1j ** (m + 1)
        Q[i_a, :] /= 1j ** (m + 1)

    # Return the matrix and the horizontal wavenumber

    return Q


def gamfromL(L, withmode=False):
    if withmode:
        w, v = np.linalg.eig(L)
        ind = np.argmax(-np.imag(w))
        return [-np.imag(w[ind]), v[:, ind]]
    else:
        w = np.linalg.eigvals(L)
        return np.max(-np.imag(w))


def omegafromL(L):
    w = np.linalg.eigvals(L)
    wsort = w[np.argsort(-np.imag(w))]
    return wsort[-1]


def gamfromparams(delta, M2, Re, Rm, k, N, ideal, withmode=False):
    L = Lmat(delta, M2, Re, Rm, k, N, ideal)
    return gamfromL(L, withmode)


def sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N, withmode=False, withTC=False, Sam=False,
                                get_frequency=False, test_Sam_no_TC=False):
    """
    Returns fastest growing mode's growth rate for Kolmogorov flow problem, set up so that you can input parameters
    in PADDIM units (except for k_star) rather than normalized to the sinusoidal flow

    Parameters
    ----------
    delta : Floquet parameter, determines periodicity in x of eigenmodes relative to sinusoidal base flow -- set to 0
    w : Amplitude of sinusoidal base flow in PADDIM units
    HB : Lorentz force coefficient in PADDIM units
    DB : Magnetic diffusion coefficient in PADDIM units
    Pr : Thermal Prandtl number
    tau : compositional diffusion coefficient / thermal diffusion coefficient
    R0 : density ratio
    k_star : KH wavenumber *normalized to finger wavenumber* (this is confusing, in hindsight, given previous params)
    N : Spectral resolution for calculating KH modes
    withmode : Boolean flag for whether or not to return
    withTC : Boolean flag for whether or not to include T and C (if false, uses the model in sec 5.1 of FRG23, if true,
             uses the model in sec 5.2)
    Sam : Boolean, whether or not to use Sam's code for contructing the matrix (which works!) vs mine (doesn't work!)
    get_frequency : Boolean, whether to return full complex eigenvalue or just the growth rate; currently only
                    implemented for with_TC=True and Sam=True

    Returns
    -------
    sigma : growth rate of the fastest-growing KH mode (at that wavenumber),
            but in units normalized to the sinusoidal base flow (should fix this maybe)
    """
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    if withTC:
        kz = k_star * lhat  # k_star is supposed to be kz normalized to finger width
        if Sam:
            A_psi = w / (2*lhat)
            A_T = -lhat * A_psi / (lamhat + l2hat)
            A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
            N_Sam = int((N-1)/2)  # Sam's definition of N is different than mine
            L = Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, R0, Pr/DB, HB, no_TC=test_Sam_no_TC)
            if withmode:
                w, v = np.linalg.eig(L)
            else:
                w = np.linalg.eigvals(L)
            ind = np.argmax(np.real(w))
            if get_frequency:
                evalue = w[ind]
            else:
                evalue = np.real(w[ind])
            if withmode:
                return [evalue, v[:, ind]]
            else:
                return evalue
        else:
            A_phi = w / lhat
            A_T = lhat * A_phi / (lamhat + l2hat)
            A_C = lhat * A_phi / (R0 * (lamhat + tau * l2hat))
            L = Lmat_withTC(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, lhat, kz, delta, N)
            return gamfromL(L, withmode)
    else:
        M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
        L = Lmat(delta, M2, Re, Rm, k_star, N)
        return gamfromL(L, withmode)


def omega_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N):
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
    L = Lmat(delta, M2, Re, Rm, k_star, N)
    return omegafromL(L)


def sigma_over_k_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N):
    return [sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N) for k_star in k_stars]


def gamma_over_k(delta, M2, Re, Rm, ks, N, ideal=False):
    return [gamfromL(Lmat(delta, M2, Re, Rm, k, N, ideal)) for k in ks]


def omega_over_k(delta, M2, Re, Rm, ks, N, ideal=False):
    return [omegafromL(Lmat(delta, M2, Re, Rm, k, N, ideal)) for k in ks]


def gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, badks_except=False, get_kmax=False):
    gammas = gamma_over_k(delta, M2, Re, Rm, ks, N, ideal)
    ind = np.argmax(gammas)
    gammax = gammas[ind]
    if badks_except and gammax > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
        if gammax == gammas[0]:
            raise KrangeError  # ('the k-range needs to be extended downwards')
        if gammax == gammas[-1]:
            raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
    if get_kmax:
        return [np.max(gammas), ks[ind]]
    else:
        return np.max(gammas)


def sigma_max_kscan_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N, badks_except=False, get_kmax=False):
    sigmas = sigma_over_k_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N)
    ind = np.argmax(sigmas)
    sigma_max = sigmas[ind]
    if badks_except and sigma_max > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
        if sigma_max == sigmas[0]:  # I don't remember why I separated them like this
            print(w, sigma_max)
            raise KrangeError  # ('the k-range needs to be extended downwards')
        if sigma_max == sigmas[-1]:
            raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
    if get_kmax:
        return [sigmas[ind], k_stars[ind]]
    else:
        return sigmas[ind]


def gammax_minus_lambda(w, lamhat, lhat, HB, Pr, DB, delta, ks, N, ideal=False, badks_exception=False, CH=1.66, tau=-1, R0=0):
    # a silly helper function that returns sigma - lambda rather than sigma
    # so that I can use root-finding packages to search for zeros of this
    # function

    # NOTE THIS MULTIPLIES sigma BY w_f l_f
    # (or rather -- gammax_kscan returns what HG19 calls sigma/(w_f l_f),
    # and so this quantity is multiplied by w_f l_f to get their sigma)
    #
    # NOTE THAT THE INPUT CH REFERS TO THE CH YOU SEE
    # IN EQ 31 IN HARRINGTON & GARAUD 2019

    M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
    krange = np.copy(ks)

    # The following while/try/except is for repeating the k scan if the max
    # occurs at the edges of the range of k sampled
    count = 0
    while True:
        count += 1
        try:
            out = gammax_kscan(delta, M2, Re, Rm, krange, N, ideal, badks_exception) * w * lhat - CH * lamhat
            break
        except ValueError:
            # the max occurs at the upper end of ks so seldomly
            # that I never bothered to implement this part
            print("w = ", w)  # for debugging
            raise
    return out


def gamma_over_k_withTC(delta, wf, HB, DB, Pr, tau, R0, ks, N, Sam=False, get_frequencies=False, get_evecs=False, test_Sam_no_TC=False, sparse_method=False, sparse2=False, k=3, pass_sigma=True, sparse_matrix=None, Richs_matrix=False):
    # note these ks are really k_stars not k_hats
    # As in, k_star = k_hat / lhat (where k_hat is \hat{k_z} in the paper)
    if sparse_method and len(ks) > 1 and Sam:
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
        lhat = np.sqrt(l2hat)
        kz_hats = ks * lhat
        A_psi = wf / (2 * lhat)
        A_T = -lhat * A_psi / (lamhat + l2hat)
        A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
        N_Sam = int((N - 1) / 2)  # Sam's definition of N is different than mine
        if Richs_matrix:
            L = Richs_build_matrix_real(N_Sam, kz_hats[0], R0, Pr, tau, lhat, A_psi, A_T, A_C, HB, DB)
        else:
            L = Sams_Lmat(N_Sam, 0, lhat, kz_hats[0], A_psi, A_T, A_C, 0, Pr, tau, R0, Pr / DB, HB, no_TC=test_Sam_no_TC)
        w, v = np.linalg.eig(L)
        # char_poly1 = fingering_modes.characteristic_polynomial(Pr, tau, R0, l2hat)
        # roots1 = char_poly1.roots()
        # root1 = np.max(np.real(roots1))
        root1 = lamhat
        char_poly2 = fingering_modes.characteristic_polynomial(Pr, tau, R0, 4 * l2hat)
        roots2 = char_poly2.roots()
        root2 = np.max(np.real(roots2))  # lamhat is the growth rate of elevator modes with kx = lhat. root2 is the growth rate of elevator modes with kx = 2*lhat
        # next, find the two eigenvalues/vectors closest to the elevator mode solutions
        # (how do I know which of those two I need? Refer back to perturbation theory in QM)
        # and the two closest to the m=2 root,
        # then enter a loop where, at each k, we use the 4 (ideally 2) eigenvectors from the previous k
        # as initial guesses for scipy.sparse.linalg.eigs.
        # Things that might go wrong:
        # - the lowest kz might not be low enough for the parasites to asymptotically approach elevator mode solutions
        # - I'm basing all my intuition for this method off of RGB star parameters (specifically fig 7a from FRG24), but it's very possible that other branches somehow matter in other parameters
        # - (for instance, my parasite_structures_vs_R0 plots for the RGB case show an oscillatory branch that's always unstable just never actually dominates, but what if in some other star it does dominate?)
        # - Maybe other elevator mode solutions (other roots at the same kx, or higher kx's) matter in other stars?
        closest_ws_to_root1 = np.argsort(np.abs(w - root1))
        elevator1_asymptote_ind1 = closest_ws_to_root1[0]
        elevator1_asymptote_ind2 = closest_ws_to_root1[1]
        elevator1_mode1 = v[:, elevator1_asymptote_ind1]
        elevator1_mode2 = v[:, elevator1_asymptote_ind2]

        closest_ws_to_root2 = np.argsort(np.abs(w - root2))
        elevator2_asymptote_ind1 = closest_ws_to_root2[0]
        elevator2_asymptote_ind2 = closest_ws_to_root2[1]
        elevator2_mode1 = v[:, elevator2_asymptote_ind1]
        elevator2_mode2 = v[:, elevator2_asymptote_ind2]

        elevator1_evalue1s = np.zeros(len(kz_hats), dtype=np.complex128)
        elevator1_evalue1s[0] = w[elevator1_asymptote_ind1]
        elevator1_evalue2s = np.zeros_like(elevator1_evalue1s)
        elevator1_evalue2s[0] = w[elevator1_asymptote_ind2]
        elevator2_evalue1s = np.zeros_like(elevator1_evalue1s)
        elevator2_evalue1s[0] = w[elevator2_asymptote_ind1]
        elevator2_evalue2s = np.zeros_like(elevator1_evalue1s)
        elevator2_evalue2s[0] = w[elevator2_asymptote_ind2]

        if get_evecs:
            elevator1_mode1s = np.zeros((len(kz_hats), len(elevator1_mode1)), dtype=np.complex128)
            elevator1_mode1s[0] = elevator1_mode1
            elevator1_mode2s = np.zeros_like(elevator1_mode1s)
            elevator1_mode2s[0] = elevator1_mode2
            elevator2_mode1s = np.zeros_like(elevator1_mode1s)
            elevator2_mode1s[0] = elevator2_mode1
            elevator2_mode2s = np.zeros_like(elevator1_mode1s)
            elevator2_mode2s[0] = elevator2_mode2
        # print(elevator1_evalue1s[0])
        # print(elevator1_evalue2s[0])
        # print(elevator2_evalue1s[0])
        # print(elevator2_evalue2s[0])
        # print(np.max(np.abs(elevator1_mode1)))
        # print(np.max(np.abs(elevator1_mode2)))
        # print(np.max(np.abs(elevator2_mode1)))
        # print(np.max(np.abs(elevator2_mode2)))
        for ki, kzhat in enumerate(kz_hats[1:], 1):
            if Richs_matrix:
                L = Richs_build_matrix_real(N_Sam, kzhat, R0, Pr, tau, lhat, A_psi, A_T, A_C, HB, DB)
            else:
                L = Sams_Lmat(N_Sam, 0, lhat, kzhat, A_psi, A_T, A_C, 0, Pr, tau, R0, Pr / DB, HB, no_TC=test_Sam_no_TC)
            if sparse_matrix == "csr":
                L = scipy.sparse.csr_matrix(L)
            if sparse_matrix == "csc":
                L = scipy.sparse.csc_matrix(L)
            if sparse_matrix == "dia":
                L = scipy.sparse.dia_matrix(L)
            # first try passing v0 but not sigma. Should do a speed test comparing what happens when you pass sigma too, but that will affect the 'which' argument
            # note I have no idea what the ncz argument is or does or if we should touch it
            # note also that if we provide sigma, we can probably speed things up by calculating "OPinv" analytically and providing it, rather than having scipy calculate it numerically
            # note also that this *might* be sped up by using Rich's version of L that has all-real coefficients
            if pass_sigma:
                out = scipy.sparse.linalg.eigs(L, k=1, sigma=elevator1_evalue1s[ki-1], v0=elevator1_mode1, which='LM')
                elevator1_evalue1s[ki] = out[0]
                # elevator1_mode1s[ki] = out[1][0]
                elevator1_mode1 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, sigma=elevator1_evalue2s[ki-1], v0=elevator1_mode2, which='LM')
                elevator1_evalue2s[ki] = out[0]
                # elevator1_mode2s[ki] = out[1][0]
                elevator1_mode2 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, sigma=elevator2_evalue1s[ki-1], v0=elevator2_mode1, which='LM')
                elevator2_evalue1s[ki] = out[0]
                # elevator2_mode1s[ki] = out[1][0]
                elevator2_mode1 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, sigma=elevator2_evalue2s[ki-1], v0=elevator2_mode2, which='LM')
                elevator2_evalue2s[ki] = out[0]
                # elevator2_mode2s[ki] = out[1][0]
                elevator2_mode2 = out[1][:, 0]
            else:  # preliminary testing suggests that this route is trash
                out = scipy.sparse.linalg.eigs(L, k=1, v0=elevator1_mode1, which='LM')
                elevator1_evalue1s[ki] = out[0]
                # elevator1_mode1s[ki] = out[1][0]
                elevator1_mode1 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, v0=elevator1_mode2, which='LM')
                elevator1_evalue2s[ki] = out[0]
                # elevator1_mode2s[ki] = out[1][0]
                elevator1_mode2 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, v0=elevator2_mode1, which='LM')
                elevator2_evalue1s[ki] = out[0]
                # elevator2_mode1s[ki] = out[1][0]
                elevator2_mode1 = out[1][:, 0]
                out = scipy.sparse.linalg.eigs(L, k=1, v0=elevator2_mode2, which='LM')
                elevator2_evalue2s[ki] = out[0]
                # elevator2_mode2s[ki] = out[1][0]
                elevator2_mode2 = out[1][:, 0]
            if get_evecs:
                elevator1_mode1s[ki] = elevator1_mode1
                elevator1_mode2s[ki] = elevator1_mode2
                elevator2_mode1s[ki] = elevator2_mode1
                elevator2_mode2s[ki] = elevator2_mode2
            # print(ki, kzhat)
            # print(elevator1_evalue1s[ki])
            # print(elevator1_evalue2s[ki])
            # print(elevator2_evalue1s[ki])
            # print(elevator2_evalue2s[ki])
            # print(np.max(np.abs(elevator1_mode1s[ki])))
            # print(np.max(np.abs(elevator1_mode2s[ki])))
            # print(np.max(np.abs(elevator2_mode1s[ki])))
            # print(np.max(np.abs(elevator2_mode2s[ki])))
        if get_evecs:
            return elevator1_evalue1s, elevator1_mode1s, elevator1_evalue2s, elevator1_mode2s, elevator2_evalue1s, elevator2_mode1s, elevator2_evalue2s, elevator2_mode2s
        else:
            return elevator1_evalue1s, elevator1_evalue2s, elevator2_evalue1s, elevator2_evalue2s
    if sparse2:
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
        lhat = np.sqrt(l2hat)
        kz_hats = ks * lhat
        A_psi = wf / (2 * lhat)
        A_T = -lhat * A_psi / (lamhat + l2hat)
        A_C = -lhat * A_psi / (R0 * (lamhat + tau * l2hat))
        N_Sam = int((N - 1) / 2)  # Sam's definition of N is different than mine
        if Richs_matrix:
            L = Richs_build_matrix_real(N_Sam, kz_hats[0], R0, Pr, tau, lhat, A_psi, A_T, A_C, HB, DB)
        else:
            L = Sams_Lmat(N_Sam, 0, lhat, kz_hats[0], A_psi, A_T, A_C, 0, Pr, tau, R0, Pr / DB, HB, no_TC=test_Sam_no_TC)
        w, v = np.linalg.eig(L)
        w_sort = np.argsort(-np.real(w))
        w_sorted = w[w_sort]
        evalues = np.zeros((len(kz_hats), k), dtype=np.complex128)
        evalues[0] = w_sorted[:k]
        evec = v[:, w_sort[0]]
        for ki, kzhat in enumerate(kz_hats[1:], 1):
            if Richs_matrix:
                L = Richs_build_matrix_real(N_Sam, kzhat, R0, Pr, tau, lhat, A_psi, A_T, A_C, HB, DB)
            else:
                L = Sams_Lmat(N_Sam, 0, lhat, kzhat, A_psi, A_T, A_C, 0, Pr, tau, R0, Pr / DB, HB, no_TC=test_Sam_no_TC)
            if sparse_matrix == "csr":
                L = scipy.sparse.csr_matrix(L)
            if sparse_matrix == "csc":
                L = scipy.sparse.csc_matrix(L)
            if sparse_matrix == "dia":
                L = scipy.sparse.dia_matrix(L)
            out = scipy.sparse.linalg.eigs(L, k=k, sigma=w_sorted[0], v0=evec, which='LM')
            w = out[0]
            w_sort = np.argsort(-np.real(w))
            w_sorted = w[w_sort]
            evalues[ki] = w_sorted
            evec = out[1][:, w_sort[0]]  # I don't think this v0 business is actually speeding things up
        return evalues[:, 0]
    else:
        return [sigma_from_fingering_params(delta, wf, HB, DB, Pr, tau, R0, k, N, withTC=True, Sam=Sam, get_frequency=get_frequencies, test_Sam_no_TC=test_Sam_no_TC) for k in ks]


def gammax_kscan_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, badks_except=False, get_kmax=False, Sam=False, test_Sam_no_TC=False, sparse=False):
    gammas = gamma_over_k_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, Sam=Sam, test_Sam_no_TC=test_Sam_no_TC, sparse2=sparse, sparse_matrix='csr')
    ind = np.argmax(gammas)
    gammax = gammas[ind]
    if badks_except and gammax > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
        if gammax == gammas[0]:
            lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
            if np.isclose(lamhat, gammax) or gammax < lamhat:
                # Sometimes, the parasite growth rate vs k is a curve that peaks at k=0 (where the mode is simply an elevator mode with no modification from the background shear) 
                # and then decreases from there. In that case, gammax == gammas[0] will be true no matter how low your ks array goes, and this whole check is unhelpful.
                print('Warning: the peak growth rate is occurring at the lowest k (but probably just be due to the fact that elevator modes are the fastest-growing DDC modes).')
            else:
                raise KrangeError  # ('the k-range needs to be extended downwards')
        if gammax == gammas[-1]:
            raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
    if get_kmax:
        return [np.max(gammas), ks[ind]]
    else:
        return np.max(gammas)


def gammax_minus_lambda_withTC(w, lamhat, delta, HB, DB, Pr, tau, R0, ks, N, badks_exception=False, C2=0.33, Sam=False, test_Sam_no_TC=False, sparse=False):
    # a silly helper function that returns sigma - lambda rather than sigma
    # so that I can use root-finding packages to search for zeros of this
    # function
    # Note that C2 refers to the C_2 in Fraser, Reifenstein, Garaud 2023 (e.g. eq 28)

    # The following while/try/except is for repeating the k scan if the max
    # occurs at the edges of the range of k sampled
    count = 0
    while True:
        count += 1
        try:
            out = gammax_kscan_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, badks_exception, Sam=Sam, test_Sam_no_TC=test_Sam_no_TC, sparse=sparse) * C2 - lamhat
            break
        except ValueError:
            # the max occurs at the upper end of ks so seldomly
            # that I never bothered to implement this part
            print("w = ", w)  # for debugging
            raise
    return out
