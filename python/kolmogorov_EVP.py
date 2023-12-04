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


def Sams_Lmat(N, f, k, m, A_Psi, A_T, A_C, flag, Pr, tau, R_0, Pm, H_b):
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

        # if (no_TC):
            # PsiT = PsiT * flag
            # PsiC = PsiC * flag

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

        if (index > 0):
            A[index, index - stride] = PsiPsi_N
            A[index + 1, index - stride] = TPsi_N
            A[index + 2, index - stride] = CPsi_N
            A[index + 1, index - stride + 1] = TT_N
            A[index + 2, index - stride + 2] = CC_N

            A[index + 3, index - stride + 3] = AA_N

        if (index + stride + 3 < dim):
            A[index, index + stride] = PsiPsi_P
            A[index + 1, index + stride] = TPsi_P
            A[index + 2, index + stride] = CPsi_P
            A[index + 1, index + stride + 1] = TT_P
            A[index + 2, index + stride + 2] = CC_P

            A[index + 3, index + stride + 3] = AA_P

    return A


def gamfromL(L, withmode=False):
    w, v = np.linalg.eig(L)
    if withmode:
        ind = np.argmax(-np.imag(w))
        return [-np.imag(w[ind]), v[:, ind]]
    else:
        return np.max(-np.imag(w))


def omegafromL(L):
    w, v = np.linalg.eig(L)
    wsort = w[np.argsort(-np.imag(w))]
    return wsort[-1]


def gamfromparams(delta, M2, Re, Rm, k, N, ideal, withmode=False):
    L = Lmat(delta, M2, Re, Rm, k, N, ideal)
    return gamfromL(L, withmode)


def sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N, withmode=False, withTC=False, Sam=False,
                                get_frequency=False):
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
            L = Sams_Lmat(N_Sam, 0, lhat, kz, A_psi, A_T, A_C, 0, Pr, tau, R0, Pr/DB, HB)
            w, v = np.linalg.eig(L)
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


def gamma_over_k_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, Sam=False, get_frequencies=False):
    # note these ks are really k_stars not k_hats
    # As in, k_star = k_hat / lhat (where k_hat is \hat{k_z} in the paper)
    return [sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k, N, withTC=True, Sam=Sam, get_frequency=get_frequencies) for k in ks]


def gammax_kscan_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, badks_except=False, get_kmax=False, Sam=False):
    gammas = gamma_over_k_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, Sam=Sam)
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


def gammax_minus_lambda_withTC(w, lamhat, delta, HB, DB, Pr, tau, R0, ks, N, badks_exception=False, C2=0.33, Sam=False):
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
            out = gammax_kscan_withTC(delta, w, HB, DB, Pr, tau, R0, ks, N, badks_exception, Sam=Sam) * C2 - lamhat
            break
        except ValueError:
            # the max occurs at the upper end of ks so seldomly
            # that I never bothered to implement this part
            print("w = ", w)  # for debugging
            raise
    return out
