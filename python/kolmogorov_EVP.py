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
import scipy


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


def Deln(k, n, delta, finger_norm=False, k0=1.0):  # \Delta_n in my notes. So simple, probably shouldn't be a function
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
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # the n over which we sum the Fourier series
    ms = list(range(-N + 1, N + 1, 1))  # this is going to be twice as long so we can loop over each n twice, once for phi and once for psi

    # the following few lines just sets up arrays of Delta_n
    delns = [Deln(k, n, delta) for n in ns]
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
        # deltanp1 = delns_m[i+2]
        # deltanm1 = delns_m[i-2]
        if i > 1 and i < len(ms) - 2:  # avoid entries near edges
            deltanp1 = delns_m[i + 2]
            deltanm1 = delns_m[i - 2]
            if m % 2 == 0:  # phi entries
                n = m / 2
                # phi_n, phi_n part
                L[i, i] = (1.0j) * (diss / Re) * deltan
                # phi_n, psi_n part
                L[i, i + 1] = M2 * k
                # phi_n, phi_n+1
                L[i, i + 2] = -k * (1 - deltanp1) / (2.0j * deltan)
                if not np.isfinite(L[i, i + 2]):
                    # Pretty sure I can get rid of this now -- I was debugging 0/0 errors, which I think only happen
                    # if you try to solve the system at k=0, which isn't interesting. And if it is, then the way to
                    # go about it is to multiply both sides of the linear system by Delta_n, and solve as a
                    # generalized eigenvalue problem
                    print(-k * (1 - deltanp1))
                    print(2.0j * deltan)
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
    # TODO: fully replace gamfromL with this
    L = Lmat(delta, M2, Re, Rm, k, N, ideal)
    return gamfromL(L, withmode)


def sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N, withmode=False):
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
    k_star : KH wavenumber *normalized to finger wavenumber* (this is confusing, in hindsight, given previous parames)
    N : Spectral resolution for calculating KH modes
    withmode : Boolean flag for whether or not to return

    Returns
    -------
    sigma : growth rate of the fastest-growing KH mode (at that wavenumber),
            but in units normalized to the sinusoidal base flow (should fix this maybe)
    """
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
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
