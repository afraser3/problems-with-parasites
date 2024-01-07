"""
Various tools for working with parasite models of the fingering instability

Mainly provides a function for finding the finger velocity w_f at which
the fingering instability saturates via secondary KH. Also provides
some convenient functions for calculating Nusselt numbers, etc.

Methods
-------
w_f(Pr, tau, R0, HB, DB, ks, N, delta=0.0, ideal=False, badks_exception=True, CH=1.66, lamhat=0.0, l2hat=0.0)
    Uses scipy.optimize's root-finding algorithms to solve for the
    finger velocity w_f such that the parasitic KH mode's growth rate
    is equal to CH * lambda_f, where lambda_f is the most-unstable
    finger mode's growth rate
w_f_HG19(Pr, tau, R0, HB, CH=1.66)
    Solves for w_f via the HG19 model using their approximate analytical expression for sigma_KH vs w (their eq 30)
    This is done by relying on scipy.optimize.root_scalar, where I'm giving it HG19's eq 32 (my function HG19_eq32),
    the derivative of that equation with respect to w (my function dEQ32dw), and an initial guess, so that it uses
    the Newton solve by default.
HG19_eq32
    Used by the Newton solve in w_f_HG19, see w_f_HG19
dEQ32dw
    Used by the Newton solve in w_f_HG19, see w_f_HG19
NuC(tau, w, lamhat, l2hat, KB=1.24)
    Evaluates the parameterization for Nu_C in terms of w given in
    Harrington & Garaud and references therein
NuT( ... )
    Like NuC but returns Nu_T
"""

import numpy as np
from scipy import optimize as opt
import kolmogorov_EVP
import fingering_modes
from joblib import Parallel, delayed
from os import cpu_count


def w_f(Pr, tau, R0, HB, DB, ks, N, delta=0.0, ideal=False, badks_exception=True, get_kmax=False, CH=1.66, lamhat=0.0,
        l2hat=0.0):
    """
    For a given fingering-unstable system (specified by Pr, tau, etc.), solves for the amplitude w_f of a
    primary/elevator mode such that the growth rate sigma of a parasitic KH mode (which depends on w_f) equals
    CH * lambda_f, where lambda_f is the growth rate of the primary, and CH is a universal constant.
    See Harrington & Garaud 2019.

    First it solves for lambda_f and the associated wavenumber using fingering_modes.gaml2max (which just solves
    Eqs. 19-20 in Garaud et al ApJ 2015, or equivalently that one Brown paper). Then it finds the roots/zeros of the
    expression sigma - CH * lambda_f, where sigma is the growth rate of the KH parasite. It does this by evaluating
    sigma as a function of w over a range of w (where w_f is the w that makes sigma-CH*lambda_f = 0). Root finding
    is done using scipy.optimize.root_scalar, where sigma-CH*lambda_f is the function scipy is told to find the roots
    of, over the domain w.

    For a given w, sigma is evaluated in kolmogorov_EVP, which solves for the eigenvalues of a sinusoidal flow with a
    flow-aligned, uniform magnetic field via Floquet analysis of the MHD Orr-Sommerfeld equation (see attached .tex
    notes), then solving for the eigenvalues of the resulting matrix. It does this for each k in the array ks, and
    returns the largest growth rate ("largest" meaning first maximized over all growth rates at each individual k, then
    maximized over each k in ks). N=33 is more than enough resolution for this step for the same parameters as the HB=10
    simulation in Harrington&Garaud. I usually use N=17. At lower Pr, higher N quickly becomes necessary. Convergence
    checks recommended.

    Uses the default of scipy.optimize.root_scalar for this case, which I think might be brentq.
    The bounds I set by default could probably be improved upon, thereby increasing the speed
    with which this converges.

    Parameters
    ----------
    Pr : Thermal Prandtl number
    tau : Compositional diffusion coefficient / thermal diffusion coefficient
    R0 : Density ratio
    HB : Lorentz force coefficient in PADDIM units (see Harrington&Garaud 2019)
    DB : Magnetic diffusion coefficient in PADDIM units
    ks : Range of wavenumbers to calculate KH modes at
    N : Spectral resolution of Floquet analysis for KH modes
    delta : Floquet parameter, determines periodicity in x of eigenmodes relative to sinusoidal base flow -- set to 0
    ideal : Whether or not to include viscosity and resistivity (haven't implemented with stratified=True)
    badks_exception : Throw error if fastest-growing mode is on either end of ks array?
    get_kmax : Return kmax along with w_f?
    CH : Model constant
    lamhat : Growth rate of fastest-growing elevator mode; recalculates if not provided
    l2hat : Squared wavenumber of fastest-growing elevator mode; recalculates if not provided

    Returns
    -------

    """

    if lamhat == 0.0 or l2hat == 0.0:
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)

    # Next, set up initial guesses of wf to perform bracketed search in the intervial w1 < wf < w2.
    # This part is tough and I never got it right, that's why you see a couple commented-out guesses and my final guess
    # wasn't perfect, hence the need to do some guess-and-check before handing things off to scipy.

    # For lower bound on w_f, take the hydro expression lambda_f/(CB*l_f) (see the bit of text between Fig 3 and eq 27
    # of HG19). For the upper bound, take the strongly-magnetized limit and multiply by 4
    # wbounds = [np.pi*lamhat/lhat, 4.0*np.sqrt(2.0*HB)]
    # that upper bound is too low when R0 gets small. I'm guessing lambda gets to be too large for the w^2 ~ 2HB
    # scaling to be appropriate. Could also be because dissipation is included -> increases wf above HG19 guess
    w1 = 2.0 * np.pi * lamhat / lhat  # HB = 0 solution
    w2 = np.sqrt(2.0 * HB)  # HB -> infinity solution
    wbounds = [w1, w1 + w2]  # the answer never lies below w1, but sometimes it's above w2
    args = (lamhat, lhat, HB, Pr, DB, delta, ks, N, ideal, badks_exception, CH)
    count = 0  # Used for troubleshooting/monitoring just how bad the original wbounds are
    while True:
        count += 1  # Every time we fail to get the wbounds right, increment count to see how many retries were needed
        # in standard use-case, the right-most bound on wf is the one that fails, so the following block just assumes
        # we need to iteratively improve w2. The ValueError exception below is for handling what happens when
        # w1 (the left-most bound on wf) is a bad guess, and that only ever happens when tinkering with
        # certain model parameters that don't need to be tinkered with.
        try:
            rbound_eval = kolmogorov_EVP.gammax_minus_lambda(wbounds[1], lamhat, lhat, HB, Pr, DB, delta, ks, N,
                                                             ideal, badks_exception, CH)
            while rbound_eval < 0:
                print('adjusting bounds preemptively, count=', count)
                count += 1
                wbounds = [wbounds[1], 2.0 * wbounds[1]]
                rbound_eval = kolmogorov_EVP.gammax_minus_lambda(wbounds[1], lamhat, lhat, HB, Pr, DB, delta, ks, N,
                                                                 ideal, badks_exception, CH)
            if count > 1:
                print('i shall proceed')
            sol = opt.root_scalar(kolmogorov_EVP.gammax_minus_lambda, args=args, bracket=wbounds)
            # return sol.root
            break
        except ValueError:
            # Now that I've added the preceding rbound_eval business, the following is mostly obsolete
            # except when playing with delta and/or CH, since it was almost always the right bound, not the left,
            # that was the issue
            lbound_eval = kolmogorov_EVP.gammax_minus_lambda(wbounds[0], lamhat, lhat, HB, Pr, DB, delta, ks, N,
                                                             ideal, badks_exception, CH)
            rbound_eval = kolmogorov_EVP.gammax_minus_lambda(wbounds[1], lamhat, lhat, HB, Pr, DB, delta, ks, N,
                                                             ideal, badks_exception, CH)
            if lbound_eval * rbound_eval > 0:  # want to end up with rbound_eval > 0
                if delta == 0.0 and CH == 1.66:
                    wbounds = [0.5 * wbounds[0], wbounds[1]]
                else:
                    wbounds = [0.25 * wbounds[0], wbounds[1]]
                print("adjusting wbounds, count=", count)
                if count > 8:
                    print(wbounds)
                    raise
            else:
                raise
    if get_kmax:
        M2, Re, Rm = kolmogorov_EVP.KHparams_from_fingering(sol.root, lhat, HB, Pr, DB)
        # TODO: this is redundant: I'm solving for kmax by re-calculating gammax
        gammax, kmax = kolmogorov_EVP.gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, badks_except=False,
                                                   get_kmax=get_kmax)
        return [sol.root, kmax]
    else:
        return sol.root


def w_f_withTC(Pr, tau, R0, HB, DB, ks, N, badks_exception=True, get_kmax=False, C2=0.33, lamhat=0.0,
        l2hat=0.0, Sam=False):
    """
    Same as w_f above (see that docstring) but using the extended "with T&C" model of Fraser, Reifenstein, Garaud 2023

    Parameters
    ----------
    Pr : Thermal Prandtl number
    tau : Compositional diffusion coefficient / thermal diffusion coefficient
    R0 : Density ratio
    HB : Lorentz force coefficient in PADDIM units (see Harrington&Garaud 2019)
    DB : Magnetic diffusion coefficient in PADDIM units
    ks : Range of wavenumbers to calculate KH modes at, note these are not exactly the \hat{k_z} in our paper, but
         rather that number divided by lhat
    N : Spectral resolution of Floquet analysis for KH modes
    badks_exception : Throw error if fastest-growing mode is on either end of ks array?
    get_kmax : Return kmax along with w_f? This may be useful for estimating finger height
    C2 : Model constant
    lamhat : Growth rate of fastest-growing elevator mode; recalculates if not provided
    l2hat : Squared wavenumber of fastest-growing elevator mode; recalculates if not provided

    Returns
    -------

    """

    if lamhat == 0.0 or l2hat == 0.0:
        lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)

    # Next, set up initial guesses of wf to perform bracketed search in the intervial w1 < wf < w2.
    # This part is tough and I never got it right, that's why you see a couple commented-out guesses and my final guess
    # wasn't perfect, hence the need to do some guess-and-check before handing things off to scipy.

    # For lower bound on w_f, take the hydro expression lambda_f/(CB*l_f) (see the bit of text between Fig 3 and eq 27
    # of HG19). For the upper bound, take the strongly-magnetized limit and multiply by 4
    # wbounds = [np.pi*lamhat/lhat, 4.0*np.sqrt(2.0*HB)]
    # that upper bound is too low when R0 gets small. I'm guessing lambda gets to be too large for the w^2 ~ 2HB
    # scaling to be appropriate. Could also be because dissipation is included -> increases wf above HG19 guess
    w1 = 2.0 * np.pi * lamhat / lhat  # HB = 0 solution
    w2 = np.sqrt(2.0 * HB)  # HB -> infinity solution
    wbounds = [w1, w1 + w2]  # the answer never lies below w1, but sometimes it's above w2
    args = (lamhat, 0, HB, DB, Pr, tau, R0, ks, N, badks_exception, C2, Sam)
    count = 0  # Used for troubleshooting/monitoring just how bad the original wbounds are
    while True:
        count += 1  # Every time we fail to get the wbounds right, increment count to see how many retries were needed
        # in standard use-case, the right-most bound on wf is the one that fails, so the following block just assumes
        # we need to iteratively improve w2. The ValueError exception below is for handling what happens when
        # w1 (the left-most bound on wf) is a bad guess, and that only ever happens when tinkering with
        # certain model parameters that don't need to be tinkered with.
        try:
            rbound_eval = kolmogorov_EVP.gammax_minus_lambda_withTC(wbounds[1], lamhat, 0, HB, DB, Pr, tau, R0, ks, N,
                                                                    badks_exception, C2, Sam)
            while rbound_eval < 0:
                print('adjusting bounds preemptively, count=', count)
                count += 1
                wbounds = [wbounds[1], 2.0 * wbounds[1]]
                rbound_eval = kolmogorov_EVP.gammax_minus_lambda_withTC(wbounds[1], lamhat, 0, HB, DB, Pr, tau, R0, ks,
                                                                        N, badks_exception, C2, Sam)
            if count > 1:
                print('i shall proceed')
            sol = opt.root_scalar(kolmogorov_EVP.gammax_minus_lambda_withTC, args=args, bracket=wbounds)
            # return sol.root
            break
        except ValueError:
            # Now that I've added the preceding rbound_eval business, the following is mostly obsolete
            lbound_eval = kolmogorov_EVP.gammax_minus_lambda_withTC(wbounds[0], lamhat, 0, HB, DB, Pr, tau, R0, ks, N,
                                                                    badks_exception, C2, Sam)
            rbound_eval = kolmogorov_EVP.gammax_minus_lambda_withTC(wbounds[1], lamhat, 0, HB, DB, Pr, tau, R0, ks, N,
                                                                    badks_exception, C2, Sam)
            if lbound_eval * rbound_eval > 0:  # want to end up with rbound_eval > 0
                wbounds = [0.25 * wbounds[0], wbounds[1]]
                print("adjusting wbounds, count=", count)
                if count > 8:
                    print(wbounds)
                    raise
            else:
                raise
    if get_kmax:
        # M2, Re, Rm = kolmogorov_EVP.KHparams_from_fingering(sol.root, lhat, HB, Pr, DB)
        # TODO: this is redundant: I'm solving for kmax by re-calculating gammax
        # delta, w, HB, DB, Pr, tau, R0, ks, N, badks_except=False, get_kmax=False
        gammax, kmax = kolmogorov_EVP.gammax_kscan_withTC(0, sol.root, HB, DB, Pr, tau, R0, ks, N, badks_except=False,
                                                   get_kmax=get_kmax, Sam=Sam)
        return [sol.root, kmax]
    else:
        return sol.root


def HG19_eq32(w, Pr, tau, R0, HB, CH=1.66):
    """
    Simply evaluates Eq. (32) in Harrington & Garaud 2019.
    Specifically, evaluates LHS - RHS (so it should evaluate to zero if w is the solution)

    Parameters
    ----------
    w: w_f in the equation
    Pr: Prandtl number, used to calculate lambda_f and l_f
    tau: ratio of diffusivities, used to calculate lambda_f and l_f
    R0: density ratio, used to calculate lambda_f and l_f
    HB: Lorentz force coefficient
    CH: fitting parameter

    Returns
    -------
    LHS - RHS of Eq. (32)
    """

    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    LHS = 0.5 * w**2.0 - HB
    RHS1 = (CH*lamhat/(0.42*lhat))**1.5
    RHS2 = np.sqrt(w)
    return LHS - (RHS1*RHS2)
    # return 0.5 * w**2.0 - HB - (CH*lamhat/(0.42*lhat))**1.5 * np.sqrt(w)


def dEQ32dw(w, Pr, tau, R0, HB, CH=1.66):
    """
    Derivative with respect to w of the function HG19_eq32 above. For list of inputs, see HG19_eq32.
    Where HG19_eq32 evaluates F(w) = LHS - RHS (LHS and RHS of Eq. 32), so that
    F(w) = 0 for the solution w, this function returns dF/dw for use in root-finding algorithm
    """

    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    return w - 0.5 * (CH*lamhat/(0.42*lhat))**1.5 / np.sqrt(w)


def w_f_HG19(Pr, tau, R0, HB, CH=1.66):
    """
    Uses a root-finding algorithm to solve EQ32 in HG19 for w

    Parameters
    ----------
    Pr: Prandtl number, used to calculate lambda_f and l_f
    tau: ratio of diffusivities, used to calculate lambda_f and l_f
    R0: density ratio, used to calculate lambda_f and l_f
    HB: Lorentz force coefficient
    CH: fitting parameter

    Returns
    -------
    w: the shear velocity that solves Eq. 32
    """

    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    # Initial guess: take the max between the strong-field limit and the hydro limit
    w0 = max(np.sqrt(2.0*HB), 2.0 * np.pi * lamhat/lhat)
    result = opt.root_scalar(HG19_eq32, args=(Pr, tau, R0, HB, CH), x0=w0, fprime=dEQ32dw)
    root = result.root  # this is wf
    if root > 0:
        return result
    # In early versions of this, sometimes I'd end up with negative predicted wf values. So I did the below
    # as a fallback. But at some point along the way, this issue vanished.
    else:
        print('w_f_HG19 initial search returned negative w, resorting to brute force bracket search')
        w1 = 10.0 * CH**1.5 * (np.sqrt(2.0*HB) + 2.0 * np.pi * lamhat/lhat)
        try:
            result = opt.root_scalar(HG19_eq32, args=(Pr, tau, R0, HB, CH),
                                     bracket=[0.0, w1])
        except ValueError:
            print("w1 = ", w1)
            print("CH = ", CH)
            raise
        return result


def NuC(tau, w, lamhat, l2hat, KB=1.24):
    # Note KB in HG19 is C_1 in FRG23, and 1.24 is the value advocated by HG19 (and BGS13) but 0.62 by FRG23
    # when using the with_TC model
    return 1 + KB * w ** 2.0 / (tau * (lamhat + tau * l2hat))


def NuT(w, lamhat, l2hat, KB=1.24):
    return 1 + KB * w ** 2.0 / (lamhat + l2hat)


def gamma_tot(tau, R0, w, lamhat, l2hat, KB=1.24):
    return R0 * NuT(w, lamhat, l2hat, KB) / (tau * NuC(tau, w, lamhat, l2hat, KB))


# def parasite_results(R0, HB, Pr, tau, DB, ks, N, lamhat, l2hat,
#                      eq32=False, double_N=False, delta=0.0, ideal=False, CH=1.66, badks_exception=True,
#                      withTC=False, Sam=False, C1=1.24, C2=0.33):
def parasite_results(R0, HB, Pr, tau, DB, ks, N, lamhat, l2hat,
                     eq32=False, double_N=False, delta=0.0, ideal=False, badks_exception=True,
                     withTC=False, Sam=False, C1=1.24, C2=1/1.66):
    # NOTE C2 and CH mean basically the same thing, one is just the inverse of the other. Pre-FRG23, CH=1.66 was fiducial. Now C2=0.33 is.
    CH = 1/C2
    if withTC:
        wf, k_max = w_f_withTC(Pr, tau, R0, HB, DB, ks, N, badks_exception=badks_exception, get_kmax=True, C2=C2, lamhat=lamhat, l2hat=l2hat, Sam=Sam)
    else:
        if eq32:
            wf = w_f_HG19(Pr, tau, R0, HB).root
        elif double_N:
            wf, k_max = w_f(Pr, tau, R0, HB, DB, ks, int(2 * N - 1), delta, ideal, CH=CH,
                            lamhat=lamhat, l2hat=l2hat, get_kmax=True)
        else:
            wf, k_max = w_f(Pr, tau, R0, HB, DB, ks, N, delta, ideal, CH=CH,
                            lamhat=lamhat, l2hat=l2hat, get_kmax=True, badks_exception=badks_exception)
    # NOTE kb and C1 are the same thing. 1.24 is the pre-FRG23 value, 0.62 is the new one
    # kb = 1.24
    fc = C1 * wf**2.0/(R0*(lamhat + tau * l2hat))
    ft = C1 * wf**2.0/(lamhat + l2hat)
    nu_t = NuT(wf, lamhat, l2hat)
    nu_c = NuC(tau, wf, lamhat, l2hat)
    m2, re = kolmogorov_EVP.KHparams_from_fingering(wf, np.sqrt(l2hat), HB, Pr, DB)[:2]
    gamma_tot = R0 * nu_t / (tau * nu_c)
    if eq32:
        names = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star"]
        return dict(zip(names, [fc, ft, nu_c, nu_t, gamma_tot, wf, re, m2]))
    else:
        names = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star", "kmax-star"]
        return dict(zip(names, [fc, ft, nu_c, nu_t, gamma_tot, wf, re, m2, k_max]))


# def results_vs_r0(r0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=False, double_N=False, delta=0.0, ideal=False,
#                   CH=1.66, badks_exception=True, withTC=False, Sam=False, C1=1.24, C2=0.33):
def results_vs_r0(r0s, HB, Pr, tau, DB, ks, N, lamhats, l2hats, eq32=False, double_N=False, delta=0.0, ideal=False,
                  badks_exception=True, withTC=False, Sam=False, C1=1.24, C2=1/1.66):
    if eq32:
        names = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star"]
    else:
        names = ["FC", "FT", "NuC", "NuT", "gammatot", "wf", "Re-star", "HB-star", "kmax-star"]
    results_scan = {name: np.zeros_like(r0s) for name in names}
    if eq32:
        # just a simple wrapper for parasite results call so that the parallel call is cleaner
        def res1(ri,r0):
            # result_ri = parasite_results(r0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=True, CH=CH, C1=C1)
            result_ri = parasite_results(r0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=True, C2=C2, C1=C1)
            return [ri, result_ri]

        parallel_results = Parallel(n_jobs=cpu_count())(delayed(res1)(ri,r0) for ri, r0 in enumerate(r0s))

        # Now unpack into results_scan
        for res in parallel_results:
            ri = res[0]
            result_ri = res[1]
            for name in names:
                results_scan[name][ri] = result_ri[name]
    else:
        # just a simple wrapper for parasite results call so that the parallel call is cleaner
        def res1(ri,r0):
            print('solving for R0 = ', r0, "ri =", ri)
            result_ri = parasite_results(r0, HB, Pr, tau, DB, ks, N, lamhats[ri], l2hats[ri], eq32=False,
                                         double_N=double_N, delta=delta, ideal=ideal, badks_exception=badks_exception, withTC=withTC, Sam=Sam, C1=C1, C2=C2)
            return [ri, result_ri]

        parallel_results = Parallel(n_jobs=cpu_count())(delayed(res1)(ri,r0) for ri, r0 in enumerate(r0s))

        # Now unpack into results_scan
        for res in parallel_results:
            ri = res[0]
            result_ri = res[1]
            for name in names:
                results_scan[name][ri] = result_ri[name]
    return results_scan
