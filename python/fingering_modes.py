"""Various functions for working with fingering instability

Provides tools for things like calculating fingering growth rate


Methods
-------
rfromR(r0, tau)
    Calculates the reduced density ratio r
lamguess(pr, tau, r0)
    provides an initial guess for the most-unstable fingering growth rate

    (lam is short for lambda, which can't be used as a variable name
    in python)

k2guess(pr, tau, r0)
    provides initial guess for corresponding wavenumber (squared)
eq1(lam, k2, pr, tau, r0)
    Evaluates the characteristic polynomial for fingering instability,
    returns 0 iff lam, k2 are a valid growth rate/wavenumber^2 pair
eq2(lam, k2, pr, tau, r0)
    Evaluates to 0 iff d(lam)/d(k2) = 0
fun(x, pr, tau, r0, passk1=False)
    A vector equivalent of eq1 and eq2. If x = [lam, k2], fun
    just returns [eq1, eq2]
jac(x, pr, tau, r0, passk1=False)
    The jacobian matrix for fun wrt lam and k2
gaml2max(pr, tau, r0)
    Uses scipy.optimize's rootsolving tools to find the roots x=[lam,k2]
    of [eq1, eq2] and returns them -- i.e., returns the most-unstable
    growth rate of the fingering instability and the corresponding
    wavenumber squared
"""
import numpy as np
from scipy import optimize as opt


def rfromR(r0, tau):
    return (r0 - 1.0) / (-1.0 + 1.0 / tau)


def lamguess(pr, tau, r0):
    r = rfromR(r0, tau)
    if r < tau:
        return np.sqrt(pr) - pr * np.sqrt(1.0 + tau / pr)
    else:
        if r > 0.5:
            return 2.0 * pr * (tau / pr) * ((1.0 / 3.0) * (1.0 - r)) ** (3.0 / 2.0) / (
                    1.0 - (1.0 - r) * (1.0 + tau / pr) / 3.0)
        else:
            return np.sqrt(pr * tau / r) - pr * np.sqrt(1 + tau / pr)


def k2guess(pr, tau, r0):
    r = rfromR(r0, tau)
    if r < tau:
        return (1.0 + tau / pr) ** (-0.5) - np.sqrt(pr) * (1.0 + (tau / pr) * (1.0 + tau / pr) ** (-2.0))
    else:
        if r > 0.5:
            return np.sqrt((1.0 - r) / 3.0)
        else:
            return np.sqrt((1.0 + tau / pr) ** (-0.5) - 2.0 * np.sqrt(r * tau / pr) * (1.0 + tau / pr) ** (-5.0 / 2.0))


def eq1(lam, k2, pr, tau, r0):
    b2 = k2 * (1.0 + pr + tau)
    b1 = k2 ** 2.0 * (tau * pr + pr + tau) + pr * (1.0 - 1.0 / r0)
    b0 = k2 ** 3.0 * tau * pr + k2 * pr * (tau - 1.0 / r0)
    return lam ** 3.0 + b2 * lam ** 2.0 + b1 * lam + b0


def eq2(lam, k2, pr, tau, r0):
    c2 = 1.0 + pr + tau
    c1 = 2.0 * k2 * (tau * pr + tau + pr)
    c0 = 3.0 * k2 ** 2.0 * tau * pr + pr * (tau - 1.0 / r0)
    return c2 * lam ** 2.0 + c1 * lam + c0


def fun(x, pr, tau, r0, passk1=False):
    # returns f(x) where f = [eq1, eq2] and x = [lam, k2]
    if passk1:  # if x[1] is k instead of k^2
        return [eq1(x[0], x[1] ** 2.0, pr, tau, r0), eq2(x[0], x[1] ** 2.0, pr, tau, r0)]
    else:
        return [eq1(x[0], x[1], pr, tau, r0), eq2(x[0], x[1], pr, tau, r0)]


def jac(x, pr, tau, r0, passk1=False):
    # Jacobian of fun(x)
    lam = x[0]
    if passk1:  # is x[1] k or k^2?
        k2 = x[1] ** 2.0
    else:
        k2 = x[1]
    b2 = k2 * (1.0 + pr + tau)
    db2dk2 = 1.0 + pr + tau  # derivative of b2 wrt k2
    b1 = k2 ** 2.0 * (tau * pr + pr + tau) + pr * (1.0 - 1.0 / r0)
    db1dk2 = 2.0 * k2 * (tau * pr + pr + tau)
    # b0 = k2 ** 3.0 * tau * pr + k2 * pr * (tau - 1.0 / r0)
    db0dk2 = 3.0 * k2 ** 2.0 * tau * pr + pr * (tau - 1.0 / r0)

    j11 = 3.0 * lam ** 2.0 + 2.0 * b2 * lam + b1  # d(eq1)/dlam
    j12 = lam ** 2.0 * db2dk2 + lam * db1dk2 + db0dk2  # d(eq1)/dk2
    if passk1:
        j12 = j12 * 2.0 * x[1]  # d(eq1)/dk = d(eq1)/dk2 * dk2/dk

    c2 = 1.0 + pr + tau
    c1 = 2.0 * k2 * (tau * pr + tau + pr)
    dc1dk2 = c1 / k2
    # c0 = 3.0 * k2 ** 2.0 * tau * pr + pr * (tau - 1.0 / r0)
    dc0dk2 = 6.0 * k2 * tau * pr

    j21 = 2.0 * c2 * lam + c1
    j22 = lam * dc1dk2 + dc0dk2
    if passk1:
        j22 = j12 * 2.0 * x[1]
    return [[j11, j12], [j21, j22]]


def gaml2max(pr, tau, r0):
    """
    Uses scipy.optimize.root with the above functions to find lambda_FGM and l^2_FGM (AKA k^2 AKA k2)
    It first tries to solve for lambda and k2 by treating k2 as its own real variable (since that's faster).
    But if it settles on a negative values of k2, then it tries again, this time solving for k instead of k2.
    """
    sol = opt.root(fun, [lamguess(pr, tau, r0), k2guess(pr, tau, r0)], args=(pr, tau, r0), jac=jac, method='hybr')
    x = sol.x
    if sol.x[1] < 0:  # if a negative k^2 is returned, then try again but solve for k instead of k^2
        sol = opt.root(fun, [lamguess(pr, tau, r0), np.sqrt(k2guess(pr, tau, r0))], args=(pr, tau, r0, True), jac=jac,
                       method='hybr')
        test = fun(sol.x, pr, tau, r0, True)
        if np.allclose(test, np.zeros_like(test)) is False:
            raise ValueError("fingering_modes.gaml2max is broken!")
        x = sol.x
        x[1] = x[1] ** 2.0  # whatever calls gaml2max expects k^2, not k
    return x


def characteristic_polynomial(pr, tau, r0, l2):
    # the following are just the coefficients from BGS13 eq 19, or see `eq1` above
    a2 = l2 * (1.0 + pr + tau)
    a1 = l2 ** 2.0 * (tau * pr + pr + tau) + pr * (1.0 - 1.0 / r0)
    a0 = l2 ** 3.0 * tau * pr + l2 * pr * (tau - 1.0 / r0)
    char_pol = np.polynomial.polynomial.Polynomial([a0, a1, a2, 1.0])
    return char_pol

