"""Return Shao-Strawderman Estimator for a given choice of p."""


import numpy as np


from functools import cached_property
from numpy.linalg import norm
from scipy.integrate import quadrature
from scipy.optimize import root_scalar, NonlinearConstraint, minimize
from .estimators import js_ve


def m(n, p):
    return (p / 2) + n - 1


def s(t, p):
    """Return reflection of t about p - 1."""
    return 2*(p - 1) - t


def f(t, n, p):
    return np.exp(-t / 2) * (t ** m(n, p))


def F(t, n, p):
    return f(t, n, p) - f(s(t, p), n, p)


def F_star(t, n, p):
    return f(t, n, p) + f(s(t, p), n, p)


def d2F(t, n, p):
    """Return second derivative of F."""
    m_n = m(n, p)
    return f(t, n, p) * (1/4 - (m_n/t) * (1 - (m_n-1)/t))


@np.vectorize
def g(t, p_star, p):
    if t < p - 1:
        return g(s(t, p), p_star, p)
    elif t < p_star:
        return 2*p_star - p - t
    else:
        return t - p


@np.vectorize
def dg(t, p_star, p):
    if t < p - 1:
        return -dg(s(t, p), p_star, p)
    elif t < p_star:
        return -1
    else:
        return 1


def ind(t, lower, upper):
    """Return indicator for lower <= t <= upper."""
    res = np.piecewise(
        t,
        [t < lower, (lower <= t) & (t <= upper), t > upper],
        [0, 1, 0]
    )
    return res


class GenerateSS:
    def __init__(self, p):
        self.p = p

    @cached_property
    def j(self):
        # TODO: Find out a numerical way to do this.
        # j = self.p + 1
        # self.j = j
        return self.p + 1

    def c_(self, i):
        p = self.p
        def integrand(t): return F(t, i, p)
        def integral(lower, upper): return quadrature(integrand, lower, upper)
        res = root_scalar(
            lambda x: integral(p-1, x) - integral(x, p),
            method="bisect", bracket=[p-1, p]
        ).root

        if (i == 0) and (res > p - 1 + np.sqrt(2)/2):
            raise Exception("c_0 is too large.")

        return res

    def b(self, x):
        return 1 - np.sqrt(2)*(x - (self.p-1))

    @cached_property
    def p_star(self, j):
        c = [self.c_(i) for i in range(j)]

        def con(x): return (self.b(x)) / (np.minimum(1/2, self.p - x))
        nlc = NonlinearConstraint(con, 0, 1)
        bounds = [c[0], min(*c[1:], self.p - 1 + 1 / np.sqrt(2))]

        result = minimize(lambda x: x, x0=np.mean(bounds),
                          bounds=[bounds], constraints=[nlc])

        return result.fun

    def g(self, t):
        return g(t, self.p_star, self.p)

    def dg(self, t):
        return dg(t, self.p_star, self.p)

    @property
    def A(self):
        b = self.b(self.p_star)
        j, p = self.j, self.p
        return 1 - np.exp(1 - b) * ((p - 2 + b) / (p - b)) ** m(j, p)

    def B_(self, i):
        # p_star = self.p_star
        # p = self.p
        # num = quadrature(lambda t: np.vectorize(dg)(t, p_star, p)
        # * f(t, i, p), p-2, p)[0]
        # num = int_F(i, p, p_star, p) - int_F(i, p, p-1, p_star)
        # num = quadrature(lambda t: -f(t, i, p), p-2, 2*(p-1) - p_star)[0] +
        # quadrature(lambda t: -f(t, i, p), p_star,p)[0]
        # + quadrature(lambda t: f(t, i, p), 2*(p-1) - p_star, p_star)[0]
        # den = integrate(lambda t: F_star(t, i-1, p) * g_1(t, p_star, p)**2,
        #  lower=p-1, upper=p_star) + integrate(lambda t: F_star(t, i-1, p)
        # * g_2(t, p_star, p)**2, lower=p_star, upper=p)
        # den = quadrature(lambda t: (np.vectorize(g)(t, p_star, p) ** 2)
        # * f(t, i-1, p), p-2, p)[0]
        # return num / den
        pass

    @cached_property
    def B(self):
        return 4 * min([self.B_(i) for i in range(self.j)])

    @property
    def a(self):
        return min(self.B, 2 * self.A * (self.p - 2) * self.b(self.p_star)) / 2

    def estimator(self, p_star=None, a=None):
        """Return Shao-Strawderman Estimator."""
        p = self.p
        if p_star is None:
            p_star = self.p_star
        if a is None:
            a = self.a

        def ss(x):
            sqn = norm(x) ** 2
            ss_coeff = ind(sqn, p-2, p) * self.g(sqn) / sqn
            return js_ve(x, p) - ss_coeff*x

        return ss
