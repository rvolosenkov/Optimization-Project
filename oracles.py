import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    def __init__(self, A, b):
        super(LeastSquaresOracle, self).__init__()
        self._A = A
        self._b = b
    
    def func(self, x):
        return 0.5 * sum((self._A @ x - self._b) ** 2)
    
    def grad(self, x):
        return self._A.T @ (self._A @ x - self._b)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    
    def __init__(self, regcoef):
        super(L1RegOracle, self).__init__()
        self._regcoef = regcoef
    
    def func(self, x):
        return self._regcoef * sum(abs(x))
    
    def prox(self, x, alpha):
        mul = alpha * self._regcoef
        return np.vectorize(lambda z: z - mul if z > mul else z + mul if z < -mul else 0)(x)


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    
    def __init__(self, A, b, regcoef):
        super(LassoProxOracle, self).__init__(LeastSquaresOracle(A, b), L1RegOracle(regcoef))
        self._A = A
        self._b = b
        self._regcoef = regcoef

    def duality_gap(self, x):
        mu = min(1, self._regcoef / np.linalg.norm(self._A.T @ (self._A @ x - self._b), ord=np.inf)) * (self._A @ x - self._b)
        return 0.5 * sum((self._A @ x - self._b) ** 2) + self._regcoef * sum(abs(x)) + 0.5 * sum(mu ** 2) + self._b.T @ mu


class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    def __init__(self, A, b, regcoef):
        super(LassoNonsmoothOracle, self).__init__()
        self._A = A
        self._b = b
        self._regcoef = regcoef
    
    def func(self, x):
        return 0.5 * sum((self._A @ x - self._b) ** 2) + self._regcoef * sum(abs(x))
    
    def subgrad(self, x):
        return self._A.T @ (self._A @ x - self._b) + self._regcoef * np.sign(x)
    
    def duality_gap(self, x):
        mu = self._A @ x - self._b
        return 0.5 * sum((self._A @ x - self._b) ** 2) + self._regcoef * sum(abs(x)) + 0.5 * sum(mu ** 2) + mu.T @ self._b


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, regcoef / np.linalg.norm(ATAx_b, ord=np.inf)) * Ax_b
    return 0.5 * sum(Ax_b ** 2) + regcoef * sum(abs(x)) + 0.5 * sum(mu ** 2) + b.T @ mu
    

def create_lasso_prox_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)

