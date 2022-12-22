from collections import defaultdict
import numpy as np
from numpy.linalg import norm
import time
import datetime


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    x_k = x_0
    x_star = x_0
    best_func_value = oracle.func(x_star)
    history = {
        'func': [],
        'time': [],
        'duality_gap': []
    }
    if x_0.size <= 2:
        history['x'] = []
    start_time = time.time()
    for k in range(max_iter):
        alpha_k = alpha_0 / np.sqrt(k+1)
        subgrad = oracle.subgrad(x_k)
        x_k = x_k - alpha_k * subgrad / np.linalg.norm(subgrad, ord=2)
        func_value = oracle.func(x_k)
        history['func'].append(func_value)
        if func_value < best_func_value:
            x_star = x_k
            if x_0.size <= 2:
                history['x'].append(x_star)
            best_func_value = oracle.func(x_star)
        duality_gap = oracle.duality_gap(x_k)
        history['duality_gap'].append(duality_gap)
        if display:
            print('x_' + str(k+1) + ' = ' + str(x_k) + '\nf(x_' + str(k+1) + ') = ' + str(func_value))
            print('gap = ' + str(duality_gap))
            print('x_star = ' + str(x_star) + '\nf(x_star) = ' + str(best_func_value))
        if abs(duality_gap) < tolerance:
            history['time'].append(time.time() - start_time)
            return x_star, 'success', history if trace else None
        history['time'].append(time.time() - start_time)
    return x_star, 'iterations_exceeded', history if trace else None


def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    x_k = x_0
    x_star = x_0
    y = 0
    L = L_0
    best_func_value = oracle.func(x_star)
    history = {
        'func': [],
        'time': [],
        'duality_gap': []
    }
    if x_0.size <= 2:
        history['x'] = []
    start_time = time.time()
    for k in range(max_iter):
        while True:
            y = oracle.prox(x_k - (1 / L) * oracle.grad(x_k), 1 / L)
            if oracle._f(y) <= oracle._f(x_k) + sum(oracle.grad(x_k) * (y - x_k)) + (L / 2) * sum((y - x_k) ** 2):
                break
            L *= 2
        x_k = y
        L = np.max(L_0, L / 2)
        func_value = oracle.func(x_k)
        history['func'].append(func_value)
        if func_value < best_func_value:
            x_star = x_k
            if x_0.size <= 2:
                history['x'].append(x_star)
            best_func_value = oracle.func(x_star)
        duality_gap = oracle.duality_gap(x_k)
        history['duality_gap'].append(duality_gap)
        if display:
            print('x_' + str(k+1) + ' = ' + str(x_k) + '\nf(x_' + str(k+1) + ') = ' + str(func_value))
            print('gap = ' + str(duality_gap))
            print('x_star = ' + str(x_star) + '\nf(x_star) = ' + str(best_func_value))
            print('L = ' + str(L))
        if abs(duality_gap) < tolerance:
            history['time'].append(time.time() - start_time)
            return x_star, 'success', history if trace else None
        history['time'].append(time.time() - start_time)
    return x_star, 'iterations_exceeded', history if trace else None


def accelerated_proximal_gradient_descent(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(y_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """
    x_star = x_0
    y = 0
    z = 0
    L = L_0
    A_k = 0
    y_k = x_0
    v_k = x_0
    grad_sum = 0
    best_func_value = oracle.func(x_star)
    history = {
        'func': [],
        'time': [],
        'duality_gap_yk': [],
        'duality_gap_zk': []
    }
    start_time = time.time()
    for k in range(max_iter):
        while True:
            roots = np.roots(np.array([L, -1, -A_k]))
            a = roots[roots > 0][0]
            t = a / (a + A_k)
            z = t * v_k + (1 - t) * y_k
            y = oracle.prox(z - (1 / L) * oracle.grad(z), 1 / L)
            if oracle._f(y) <= oracle._f(z) + sum(oracle.grad(z) * (y - z)) + (L / 2) * sum((y - z) ** 2):
                break
            L *= 2
        a_k = a
        A_k += a
        y_k = y
        z_k = z
        grad_sum += a_k * oracle.grad(z_k)
        v_k = oracle.prox(x_0 - grad_sum, A_k)
        L = np.max(L_0, L / 2)
        func_value_yk = oracle.func(y_k)
        history['func'].append(func_value_yk)
        if func_value_yk < best_func_value:
            x_star = y_k
            best_func_value = oracle.func(x_star)
        func_value_zk = oracle.func(z_k)
        if func_value_zk < best_func_value:
            x_star = z_k
            best_func_value = oracle.func(x_star)
        duality_gap_yk = oracle.duality_gap(y_k)
        history['duality_gap_yk'].append(duality_gap_yk)
        duality_gap_zk = oracle.duality_gap(z_k)
        history['duality_gap_zk'].append(duality_gap_zk)
        if display:
            print('y_' + str(k) + ' = ' + str(y_k) + '\nf(y_' + str(k) + ') = ' + str(func_value_yk))
            print('z_' + str(k) + ' = ' + str(z_k) + '\nf(z_' + str(k) + ') = ' + str(func_value_zk))
            print('gap_yk = ' + str(duality_gap_yk) + '\ngap_zk = ' + str(duality_gap_zk))
            print('x_star = ' + str(x_star) + '\nf(x_star) = ' + str(best_func_value))
            print('L = ' + str(L))
        if abs(duality_gap_yk) < tolerance or abs(duality_gap_zk) < tolerance:
            history['time'].append(time.time() - start_time)
            return x_star, 'success', history if trace else None
        history['time'].append(time.time() - start_time)
    return x_star, 'iterations_exceeded', history if trace else None
