import sympy
import numpy as np
from typing import List


def create_forward_difference_table(x: np.ndarray, f: np.ndarray) -> List[np.ndarray]:
    """
    Creates the forward difference table for a given set of data.

    Parameters
    ----------
    x : array_like
        Array of x values where the function is evaluated. Must be equispaced.
    f : array_like
        Array of corresponding function values f(x).

    Returns
    -------
    list of array_like
        A list where each element represents a column of the forward difference table.
    """

    if len(x) != len(f):
        raise ValueError("The lengths of x and f arrays must be the same.")

    table = [np.array(f)]
    current_diff = np.array(f)

    while len(current_diff) > 1:
        new_diff = np.diff(current_diff)
        table.append(new_diff)
        current_diff = new_diff
    return table

def forward_diff(f=None, x0: float = None, h: float = None, order: int = 1, terms: int = 2, x: np.ndarray = None,
                 f_values: np.ndarray = None, at: float = None) -> float:
    """
    Approximates the derivative of a function using forward finite differences.

    Parameters
    ----------
    f: callable
       The function we want to differentiate
    x0 : float
        The point at which to evaluate the derivative.
    h : float
        The step size.
    order : int, optional
        The order of the derivative to compute (1 for first derivative), by default 1
    terms: int
        The number of terms to use for the derivative calculation, by default 2
    x: np.ndarray, optional
       The x values to be used, overrides f, h and x0
    f_values: np.ndarray, optional
       The f(x) values to be used, overrides f, h and x0
    at: float, optional
        The point at which we are approximating the value of the derivative, only used when x and f_values is provided.


    Returns
    -------
    float
        The approximated derivative at the specified point x.

    Raises
    ------
    ValueError
        If order is not a positive integer.
        If number of terms is less than 1
        If the parameters do not match.
    """
    if order < 1 or not isinstance(order, int):
        raise ValueError("Order must be a positive integer")
    if terms < 1:
        raise ValueError("Terms must be equal or greater than 1")
    if terms <= order:
        terms = order + 5
    if x is None and at is None and f is not None:
        if x0 is None or h is None:
            raise ValueError("When no x or at is given, h and x0 must be provided")
        x = np.array([x0 + i * h for i in range(0, terms + 1)])
        f_values = np.array([f(val) for val in x])
        table = create_forward_difference_table(x, f_values)
        s = 0
        idx = 0
        h = x[1] - x[0]
    elif x is not None and f_values is not None and at is not None:
        if at not in x:
            raise ValueError("The 'at' point must be in x")
        h = np.diff(x)[0] if len(x) > 1 else 1
        s = (at - x[0]) / h
        table = create_forward_difference_table(x, f_values)
        terms = len(x) - 1
        x0 = at
    else:
        raise ValueError("Invalid parameters for forward_diff method")


    s_sym = sympy.Symbol('s')
    f_approx = table[0][0]  # start with the base function value f(x0)
    for i in range(1, terms + 1):
        coeff = 1
        for j in range(1, i + 1):
            coeff *= (s_sym - j + 1) / j
        f_approx += coeff * table[i][0]
    df_approx_sym = f_approx
    for _ in range(order):
        df_approx_sym = sympy.diff(df_approx_sym, s_sym)
    df_callable = sympy.lambdify(s_sym, df_approx_sym, "numpy")
    return df_callable(s) / (h ** order)