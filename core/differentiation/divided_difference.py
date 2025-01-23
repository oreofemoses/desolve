# import numpy as np
# from typing import List, Tuple
# import sympy
#
#
# def create_divided_difference_table(x: np.ndarray, f: np.ndarray) -> List[np.ndarray]:
#     """
#     Creates the divided difference table for a given set of data.
#
#     Parameters
#     ----------
#     x : array_like
#         Array of x values where the function is evaluated.
#     f : array_like
#         Array of corresponding function values f(x).
#
#     Returns
#     -------
#     list of array_like
#         A list where each element represents a column of the divided difference table.
#     """
#     if len(x) != len(f):
#         raise ValueError("The lengths of x and f arrays must be the same.")
#
#     table = [np.array(f)]
#     current_diff = np.array(f)
#     current_x = x
#     col = 0
#     while len(current_diff) > 1:
#         new_diff = []
#         col += 1
#         for i in range(0, len(current_diff) - 1):
#             print(new_diff, i, col)
#             diff = (current_diff[i + 1] - current_diff[i]) / (current_x[i + col] - current_x[i])
#             new_diff.append(diff)
#         current_diff = np.array(new_diff)
#         table.append(current_diff)
#     return table
#
#
#
# def forward_diff_divided(f=None, at: float = None, order: int = 1, terms: int = 5, x: np.ndarray = None,
#                          f_values: np.ndarray = None, h: float = None) -> float:
#     """
#     Approximates the derivative of a function using the divided differences method for non-equispaced data.
#
#     Parameters
#     ----------
#     f : callable, optional
#         The function to differentiate.
#     at : float
#         The value at which to evaluate the derivative
#     order : int, optional
#         The order of the derivative (1 for first derivative, 2 for second), by default 1
#     x: np.ndarray, optional
#        The x values to be used, overrides f, a and b
#     f_values: np.ndarray, optional
#        The f(x) values to be used, overrides f, a and b
#     h: float
#         The step size to be used when a function is provided
#
#     Returns
#     -------
#     float
#         The approximated derivative at the specified point x.
#
#     Raises
#     ------
#     ValueError;
#         If the input arrays are not of the same length.
#         If a valid combination of the parameters is not provided
#     """
#
#     if x is None and f is None:
#         raise ValueError("If x is not present, the function f must be specified")
#     elif x is not None and f_values is not None:
#         table = create_divided_difference_table(x, f_values)
#         x_sym = sympy.Symbol('x')
#         f_approx = table[0][0]
#         for i in range(1, len(table)):
#             coeff = 1
#             for j in range(0, i):
#                 coeff *= (x_sym - x[j])
#             f_approx += coeff * table[i][0]
#
#         df_approx_sym = f_approx
#         for _ in range(order):
#             df_approx_sym = sympy.diff(df_approx_sym, x_sym)
#         df_callable = sympy.lambdify(x_sym, df_approx_sym, "numpy")
#
#         return df_callable(at)
#     elif f is not None and at is not None and h is not None:
#         x_val = np.linspace(at - h, at + h,
#                             terms + 1)  # the num = 4(in this implementation) should be dynamic and be changed
#         f_values = np.array([f(val) for val in x_val])
#         table = create_divided_difference_table(x_val, f_values)
#         x_sym = sympy.Symbol('x')
#         f_approx = table[0][0]
#         for i in range(1, len(table)):
#             coeff = 1
#             for j in range(0, i):
#                 coeff *= (x_sym - x_val[j])
#             f_approx += coeff * table[i][0]
#         df_approx_sym = f_approx
#         for _ in range(order):
#             df_approx_sym = sympy.diff(df_approx_sym, x_sym)
#         df_callable = sympy.lambdify(x_sym, df_approx_sym, "numpy")
#
#         return df_callable(at)
#     else:
#         raise ValueError("Invalid parameters for divided difference")

import numpy as np
from typing import List
import sympy


def create_divided_difference_table(x: np.ndarray, f: np.ndarray) -> List[np.ndarray]:
    """
    Creates the divided difference table for a given set of data.

    Parameters
    ----------
    x : array_like
        Array of x values where the function is evaluated.
    f : array_like
        Array of corresponding function values f(x).

    Returns
    -------
    list of array_like
        A list where each element represents a column of the divided difference table.
    """
    if len(x) != len(f):
        raise ValueError("The lengths of x and f arrays must be the same.")

    table = [np.array(f)]
    current_diff = np.array(f)
    current_x = x
    col = 0
    while len(current_diff) > 1:
        new_diff = []
        col += 1
        for i in range(0, len(current_diff) - 1):
            diff = (current_diff[i + 1] - current_diff[i]) / (current_x[i + col] - current_x[i])
            new_diff.append(diff)
        current_diff = np.array(new_diff)
        table.append(current_diff)
    return table


def forward_diff_divided(x: np.ndarray, f_values: np.ndarray, at: float, order: int = 1) -> float:
    """
    Approximates the derivative of a function using the divided differences method for non-equispaced data.

    Parameters
    ----------
    x: np.ndarray
        The x values to be used
    f_values: np.ndarray
        The f(x) values to be used
    at : float
        The value at which to evaluate the derivative
    order : int, optional
        The order of the derivative (1 for first derivative, 2 for second), by default 1

    Returns
    -------
    float
        The approximated derivative at the specified point x.

    Raises
    ------
    ValueError;
        If the input arrays are not of the same length.
        If x or f_values is not specified
    """

    if x is None or f_values is None:
        raise ValueError("x and f_values must be specified")
    if len(x) != len(f_values):
        raise ValueError("The lengths of x and f arrays must be the same.")

    table = create_divided_difference_table(x, f_values)
    x_sym = sympy.Symbol('x')
    f_approx = table[0][0]
    for i in range(1, len(table)):
        coeff = 1
        for j in range(0, i):
            coeff *= (x_sym - x[j])
        f_approx += coeff * table[i][0]

    df_approx_sym = f_approx
    for _ in range(order):
        df_approx_sym = sympy.diff(df_approx_sym, x_sym)
    df_callable = sympy.lambdify(x_sym, df_approx_sym, "numpy")

    return df_callable(at)