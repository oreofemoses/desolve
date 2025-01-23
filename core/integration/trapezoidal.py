import numpy as np
from typing import Callable

def composite_trapezoidal(f: Callable[[float], float] = None, a: float = None, b: float = None, n: int = None, x : np.ndarray = None, f_values: np.ndarray = None) -> float:
    """
    Approximates the definite integral of a function using the composite trapezoidal rule.

    Parameters
    ----------
    f : callable, optional
        The function to integrate.
    a : float, optional
        The lower limit of integration.
    b : float, optional
        The upper limit of integration.
    n : int, optional
        The number of subintervals.
    x : np.ndarray, optional
        The x values to be used, overrides f, a and b
    f_values : np.ndarray, optional
      The values of the function at the x values

    Returns
    -------
    float
        The approximated integral value.

    Raises
    ------
    ValueError
        If n is not a positive integer.
    """
    if x is None and f is not None and a is not None and b is not None and n is not None:
      if not isinstance(n, int) or n <= 0:
          raise ValueError("n must be a positive integer")
      h = (b - a) / n
      x = np.linspace(a, b, n + 1)
      f_values = np.array([f(val) for val in x])
    elif x is not None and f_values is not None:
        h = np.diff(x)[0] if len(x) > 1 else 1
    else:
        raise ValueError("Invalid parameters provided.")

    integral = (h / 2) * (f_values[0] + 2 * np.sum(f_values[1:-1]) + f_values[-1] if len(x) > 1 else f_values[0] )
    return integral