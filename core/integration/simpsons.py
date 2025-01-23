import numpy as np
from typing import Callable, List, Literal

def composite_simpsons(f: Callable[[float], float] = None, a: float = None, b: float = None, n: int = None,
                       rule: Literal["1/3", "3/8"] = "1/3", x : np.ndarray = None, f_values : np.ndarray = None) -> float:
    """
    Approximates the definite integral of a function using the composite Simpson's 1/3 or 3/8 rule.

    Parameters
    ----------
    f : callable, optional
        The function to integrate.
    a : float, optional
        The lower limit of integration.
    b : float, optional
        The upper limit of integration.
    n : int, optional
        The number of subintervals. For 1/3 rule, it must be even, for the 3/8 it must be a multiple of 3.
    rule: Literal["1/3", "3/8"]
        The Simpson's rule to be used, by default 1/3
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
        If n is not a positive even integer for 1/3 rule, or a multiple of 3 for 3/8 rule
    """
    if rule == "1/3":
         if  x is None and f_values is None and f is not None and a is not None and b is not None and n is not None:
            if not isinstance(n, int) or n <= 0 or n % 2 != 0:
                raise ValueError("n must be a positive even integer for Simpson's 1/3 rule")
            h = (b - a) / n
            x = np.linspace(a, b, n + 1)
            f_values = np.array([f(val) for val in x])

            integral = (h / 3) * (f_values[0] + 4 * np.sum(f_values[1:-1:2]) + 2 * np.sum(f_values[2:-1:2]) + f_values[-1])
            return integral
         elif x is not None and f_values is not None and a is not None and b is not None :
              h = np.diff(x)[0] if len(x) > 1 else 1
              integral = (h/3)*(f_values[0] + 4 * np.sum(f_values[1:-1:2]) + 2*np.sum(f_values[2:-1:2]) + f_values[-1] if len(x) > 1 else f_values[0] )

              return integral
         else:
            raise ValueError("Parameters do not match")

    elif rule == "3/8":
        if not isinstance(n, int) or n <= 0 or n % 3 != 0:
               raise ValueError("n must be a positive integer multiple of 3 for Simpson's 3/8 rule")

        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        f_values = np.array([f(val) for val in x])
        integral = (3 * h / 8) * (f_values[0] + 3 * np.sum(f_values[1:-1:3]) + 3 * np.sum(f_values[2:-1:3]) + 2 * np.sum(f_values[3:-1:3]) + f_values[-1])
        return integral

    else:
         raise ValueError("The rule parameter must be '1/3' or '3/8'")