import numpy as np
from typing import Callable, List, Literal
from core.integration.trapezoidal import composite_trapezoidal
from core.integration.simpsons import composite_simpsons


def romberg_integration(f: Callable[[float], float], a: float, b: float, n_steps: int,
                        rule: Literal["trapezoidal", "simpsons"] = "trapezoidal") -> float:
    """
    Approximates the definite integral of a function using Romberg integration with either the trapezoidal or Simpson's 1/3 rule.

    Parameters
    ----------
    f : callable
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n_steps : int
        The number of steps to do in the romberg table, corresponding to values of h, h/2, h/4...
    rule : Literal["trapezoidal", "simpsons"]
         Whether to use the trapezoidal or the simpson's method as the base method.

    Returns
    -------
    float
        The approximated integral value after romberg extrapolation

    Raises
    ------
    ValueError
        If n_steps is not a positive integer.
    """
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if rule not in ["trapezoidal", "simpsons"]:
        raise ValueError("The rule parameter must be 'trapezoidal' or 'simpsons'")

    romberg_table = []

    if rule == "trapezoidal":
        n_values = [2 ** i for i in range(n_steps)]
        h_values = [(b - a) / n for n in n_values]
        base_values = [composite_trapezoidal(f, a, b, n) for n in n_values]
        romberg_table = [base_values]
        for m in range(1, n_steps):
            new_values = []
            for i in range(0, n_steps - m):
                h = h_values[i]
                previous_h = romberg_table[m - 1][i]
                previous_h_2 = romberg_table[m - 1][i + 1]
                new_value = (4 ** m * previous_h_2 - previous_h) / (4 ** m - 1)
                new_values.append(new_value)
            romberg_table.append(new_values)

    elif rule == "simpsons":
        n_values = [2 * (2 ** i) for i in range(n_steps)]
        h_values = [(b - a) / n for n in n_values]
        base_values = [composite_simpsons(f, a, b, n, rule="1/3") for n in n_values]
        romberg_table = [base_values]
        print(romberg_table)
        for m in range(1, n_steps):
            new_values = []
            for i in range(0, n_steps - m):
                print(romberg_table)
                h = h_values[i]
                previous_h = romberg_table[m - 1][i]
                previous_h_2 = romberg_table[m - 1][i + 1]
                new_value = (4 ** (m + 1) * previous_h_2 - previous_h) / (4 ** (m + 1) - 1)
                new_values.append(new_value)
            romberg_table.append(new_values)
    return romberg_table[-1][0]