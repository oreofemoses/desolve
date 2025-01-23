# import numpy as np
# import pytest
# import sympy
# from core.differentiation.finite_difference import forward_diff
#
# x_sym = sympy.Symbol('x')
#
# # Define Symbolic functions and their derivatives
# f1_sym = x_sym ** 2
# df1_sym = sympy.diff(f1_sym, x_sym)
# d2f1_sym = sympy.diff(df1_sym, x_sym)
# f1 = sympy.lambdify(x_sym, f1_sym, "numpy")
# df1 = sympy.lambdify(x_sym, df1_sym, "numpy")
# d2f1 = sympy.lambdify(x_sym, d2f1_sym, "numpy")
#
# f2_sym = x_sym ** 3
# df2_sym = sympy.diff(f2_sym, x_sym)
# d2f2_sym = sympy.diff(df2_sym, x_sym)
# d3f2_sym = sympy.diff(d2f2_sym, x_sym)
# f2 = sympy.lambdify(x_sym, f2_sym, "numpy")
# df2 = sympy.lambdify(x_sym, df2_sym, "numpy")
# d2f2 = sympy.lambdify(x_sym, d2f2_sym, "numpy")
# d3f2 = sympy.lambdify(x_sym, d3f2_sym, "numpy")
#
# f3_sym = sympy.sin(x_sym)
# df3_sym = sympy.diff(f3_sym, x_sym)
# d2f3_sym = sympy.diff(df3_sym, x_sym)
# d3f3_sym = sympy.diff(d2f3_sym, x_sym)
# f3 = sympy.lambdify(x_sym, f3_sym, "numpy")
# df3 = sympy.lambdify(x_sym, df3_sym, "numpy")
# d2f3 = sympy.lambdify(x_sym, d2f3_sym, "numpy")
# d3f3 = sympy.lambdify(x_sym, d3f3_sym, "numpy")
#
# f4_sym = sympy.cos(x_sym)
# df4_sym = sympy.diff(f4_sym, x_sym)
# d2f4_sym = sympy.diff(df4_sym, x_sym)
# d3f4_sym = sympy.diff(d2f4_sym, x_sym)
# f4 = sympy.lambdify(x_sym, f4_sym, "numpy")
# df4 = sympy.lambdify(x_sym, df4_sym, "numpy")
# d2f4 = sympy.lambdify(x_sym, d2f4_sym, "numpy")
# d3f4 = sympy.lambdify(x_sym, d3f4_sym, "numpy")
#
# f5_sym = sympy.exp(x_sym)
# df5_sym = sympy.diff(f5_sym, x_sym)
# d2f5_sym = sympy.diff(df5_sym, x_sym)
# d3f5_sym = sympy.diff(d2f5_sym, x_sym)
# f5 = sympy.lambdify(x_sym, f5_sym, "numpy")
# df5 = sympy.lambdify(x_sym, df5_sym, "numpy")
# d2f5 = sympy.lambdify(x_sym, d2f5_sym, "numpy")
# d3f5 = sympy.lambdify(x_sym, d3f5_sym, "numpy")
#
# f6_sym = sympy.log(x_sym)
# df6_sym = sympy.diff(f6_sym, x_sym)
# d2f6_sym = sympy.diff(df6_sym, x_sym)
# d3f6_sym = sympy.diff(d2f6_sym, x_sym)
# f6 = sympy.lambdify(x_sym, f6_sym, "numpy")
# df6 = sympy.lambdify(x_sym, df6_sym, "numpy")
# d2f6 = sympy.lambdify(x_sym, d2f6_sym, "numpy")
# d3f6 = sympy.lambdify(x_sym, d3f6_sym, "numpy")
#
# f7_sym = 1 / x_sym
# df7_sym = sympy.diff(f7_sym, x_sym)
# d2f7_sym = sympy.diff(df7_sym, x_sym)
# d3f7_sym = sympy.diff(d2f7_sym, x_sym)
# f7 = sympy.lambdify(x_sym, f7_sym, "numpy")
# df7 = sympy.lambdify(x_sym, df7_sym, "numpy")
# d2f7 = sympy.lambdify(x_sym, d2f7_sym, "numpy")
# d3f7 = sympy.lambdify(x_sym, d3f7_sym, "numpy")
#
# f8_sym = sympy.sqrt(x_sym)
# df8_sym = sympy.diff(f8_sym, x_sym)
# d2f8_sym = sympy.diff(df8_sym, x_sym)
# d3f8_sym = sympy.diff(d2f8_sym, x_sym)
# f8 = sympy.lambdify(x_sym, f8_sym, "numpy")
# df8 = sympy.lambdify(x_sym, df8_sym, "numpy")
# d2f8 = sympy.lambdify(x_sym, d2f8_sym, "numpy")
# d3f8 = sympy.lambdify(x_sym, d3f8_sym, "numpy")
#
# f9_sym = sympy.atan(x_sym)
# df9_sym = sympy.diff(f9_sym, x_sym)
# d2f9_sym = sympy.diff(df9_sym, x_sym)
# d3f9_sym = sympy.diff(d2f9_sym, x_sym)
# f9 = sympy.lambdify(x_sym, f9_sym, "numpy")
# df9 = sympy.lambdify(x_sym, df9_sym, "numpy")
# d2f9 = sympy.lambdify(x_sym, d2f9_sym, "numpy")
# d3f9 = sympy.lambdify(x_sym, d3f9_sym, "numpy")
#
# f10_sym = x_sym ** 4 + 2 * x_sym ** 2 - x_sym
# df10_sym = sympy.diff(f10_sym, x_sym)
# d2f10_sym = sympy.diff(df10_sym, x_sym)
# d3f10_sym = sympy.diff(d2f10_sym, x_sym)
# f10 = sympy.lambdify(x_sym, f10_sym, "numpy")
# df10 = sympy.lambdify(x_sym, df10_sym, "numpy")
# d2f10 = sympy.lambdify(x_sym, d2f10_sym, "numpy")
# d3f10 = sympy.lambdify(x_sym, d3f10_sym, "numpy")
#
#
# def test_forward_diff_sympy_x_squared():
#     h = 1
#     x0 = 1
#     terms = 5
#     expected_derivative = df1(x0)
#     calculated_derivative = forward_diff(f1, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f1(x0)
#     calculated_derivative = forward_diff(f1, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_sympy_x_cubed():
#     h = 1
#     x0 = 2
#     terms = 5
#     expected_derivative = df2(x0)
#     calculated_derivative = forward_diff(f2, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f2(x0)
#     calculated_derivative = forward_diff(f2, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d3f2(x0)
#     calculated_derivative = forward_diff(f2, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_sympy_sin_x():
#     h = 0.1
#     x0 = np.pi / 4
#     terms = 10
#     expected_derivative = df3(x0)
#     calculated_derivative = forward_diff(f3, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f3(x0)
#     calculated_derivative = forward_diff(f3, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d3f3(x0)
#     calculated_derivative = forward_diff(f3, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_sympy_cos_x():
#     h = 0.1
#     x0 = np.pi / 3
#     terms = 10
#     expected_derivative = df4(x0)
#     calculated_derivative = forward_diff(f4, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f4(x0)
#     calculated_derivative = forward_diff(f4, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d3f4(x0)
#     calculated_derivative = forward_diff(f4, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_sympy_exp_x():
#     h = 0.1
#     x0 = 1
#     terms = 10
#     expected_derivative = df5(x0)
#     calculated_derivative = forward_diff(f5, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f5(x0)
#     calculated_derivative = forward_diff(f5, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d3f5(x0)
#     calculated_derivative = forward_diff(f5, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_sympy_log_x():
#     h = 0.1
#     x0 = 1
#     terms = 10
#     expected_derivative = df6(x0)
#     calculated_derivative = forward_diff(f6, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f6(x0)
#     calculated_derivative = forward_diff(f6, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#     expected_derivative = d3f6(x0)
#     calculated_derivative = forward_diff(f6, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#
# def test_forward_diff_sympy_inverse_x():
#     h = 0.1
#     x0 = 2
#     terms = 10
#     expected_derivative = df7(x0)
#     calculated_derivative = forward_diff(f7, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f7(x0)
#     calculated_derivative = forward_diff(f7, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#     expected_derivative = d3f7(x0)
#     calculated_derivative = forward_diff(f7, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#
# def test_forward_diff_sympy_sqrt_x():
#     h = 0.1
#     x0 = 4
#     terms = 10
#     expected_derivative = df8(x0)
#     calculated_derivative = forward_diff(f8, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f8(x0)
#     calculated_derivative = forward_diff(f8, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#     expected_derivative = d3f8(x0)
#     calculated_derivative = forward_diff(f8, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#
# def test_forward_diff_sympy_atan_x():
#     h = 0.1
#     x0 = 1
#     terms = 10
#     expected_derivative = df9(x0)
#     calculated_derivative = forward_diff(f9, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f9(x0)
#     calculated_derivative = forward_diff(f9, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#     expected_derivative = d3f9(x0)
#     calculated_derivative = forward_diff(f9, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-3)
#
#
# def test_forward_diff_sympy_polynomial():
#     h = 0.1
#     x0 = 1
#     terms = 10
#     expected_derivative = df10(x0)
#     calculated_derivative = forward_diff(f10, x0, h, terms=terms)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d2f10(x0)
#     calculated_derivative = forward_diff(f10, x0, h, terms=terms, order=2)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#     expected_derivative = d3f10(x0)
#     calculated_derivative = forward_diff(f10, x0, h, terms=terms, order=3)
#     assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)
#
#
# def test_forward_diff_raises_value_error():
#     x0 = 1
#     h = 0.1
#     f = lambda x: x ** 2
#     with pytest.raises(ValueError):
#         forward_diff(f, x0, h, order=0)
#     with pytest.raises(ValueError):
#         forward_diff(f, x0, h, terms=0)

import numpy as np
import pytest
import sympy
from core.differentiation.divided_difference import forward_diff_divided

x_sym = sympy.Symbol('x')

# Define Symbolic functions and their derivatives
f1_sym = x_sym ** 2
df1_sym = sympy.diff(f1_sym, x_sym)
d2f1_sym = sympy.diff(df1_sym, x_sym)
f1 = sympy.lambdify(x_sym, f1_sym, "numpy")
df1 = sympy.lambdify(x_sym, df1_sym, "numpy")
d2f1 = sympy.lambdify(x_sym, d2f1_sym, "numpy")

f2_sym = x_sym ** 3
df2_sym = sympy.diff(f2_sym, x_sym)
d2f2_sym = sympy.diff(df2_sym, x_sym)
d3f2_sym = sympy.diff(d2f2_sym, x_sym)
f2 = sympy.lambdify(x_sym, f2_sym, "numpy")
df2 = sympy.lambdify(x_sym, df2_sym, "numpy")
d2f2 = sympy.lambdify(x_sym, d2f2_sym, "numpy")
d3f2 = sympy.lambdify(x_sym, d3f2_sym, "numpy")

f3_sym = sympy.sin(x_sym)
df3_sym = sympy.diff(f3_sym, x_sym)
d2f3_sym = sympy.diff(df3_sym, x_sym)
d3f3_sym = sympy.diff(d2f3_sym, x_sym)
f3 = sympy.lambdify(x_sym, f3_sym, "numpy")
df3 = sympy.lambdify(x_sym, df3_sym, "numpy")
d2f3 = sympy.lambdify(x_sym, d2f3_sym, "numpy")
d3f3 = sympy.lambdify(x_sym, d3f3_sym, "numpy")

f4_sym = sympy.cos(x_sym)
df4_sym = sympy.diff(f4_sym, x_sym)
d2f4_sym = sympy.diff(df4_sym, x_sym)
d3f4_sym = sympy.diff(d2f4_sym, x_sym)
f4 = sympy.lambdify(x_sym, f4_sym, "numpy")
df4 = sympy.lambdify(x_sym, df4_sym, "numpy")
d2f4 = sympy.lambdify(x_sym, d2f4_sym, "numpy")
d3f4 = sympy.lambdify(x_sym, d3f4_sym, "numpy")

f5_sym = sympy.exp(x_sym)
df5_sym = sympy.diff(f5_sym, x_sym)
d2f5_sym = sympy.diff(df5_sym, x_sym)
d3f5_sym = sympy.diff(d2f5_sym, x_sym)
f5 = sympy.lambdify(x_sym, f5_sym, "numpy")
df5 = sympy.lambdify(x_sym, df5_sym, "numpy")
d2f5 = sympy.lambdify(x_sym, d2f5_sym, "numpy")
d3f5 = sympy.lambdify(x_sym, d3f5_sym, "numpy")

f6_sym = sympy.log(x_sym)
df6_sym = sympy.diff(f6_sym, x_sym)
d2f6_sym = sympy.diff(df6_sym, x_sym)
d3f6_sym = sympy.diff(d2f6_sym, x_sym)
f6 = sympy.lambdify(x_sym, f6_sym, "numpy")
df6 = sympy.lambdify(x_sym, df6_sym, "numpy")
d2f6 = sympy.lambdify(x_sym, d2f6_sym, "numpy")
d3f6 = sympy.lambdify(x_sym, d3f6_sym, "numpy")

f7_sym = 1 / x_sym
df7_sym = sympy.diff(f7_sym, x_sym)
d2f7_sym = sympy.diff(df7_sym, x_sym)
d3f7_sym = sympy.diff(d2f7_sym, x_sym)
f7 = sympy.lambdify(x_sym, f7_sym, "numpy")
df7 = sympy.lambdify(x_sym, df7_sym, "numpy")
d2f7 = sympy.lambdify(x_sym, d2f7_sym, "numpy")
d3f7 = sympy.lambdify(x_sym, d3f7_sym, "numpy")

f8_sym = sympy.sqrt(x_sym)
df8_sym = sympy.diff(f8_sym, x_sym)
d2f8_sym = sympy.diff(df8_sym, x_sym)
d3f8_sym = sympy.diff(d2f8_sym, x_sym)
f8 = sympy.lambdify(x_sym, f8_sym, "numpy")
df8 = sympy.lambdify(x_sym, df8_sym, "numpy")
d2f8 = sympy.lambdify(x_sym, d2f8_sym, "numpy")
d3f8 = sympy.lambdify(x_sym, d3f8_sym, "numpy")

f9_sym = sympy.atan(x_sym)
df9_sym = sympy.diff(f9_sym, x_sym)
d2f9_sym = sympy.diff(df9_sym, x_sym)
d3f9_sym = sympy.diff(d2f9_sym, x_sym)
f9 = sympy.lambdify(x_sym, f9_sym, "numpy")
df9 = sympy.lambdify(x_sym, df9_sym, "numpy")
d2f9 = sympy.lambdify(x_sym, d2f9_sym, "numpy")
d3f9 = sympy.lambdify(x_sym, d3f9_sym, "numpy")

f10_sym = x_sym ** 4 + 2 * x_sym ** 2 - x_sym
df10_sym = sympy.diff(f10_sym, x_sym)
d2f10_sym = sympy.diff(df10_sym, x_sym)
d3f10_sym = sympy.diff(d2f10_sym, x_sym)
f10 = sympy.lambdify(x_sym, f10_sym, "numpy")
df10 = sympy.lambdify(x_sym, df10_sym, "numpy")
d2f10 = sympy.lambdify(x_sym, d2f10_sym, "numpy")
d3f10 = sympy.lambdify(x_sym, d3f10_sym, "numpy")


def test_forward_diff_divided_example_3_10():
    x = np.array([1.0, 1.5, 2.0, 3.0])
    f = np.array([0.0, 0.40547, 0.69315, 1.09861])
    at = 1.6
    expected_first_derivative = 0.63258  # From example
    expected_second_derivative = -0.43447  # From example

    first_derivative = forward_diff_divided(x=x, f_values=f, at=at, order=1)
    second_derivative = forward_diff_divided(x=x, f=f, at=at, order=2)
    assert np.isclose(first_derivative, expected_first_derivative, atol=1e-5)
    assert np.isclose(second_derivative, expected_second_derivative, atol=1e-5)


def test_forward_diff_divided_sympy_log_x():
    x0 = 1
    expected_derivative = df6(x0)
    calculated_derivative = forward_diff_divided(f=f6, at=x0, order=1, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)

    expected_derivative = d2f6(x0)
    calculated_derivative = forward_diff_divided(f=f6, at=x0, order=2, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-4)


def test_forward_diff_divided_sympy_inverse_x():
    x0 = 2
    expected_derivative = df7(x0)
    calculated_derivative = forward_diff_divided(f=f7, at=x0, order=1, x=np.array([1.5, 2, 2.5, 3]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)

    expected_derivative = d2f7(x0)
    calculated_derivative = forward_diff_divided(f=f7, at=x0, order=2, x=np.array([1.5, 2, 2.5, 3]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-4)


def test_forward_diff_divided_sympy_sqrt_x():
    x0 = 4
    expected_derivative = df8(x0)
    calculated_derivative = forward_diff_divided(f=f8, at=x0, order=1, x=np.array([3.5, 4, 4.5, 5]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)

    expected_derivative = d2f8(x0)
    calculated_derivative = forward_diff_divided(f=f8, at=x0, order=2, x=np.array([3.5, 4, 4.5, 5]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-4)


def test_forward_diff_divided_sympy_atan_x():
    x0 = 1
    expected_derivative = df9(x0)
    calculated_derivative = forward_diff_divided(f=f9, at=x0, order=1, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)

    expected_derivative = d2f9(x0)
    calculated_derivative = forward_diff_divided(f=f9, at=x0, order=2, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-4)


def test_forward_diff_divided_sympy_polynomial():
    x0 = 1
    expected_derivative = df10(x0)
    calculated_derivative = forward_diff_divided(f=f10, at=x0, order=1, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-5)

    expected_derivative = d2f10(x0)
    calculated_derivative = forward_diff_divided(f=f10, at=x0, order=2, x=np.array([0.5, 1, 1.5, 2]))
    assert np.isclose(calculated_derivative, expected_derivative, atol=1e-4)


def test_forward_diff_divided_raises_value_error():
    x0 = 1
    f = lambda x: x ** 2
    x = np.array([1, 2, 3])
    f_values = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        forward_diff_divided(x=x, f_values=f_values, at=1.5, f=f)
    with pytest.raises(ValueError):
        forward_diff_divided(x=x, f=f_values, at=4)
    with pytest.raises(ValueError):
        forward_diff_divided(at=1.5)