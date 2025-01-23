# import numpy as np
# import pytest
# import sympy
# from core.integration.trapezoidal import composite_trapezoidal
# from core.integration.simpsons import composite_simpsons
# from core.integration.romberg import romberg_integration
#
# x_sym = sympy.Symbol('x')
#
# # Define Symbolic functions and their derivatives
# f1_sym = x_sym ** 2
# f1 = sympy.lambdify(x_sym, f1_sym, "numpy")
# integral1_sym = sympy.integrate(f1_sym, (x_sym, 0, 1))
# integral1 = sympy.lambdify([], integral1_sym, "numpy")
#
# f2_sym = x_sym ** 3
# f2 = sympy.lambdify(x_sym, f2_sym, "numpy")
# integral2_sym = sympy.integrate(f2_sym, (x_sym, 0, 1))
# integral2 = sympy.lambdify([], integral2_sym, "numpy")
#
# f3_sym = sympy.sin(x_sym)
# f3 = sympy.lambdify(x_sym, f3_sym, "numpy")
# integral3_sym = sympy.integrate(f3_sym, (x_sym, 0, np.pi))
# integral3 = sympy.lambdify([], integral3_sym, "numpy")
#
# f4_sym = sympy.cos(x_sym)
# f4 = sympy.lambdify(x_sym, f4_sym, "numpy")
# integral4_sym = sympy.integrate(f4_sym, (x_sym, 0, np.pi / 2))
# integral4 = sympy.lambdify([], integral4_sym, "numpy")
#
# f5_sym = sympy.exp(x_sym)
# f5 = sympy.lambdify(x_sym, f5_sym, "numpy")
# integral5_sym = sympy.integrate(f5_sym, (x_sym, 0, 1))
# integral5 = sympy.lambdify([], integral5_sym, "numpy")
#
# TRAPEZOIDAL_ATOL = 1e-5
# SIMPSONS_ATOL = 1e-5
# ROMBERG_ATOL = 1e-5
#
#
# def test_composite_trapezoidal_x_squared():
#     a = 0
#     b = 1
#     n = 4
#     expected_result = integral1()
#     calculated_result = composite_trapezoidal(f1, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=TRAPEZOIDAL_ATOL)
#
#
# def test_composite_trapezoidal_x_cubed():
#     a = 0
#     b = 1
#     n = 4
#     expected_result = integral2()
#     calculated_result = composite_trapezoidal(f2, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=TRAPEZOIDAL_ATOL)
#
#
# def test_composite_trapezoidal_sin_x():
#     a = 0
#     b = np.pi
#     n = 32
#     expected_result = integral3()
#     calculated_result = composite_trapezoidal(f3, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=TRAPEZOIDAL_ATOL)
#
#
# def test_composite_trapezoidal_cos_x():
#     a = 0
#     b = np.pi / 2
#     n = 32
#     expected_result = integral4()
#     calculated_result = composite_trapezoidal(f4, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=TRAPEZOIDAL_ATOL)
#
#
# def test_composite_trapezoidal_exp_x():
#     a = 0
#     b = 1
#     n = 32
#     expected_result = integral5()
#     calculated_result = composite_trapezoidal(f5, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=TRAPEZOIDAL_ATOL)
#
#
# def test_composite_trapezoidal_raises_value_error():
#     a = 0
#     b = 1
#     f = lambda x: x ** 2
#     with pytest.raises(ValueError):
#         composite_trapezoidal(f, a, b, n=-1)
#
#
# def test_composite_simpsons_x_squared():
#     a = 0
#     b = 1
#     n = 4
#     expected_result = integral1()
#     calculated_result = composite_simpsons(f1, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_x_cubed():
#     a = 0
#     b = 1
#     n = 4
#     expected_result = integral2()
#     calculated_result = composite_simpsons(f2, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_sin_x():
#     a = 0
#     b = np.pi
#     n = 32
#     expected_result = integral3()
#     calculated_result = composite_simpsons(f3, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_cos_x():
#     a = 0
#     b = np.pi / 2
#     n = 32
#     expected_result = integral4()
#     calculated_result = composite_simpsons(f4, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_exp_x():
#     a = 0
#     b = 1
#     n = 32
#     expected_result = integral5()
#     calculated_result = composite_simpsons(f5, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_raises_value_error():
#     a = 0
#     b = 1
#     f = lambda x: x ** 2
#     with pytest.raises(ValueError):
#         composite_simpsons(f, a, b, n=-1)
#     with pytest.raises(ValueError):
#         composite_simpsons(f, a, b, n=1)
#     with pytest.raises(ValueError):
#         composite_simpsons(f, a, b, n=2, rule="3/8")
#     with pytest.raises(ValueError):
#         composite_simpsons(f, a, b, n=1, rule="invalid")
#
#
# def test_composite_simpsons_3_8_x_cubed():
#     a = 0
#     b = 1
#     n = 9
#     expected_result = integral2()
#     calculated_result = composite_simpsons(f2, a, b, n, rule="3/8")
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_3_8_sin_x():
#     a = 0
#     b = np.pi
#     n = 99
#     expected_result = integral3()
#     calculated_result = composite_simpsons(f3, a, b, n, rule="3/8")
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_3_8_cos_x():
#     a = 0
#     b = np.pi / 2
#     n = 99
#     expected_result = integral4()
#     calculated_result = composite_simpsons(f4, a, b, n, rule="3/8")
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_composite_simpsons_3_8_exp_x():
#     a = 0
#     b = 1
#     n = 99
#     expected_result = integral5()
#     calculated_result = composite_simpsons(f5, a, b, n, rule="3/8")
#     assert np.isclose(calculated_result, expected_result, atol=SIMPSONS_ATOL)
#
#
# def test_romberg_integration_trapezoidal_x_squared():
#     a = 0
#     b = 1
#     n_steps = 3
#     expected_result = integral1()
#     calculated_result = romberg_integration(f1, a, b, n_steps, rule="trapezoidal")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_trapezoidal_x_cubed():
#     a = 0
#     b = 1
#     n_steps = 3
#     expected_result = integral2()
#     calculated_result = romberg_integration(f2, a, b, n_steps, rule="trapezoidal")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_trapezoidal_sin_x():
#     a = 0
#     b = np.pi
#     n_steps = 5
#     expected_result = integral3()
#     calculated_result = romberg_integration(f3, a, b, n_steps, rule="trapezoidal")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_trapezoidal_cos_x():
#     a = 0
#     b = np.pi / 2
#     n_steps = 5
#     expected_result = integral4()
#     calculated_result = romberg_integration(f4, a, b, n_steps, rule="trapezoidal")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_trapezoidal_exp_x():
#     a = 0
#     b = 1
#     n_steps = 5
#     expected_result = integral5()
#     calculated_result = romberg_integration(f5, a, b, n_steps, rule="trapezoidal")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_simpsons_x_squared():
#     a = 0
#     b = 1
#     n_steps = 3
#     expected_result = integral1()
#     calculated_result = romberg_integration(f1, a, b, n_steps, rule="simpsons")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_simpsons_x_cubed():
#     a = 0
#     b = 1
#     n_steps = 3
#     expected_result = integral2()
#     calculated_result = romberg_integration(f2, a, b, n_steps, rule="simpsons")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_simpsons_sin_x():
#     a = 0
#     b = np.pi
#     n_steps = 5
#     expected_result = integral3()
#     calculated_result = romberg_integration(f3, a, b, n_steps, rule="simpsons")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_simpsons_cos_x():
#     a = 0
#     b = np.pi / 2
#     n_steps = 5
#     expected_result = integral4()
#     calculated_result = romberg_integration(f4, a, b, n_steps, rule="simpsons")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_simpsons_exp_x():
#     a = 0
#     b = 1
#     n_steps = 5
#     expected_result = integral5()
#     calculated_result = romberg_integration(f5, a, b, n_steps, rule="simpsons")
#     assert np.isclose(calculated_result, expected_result, atol=ROMBERG_ATOL)
#
#
# def test_romberg_integration_raises_value_error():
#     a = 0
#     b = 1
#     f = lambda x: x ** 2
#     with pytest.raises(ValueError):
#         romberg_integration(f, a, b, n_steps=-1)
#
#
# def test_trapezoidal_exercise_3_2_1():
#     a = 0
#     b = 1
#     n = 4
#     f = lambda x: 1 / x
#     expected_result = 1.73284
#     calculated_result = composite_trapezoidal(f, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=1e-5)
#
#
# def test_trapezoidal_exercise_3_2_3():
#     a = 0
#     b = np.pi
#     n = 6
#     f = lambda x: np.sin(x)
#     expected_result = 1.954097
#     calculated_result = composite_trapezoidal(f, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=1e-5)
#
#
# def test_simpsons_exercise_3_2_7():
#     a = 0
#     b = 1
#     n = 4
#     f = lambda x: x * np.exp(x)
#     expected_result = 0.99977
#     calculated_result = composite_simpsons(f, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=1e-5)
#
#
# def test_simpsons_exercise_3_2_8():
#     a = 0
#     b = 2
#     n = 4
#     f = lambda x: np.exp(x)
#     expected_result = 6.389242
#     calculated_result = composite_simpsons(f, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=1e-5)
#
#
# def test_simpsons_exercise_3_2_9():
#     a = 0
#     b = 1
#     n = 6
#     f = lambda x: 1 / (1 + x ** 2)
#     expected_result = 0.785396
#     calculated_result = composite_simpsons(f, a, b, n)
#     assert np.isclose(calculated_result, expected_result, atol=1e-5)