# import streamlit as st
# import sympy
# import numpy as np
# import pandas as pd
# from core.differentiation.finite_difference import forward_diff
# from core.differentiation.divided_difference import forward_diff_divided
# from core.integration.trapezoidal import composite_trapezoidal
# from core.integration.simpsons import composite_simpsons
# from core.integration.romberg import romberg_integration
#
# def plot_diff(x, f_values, x0, terms, order, function_str, plot_points):
#     if len(x) == 0:
#         st.write("No data to show")
#         return
#
#     if callable(f_values):
#         x_range = plot_points/(x[0] -x[1]) if abs(x0) > 1e-3 else 1
#         x_vals = np.linspace(x0 - x_range, x0 + x_range, plot_points)
#         f_values = np.array(
#             [forward_diff(f=f_values, x0=val, h=np.diff(x)[0] if len(x) > 1 else 0.1, terms=terms, order=order) for val
#              in x_vals])
#
#         chart_data = pd.DataFrame({"x": x_vals, "Δx": f_values})
#         st.line_chart(chart_data, x="x", y="Δx", height=300, use_container_width=True)
#         st.write(f"Derivative point (x0 = {x0})")
#
#     else:
#         x_range = max(x) - min(x) if max(x) - min(x) > 1e-3 else 1
#         x_vals = np.linspace(min(x) - x_range / 2, max(x) + x_range / 2, 100)
#         f_values = np.array([forward_diff_divided(x=x, f=f_values, at=val, order=order) for val in x if val <= max(x)])
#         chart_data = pd.DataFrame({"x": x[0:len(f_values)], "Δx": f_values})
#         st.line_chart(chart_data, x="x", y="Δx", height=300, use_container_width=True)
#         st.write(f"Derivative point (x0 = {x0})")
#
#
# def plot_int(f, a, b, function_str):
#     x_range = b - a if abs(b - a) > 1e-3 else 1
#     x_vals = np.linspace(a - x_range / 2, b + x_range / 2, 500)
#     f_values = np.array([f(x) for x in x_vals])
#     chart_data = pd.DataFrame({"x": x_vals, "f(x)": f_values})
#     st.area_chart(chart_data, x="x", y="f(x)", height=300, use_container_width=True)
#
#
# def main():
#     st.set_page_config(
#         page_title="DEsolve: Numerical Methods",
#         layout="wide"
#     )
#     st.title("DEsolve: Numerical Methods Tool")
#
#     st.sidebar.header("Navigation")
#     method_choice = st.sidebar.radio(label="Choose Method:", options=["Differentiation", "Integration", "IVP", "BVP"])
#
#     if method_choice == "Differentiation":
#         with st.container():
#             st.header("Numerical Differentiation")
#             method_type = st.selectbox("Choose your Method", options=["finite difference", "divided difference"],
#                                        key="diff_method")
#
#             function_str = st.text_input("Function (e.g., x**2 + sin(x))", value="sin(x)", key="diff_func")
#             x0 = st.number_input("Point of Evaluation (x0)", value=1.0, key="diff_x0")
#             h = st.number_input("Step Size (h)", value=0.1, key="diff_h", min_value=0.0001,
#                                 disabled=method_type != "finite difference")
#             order = st.selectbox("Derivative Order", options=[1, 2, 3], key="diff_order")
#             terms = st.slider("Number of Terms", 1, 10, 5, key="diff_terms",
#                               disabled=method_type != "finite difference")
#             plot_points = st.slider("Plot Points", 100, 1000, 100, key="diff_plot_points")
#             csv_file = st.file_uploader("Upload CSV Data", type=["csv"], key="diff_csv_file")
#             text_input_data = st.text_area("Or input data here (x, f(x) per line)", key="diff_data_text")
#             calculate_diff = st.button("Calculate", key="calculate_diff")
#
#             if calculate_diff:
#                 try:
#                     x_sym = sympy.Symbol('x')
#                     f_sym = sympy.sympify(function_str)
#                     f = sympy.lambdify(x_sym, f_sym, "numpy")
#                     if csv_file:
#                         df = pd.read_csv(csv_file, header=None, names=["x", "f"])
#                         x = df["x"].to_numpy()
#                         f_values = df["f"].to_numpy()
#                         if method_type == "finite difference":
#                             result = forward_diff(x=x, f_values=f_values, order=order, at=x0, terms=terms)
#                             plot_diff(x, f_values, x0, terms, order, function_str, plot_points=0)
#                             st.write(f"Result: {result}")
#                         elif method_type == "divided difference":
#                             result = forward_diff_divided(x=x, f=f_values, order=order, at=x0)
#                             plot_diff(x, f_values, x0, terms, order, function_str, plot_points=0)
#                             st.write(f"Result: {result}")
#
#
#                     elif text_input_data:
#                         lines = text_input_data.splitlines()
#                         data = [list(map(float, line.split(','))) for line in lines if line]
#                         x = np.array([row[0] for row in data])
#                         f_values = np.array([row[1] for row in data])
#                         if method_type == "finite difference":
#                             result = forward_diff(x=x, f_values=f_values, order=order, at=x0, terms=terms)
#                             plot_diff(x, f_values, x0, terms, order, function_str, plot_points=0)
#                             st.write(f"Result: {result}")
#                         elif method_type == "divided difference":
#                             result = forward_diff_divided(x=x, f=f_values, order=order, at=x0)
#                             plot_diff(x, f_values, x0, terms, order, function_str, plot_points=0)
#                             st.write(f"Result: {result}")
#
#                     else:
#                         if method_type == "finite difference":
#                             result = forward_diff(f, x0, h, order, terms)
#                             st.write(f"Result: {result}")
#                             with st.expander("See Plot"):
#                                 x_vals = np.array([x0 + i * h for i in range(0, terms + 1)])
#                                 plot_diff(x=x_vals, f_values=lambda val: f(val), x0=x0, terms=terms, order=order,
#                                           function_str=function_str, plot_points=plot_points)
#                         elif method_type == "divided difference":
#                             st.error("You must specify data when using divided difference.")
#
#
#                 except (ValueError, TypeError, sympy.SympifyError, pd.errors.ParserError) as e:
#                     st.error(str(e))
#
#             if st.button("Clear", key="clear_diff"):
#                 st.session_state["diff_func"] = "sin(x)"
#                 st.session_state["diff_x0"] = 1.0
#                 st.session_state["diff_h"] = 0.1
#                 st.session_state["diff_order"] = 1
#                 st.session_state["diff_terms"] = 5
#     elif method_choice == "Integration":
#         with st.container():
#             st.header("Numerical Integration")
#             function_str_int = st.text_input("Function (e.g., x**2 + sin(x))", value="x**2 + 2*x + 1", key="int_func")
#             a = st.number_input("Lower Limit (a)", value=0.0, key="int_a")
#             b = st.number_input("Upper Limit (b)", value=1.0, key="int_b")
#             n = st.number_input("Number of Subintervals (N)", value=10, key="int_n", step=1, min_value=1)
#             method = st.selectbox("Integration Method", options=["trapezoidal", "simpsons", "romberg"],
#                                   key="int_method")
#             romberg_steps = st.slider("Romberg Steps", 1, 10, 5, key="int_romberg_steps", disabled=method != "romberg")
#             csv_file_int = st.file_uploader("Upload CSV Data", type=["csv"], key="int_csv_file")
#             text_input_data_int = st.text_area("Or input data here (x, f(x) per line)", key="int_data_text")
#
#             calculate_int = st.button("Calculate", key="calculate_int")
#
#             if calculate_int:
#                 try:
#                     x_sym = sympy.Symbol('x')
#                     f_sym = sympy.sympify(function_str_int)
#                     f = sympy.lambdify(x_sym, f_sym, "numpy")
#
#                     if csv_file_int:
#                         df = pd.read_csv(csv_file_int, header=None, names=["x", "f"])
#                         x = df["x"].to_numpy()
#                         f_values = df["f"].to_numpy()
#                         result = composite_trapezoidal(x=x, f_values=f_values, a=a, b=b, n=int((b - a) / np.diff(x)[0]))
#                         plot_int(lambda val, x=x, f_values=f_values: np.interp(val, x, f_values), a, b,
#                                  function_str_int)
#                         st.write(f"Result: {result}")
#
#                     elif text_input_data_int:
#                         lines = text_input_data_int.splitlines()
#                         data = [list(map(float, line.split(','))) for line in lines if line]
#                         x = np.array([row[0] for row in data])
#                         f_values = np.array([row[1] for row in data])
#                         result = composite_trapezoidal(x=x, f_values=f_values, a=a, b=b, n=int((b - a) / np.diff(x)[0]))
#                         plot_int(lambda val, x=x, f_values=f_values: np.interp(val, x, f_values), a, b,
#                                  function_str_int)
#                         st.write(f"Result: {result}")
#
#
#                     elif method == "trapezoidal":
#                         result = composite_trapezoidal(f, a, b, n)
#                         with st.expander("See Plot"):
#                             plot_int(f, a, b, function_str_int)
#                     elif method == "simpsons":
#                         if romberg_steps == 0:
#                             result = composite_simpsons(f, a, b, n, rule="1/3")
#                         else:
#                             result = romberg_integration(f, a, b, romberg_steps, rule="simpsons")
#                         with st.expander("See Plot"):
#                             plot_int(f, a, b, function_str_int)
#                     elif method == "romberg":
#                         result = romberg_integration(f, a, b, n, rule="trapezoidal")
#                         with st.expander("See Plot"):
#                             plot_int(f, a, b, function_str_int)
#                 except (ValueError, TypeError, sympy.SympifyError, pd.errors.ParserError) as e:
#                     st.error(str(e))
#
#             if st.button("Clear", key="clear_int"):
#                 st.session_state["int_func"] = "x**2 + 2*x + 1"
#                 st.session_state["int_a"] = 0.0
#                 st.session_state["int_b"] = 1.0
#                 st.session_state["int_n"] = 10
#                 st.session_state["int_method"] = "trapezoidal"
#                 st.session_state["int_romberg_steps"] = 5
#
#
# if __name__ == "__main__":
#     main()

import streamlit as st
import sympy
import numpy as np
import pandas as pd
from core.differentiation.finite_difference import forward_diff
from core.differentiation.divided_difference import forward_diff_divided
from core.integration.trapezoidal import composite_trapezoidal
from core.integration.simpsons import composite_simpsons
from core.integration.romberg import romberg_integration


def plot_diff(x, f_values, x0, terms, order, function_str, plot_points, h=0.1):
    """
    Plots the function and its derivatives using Streamlit's built-in charting tools.

    Parameters:
    - x: array-like, x values of the data or function
    - f_values: array-like, corresponding f(x) values
    - x0: float, point of evaluation
    - terms: int, number of terms for divided difference
    - order: int, highest order of derivative to compute
    - function_str: str, function as a string (for display)
    - plot_points: int, number of points for plotting
    - h: float, step size (for function input)
    """
    # Generate evenly spaced x values for smoother plotting
    x_min, x_max = min(x), max(x)
    x_plot = np.linspace(x_min, x_max, plot_points)
    f_interp = np.interp(x_plot, x, f_values)

    # Initialize chart data
    chart_data = pd.DataFrame({"x": x_plot, "f(x)": f_interp})

    # Add derivatives to the chart data
    for d in range(1, order + 1):
        derivative = [forward_diff_divided(x, f_values, at=xi, order=d) for xi in x_plot]
        chart_data[f"f^{d}(x)"] = derivative
    print(chart_data)
    # Display the chart in Streamlit
    st.line_chart(chart_data.set_index("x"), use_container_width=True)


def plot_int(f, a, b, function_str, method="trapezoidal"):
    # Define the range of x values for plotting the integral
    x_vals = np.linspace(a, b, 500)  # 500 is just a good number for visualization

    # Generate table data for integration points and their corresponding integral values
    integral_values = []

    # Loop through the x values and calculate the integration points using the chosen method
    for i in range(1, len(x_vals)):
        a_i, b_i = x_vals[0], x_vals[i]

        # Depending on the method, calculate the integral between a_i and b_i
        if method == "trapezoidal":
            integral = composite_trapezoidal(f, a_i, b_i, 10)  # You can adjust the number of subintervals
        elif method == "simpsons":
            integral = composite_simpsons(f, a_i, b_i, 10)  # Same for Simpson's rule
        elif method == "romberg":
            integral = romberg_integration(f, a_i, b_i, 5)  # Adjust steps for Romberg's method
        else:
            integral = 0  # Default if no method is matched

        integral_values.append(integral)

    # Create a DataFrame to store x and corresponding integral values for plotting
    chart_data = pd.DataFrame({"x": x_vals[1:], "Integral": integral_values})
    print(chart_data)

    # Plot the integral curve
    st.area_chart(chart_data, x="x", y="Integral", height=300, use_container_width=True)


def main():
    st.set_page_config(
        page_title="DEsolve: Numerical Methods",
        layout="wide"
    )
    st.title("DEsolve: Numerical Methods Tool")

    st.sidebar.header("Navigation")
    method_choice = st.sidebar.radio(label="Choose Method:", options=["Differentiation", "Integration", "IVP", "BVP"])

    if method_choice == "Differentiation":
        with st.container():
            st.header("Numerical Differentiation")
            method_type = st.selectbox("Choose your Method", options=["finite difference", "divided difference"],
                                       key="diff_method")

            function_str = st.text_input("Function (e.g., x**2 + sin(x))", value="sin(x)", key="diff_func")
            x0 = st.number_input("Point of Evaluation (x0)", value=1.0, key="diff_x0")
            h = st.number_input("Step Size (h)", value=0.1, key="diff_h", min_value=0.0001,
                                disabled= function_str=="")
            order = st.selectbox("Derivative Order", options=list(range(1,51)),key="diff_order")
            terms = st.slider("Number of Terms", 1, 10, 5, key="diff_terms",
                              disabled= function_str=="")
            plot_points = st.slider("Plot Points", 100, 1000, 100, key="diff_plot_points")
            csv_file = st.file_uploader("Upload CSV Data", type=["csv"], key="diff_csv_file")
            text_input_data = st.text_area("Or input data here (x, f(x) per line)", key="diff_data_text")
            calculate_diff = st.button("Calculate", key="calculate_diff")

            if calculate_diff:
                try:
                    x_sym = sympy.Symbol('x')
                    f_sym = sympy.sympify(function_str)
                    f = sympy.lambdify(x_sym, f_sym, "numpy")
                    if csv_file:
                        df = pd.read_csv(csv_file, header=None, names=["x", "f"])
                        x = df["x"].to_numpy()
                        f_values = df["f"].to_numpy()
                        if method_type == "finite difference":
                            result = forward_diff(x=x, f_values=f_values, order=order, at=x0, terms=terms)
                            plot_diff(x, f_values, x0, terms, order, function_str, plot_points,h)
                            st.write(f"Result: {result}")
                        elif method_type == "divided difference":
                            result = forward_diff_divided(x=x, f_values=f_values, order=order, at=x0)
                            plot_diff(x, f_values, x0, terms, order, function_str, plot_points,h)
                            st.write(f"Result: {result}!")


                    elif text_input_data:
                        lines = text_input_data.splitlines()
                        data = [list(map(float, line.split(','))) for line in lines if line]
                        x = np.array([row[0] for row in data])
                        f_values = np.array([row[1] for row in data])
                        if method_type == "finite difference":
                            result = forward_diff(x=x, f_values=f_values, order=order, at=x0, terms=terms)
                            plot_diff(x, f_values, x0, terms, order, function_str, plot_points,h)
                            st.write(f"Result: {result}")
                        elif method_type == "divided difference":
                            result = forward_diff_divided(x=x, f_values=f_values, order=order, at=x0)
                            plot_diff(x, f_values, x0, terms, order, function_str, plot_points,h)
                            st.write(f"Result: {result}!")

                    else:
                        if method_type == "finite difference":
                            x_vals = np.linspace(x0 - h, x0 + h, terms + 1)
                            f_values = np.array([f(val) for val in x_vals])
                            result = forward_diff(f, x0, h, order, terms)
                            st.write(f"Result: {result}")
                            with st.expander("See Plot"):
                                plot_diff(x=x_vals, f_values=f_values, x0=x0, terms=terms, order=order,
                                          function_str=function_str, plot_points=plot_points, h=h)
                        elif method_type == "divided difference":
                            x_vals = np.linspace(x0 - h, x0 + h, terms + 1)
                            f_values = np.array([f(val) for val in x_vals])
                            result = forward_diff_divided(x=x_vals, f_values=f_values, at=x0, order=order)
                            st.write(f"Result: {result}")
                            with st.expander("See Plot"):
                                plot_diff(x=x_vals, f_values=f_values, x0=x0, terms=terms, order=order,
                                          function_str=function_str, plot_points=plot_points, h=h)

                except (ValueError, TypeError, sympy.SympifyError, pd.errors.ParserError) as e:
                    st.error(str(e))

            if st.button("Clear", key="clear_diff"):
                st.session_state["diff_func"] = "sin(x)"
                st.session_state["diff_x0"] = 1.0
                st.session_state["diff_h"] = 0.1
                st.session_state["diff_order"] = 1
                st.session_state["diff_terms"] = 5
    elif method_choice == "Integration":
        with st.container():
            st.header("Numerical Integration")
            function_str_int = st.text_input("Function (e.g., x**2 + sin(x))", value="x**2 + 2*x + 1",
                                             key="int_func")
            a = st.number_input("Lower Limit (a)", value=0.0, key="int_a")
            b = st.number_input("Upper Limit (b)", value=1.0, key="int_b")
            n = st.number_input("Number of Subintervals (N)", value=10, key="int_n", step=1, min_value=1)
            method = st.selectbox("Integration Method", options=["trapezoidal", "simpsons", "romberg"],
                                  key="int_method")
            romberg_steps = st.slider("Romberg Steps", 1, 10, 5, key="int_romberg_steps",
                                      disabled=method != "romberg")
            csv_file_int = st.file_uploader("Upload CSV Data", type=["csv"], key="int_csv_file")
            text_input_data_int = st.text_area("Or input data here (x, f(x) per line)", key="int_data_text")

            calculate_int = st.button("Calculate", key="calculate_int")

            if calculate_int:
                try:
                    # CSV Input
                    if csv_file_int:
                        df = pd.read_csv(csv_file_int, header=None, names=["x", "f"])
                        x = df["x"].to_numpy()
                        f_values = df["f"].to_numpy()
                        a, b = x[0], x[-1]

                        if method == "trapezoidal":
                            result = composite_trapezoidal(x=x, f_values=f_values, a=a, b=b,
                                                           n=int((b - a) / np.diff(x)[0]))
                        elif method == "simpsons":
                            result = composite_simpsons(x=x, f_values=f_values, a=a, b=b,
                                                        n=int((b - a) / np.diff(x)[0]))
                        elif method == "romberg":
                            result = romberg_integration(lambda val: np.interp(val, x, f_values), a, b,
                                                         romberg_steps)
                        else:
                            raise ValueError("Unsupported method.")

                        plot_int(lambda val: np.interp(val, x, f_values), a, b, function_str_int)
                        st.write(f"Result: {result}")

                    # Text Input
                    elif text_input_data_int:
                        lines = text_input_data_int.splitlines()
                        data = [list(map(float, line.split(','))) for line in lines if line]
                        x = np.array([row[0] for row in data])
                        f_values = np.array([row[1] for row in data])
                        a, b = x[0], x[-1]

                        if method == "trapezoidal":
                            result = composite_trapezoidal(x=x, f_values=f_values, a=a, b=b,
                                                           n=int((b - a) / np.diff(x)[0]))
                        elif method == "simpsons":
                            result = composite_simpsons(x=x, f_values=f_values, a=a, b=b,
                                                        n=int((b - a) / np.diff(x)[0]))
                        elif method == "romberg":
                            result = romberg_integration(lambda val: np.interp(val, x, f_values), a, b,
                                                         romberg_steps)
                        else:
                            raise ValueError("Unsupported method.")

                        plot_int(lambda val: np.interp(val, x, f_values), a, b, function_str_int)
                        st.write(f"Result: {result}")

                    # Function Input
                    else:
                        x_sym = sympy.Symbol('x')
                        f_sym = sympy.sympify(function_str_int)
                        f = sympy.lambdify(x_sym, f_sym, "numpy")

                        if method == "trapezoidal":
                            result = composite_trapezoidal(f, a, b, n)
                        elif method == "simpsons":
                            result = composite_simpsons(f, a, b, n)
                        elif method == "romberg":
                            result = romberg_integration(f, a, b, romberg_steps)
                        else:
                            raise ValueError("Unsupported method.")

                        plot_int(f, a, b, function_str_int)
                        st.write(f"Result: {result}")

                except (ValueError, TypeError, sympy.SympifyError, pd.errors.ParserError) as e:
                    st.error(f"Error: {e}")

            if st.button("Clear", key="clear_int"):
                st.session_state["int_func"] = "x**2 + 2*x + 1"
                st.session_state["int_a"] = 0.0
                st.session_state["int_b"] = 1.0
                st.session_state["int_n"] = 10
                st.session_state["int_method"] = "trapezoidal"
                st.session_state["int_romberg_steps"] = 5


if __name__ == "__main__":
    main()