import tkinter as tk
from tkinter import ttk
import sympy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app.gui.error_view import ErrorView
from core.integration.trapezoidal import composite_trapezoidal
from core.integration.simpsons import composite_simpsons
from core.integration.romberg import romberg_integration


class IntegrationView(ttk.Frame):
    """
    Panel for numerical integration input and display.
    """

    def __init__(self, parent):
        super().__init__(parent, padding="10", style="my.TFrame")
        self.parent = parent
        self.error_view = ErrorView(parent)
        self.create_widgets()

    def create_widgets(self):
        # Function Input
        ttk.Label(self, text="Function (e.g., x**2 + sin(x))", style="TLabel").grid(row=0, column=0, sticky=tk.W)
        self.function_entry = ttk.Entry(self, width=30, style="TEntry")
        self.function_entry.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Lower Limit Input
        ttk.Label(self, text="Lower Limit (a)", style="TLabel").grid(row=1, column=0, sticky=tk.W)
        self.a_entry = ttk.Spinbox(self, from_=-100, to=100, width=10, style="TSpinbox")
        self.a_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Upper Limit Input
        ttk.Label(self, text="Upper Limit (b)", style="TLabel").grid(row=2, column=0, sticky=tk.W)
        self.b_entry = ttk.Spinbox(self, from_=-100, to=100, width=10, style="TSpinbox")
        self.b_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Number of Subintervals Input
        ttk.Label(self, text="Number of Subintervals (N)", style="TLabel").grid(row=3, column=0, sticky=tk.W)
        self.n_entry = ttk.Spinbox(self, from_=1, to=10000, width=10, style="TSpinbox")
        self.n_entry.grid(row=3, column=1, sticky=tk.W, padx=5)

        # Integration Method
        ttk.Label(self, text="Integration Method", style="TLabel").grid(row=4, column=0, sticky=tk.W)
        self.method_combobox = ttk.Combobox(self, values=["trapezoidal", "simpsons", "romberg"], width=10,
                                            style="TCombobox")
        self.method_combobox.set("trapezoidal")
        self.method_combobox.grid(row=4, column=1, sticky=tk.W, padx=5)

        # Romberg Steps
        ttk.Label(self, text="Romberg Steps", style="TLabel").grid(row=5, column=0, sticky=tk.W)
        self.romberg_steps_entry = ttk.Spinbox(self, from_=1, to=10, width=10, style="TSpinbox")
        self.romberg_steps_entry.grid(row=5, column=1, sticky=tk.W, padx=5)

        # Calculate Button
        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate_integral, style="TButton")
        self.calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Output Display
        ttk.Label(self, text="Result:", style="TLabel").grid(row=7, column=0, sticky=tk.W)
        self.output_label = ttk.Label(self, text="", style="TLabel")
        self.output_label.grid(row=7, column=1, sticky=tk.W, padx=5)

        # Plot area
        self.plot_frame = ttk.Frame(self, style="my.TFrame")
        self.plot_frame.grid(row=9, column=0, columnspan=2, pady=10, sticky=tk.NSEW)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        # Clear Button
        self.clear_button = ttk.Button(self, text="Clear", command=self.clear_inputs, style="TButton")
        self.clear_button.grid(row=8, column=0, columnspan=2, pady=10)

    def calculate_integral(self):
        try:
            function_str = self.function_entry.get()
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            n = int(self.n_entry.get())
            method = self.method_combobox.get()

            x_sym = sympy.Symbol('x')
            f_sym = sympy.sympify(function_str)
            f = sympy.lambdify(x_sym, f_sym, "numpy")

            if method == "trapezoidal":
                result = composite_trapezoidal(f, a, b, n)
            elif method == "simpsons":
                n = int(self.n_entry.get())
                romberg_steps = int(self.romberg_steps_entry.get()) if self.romberg_steps_entry.get() else 0

                if romberg_steps == 0:
                    result = composite_simpsons(f, a, b, n, rule="1/3")
                else:
                    result = romberg_integration(f, a, b, romberg_steps, rule="simpsons")

            elif method == "romberg":
                n_steps = int(self.romberg_steps_entry.get())
                result = romberg_integration(f, a, b, n_steps, rule="trapezoidal")

            self.output_label.config(text=result)
            self.plot_graph(f, a, b)

        except (ValueError, TypeError, sympy.SympifyError) as e:
            self.error_view.show_error(str(e))

    def clear_inputs(self):
        self.function_entry.delete(0, tk.END)
        self.a_entry.delete(0, tk.END)
        self.b_entry.delete(0, tk.END)
        self.n_entry.delete(0, tk.END)
        self.method_combobox.set("trapezoidal")
        self.romberg_steps_entry.delete(0, tk.END)
        self.output_label.config(text="")
        self.clear_plot()

    def plot_graph(self, f, a, b):
        self.clear_plot()
        x_range = b - a if abs(b - a) > 1e-3 else 1
        x_vals = np.linspace(a - x_range / 2, b + x_range / 2, 500)
        f_values = np.array([f(x) for x in x_vals])
        fig = plt.Figure(figsize=(4, 3), dpi=100, facecolor="#B1DDC6")
        ax = fig.add_subplot(111, facecolor="#B1DDC6")
        ax.plot(x_vals, f_values, label=f"f(x) = {self.function_entry.get()}", color="#007bff")
        ax.fill_between(x_vals, f_values, alpha=0.3, color="#007bff")
        ax.set_xlabel("x", color="#333")
        ax.set_ylabel("f(x)", color="#333")
        ax.set_title(f"Numerical Integration (a={a}, b={b})", color="#333")
        ax.tick_params(colors="#333")
        ax.legend(facecolor="#B1DDC6", edgecolor="white", labelcolor="#333")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

        self.plot_widget = canvas

    def clear_plot(self):
        if hasattr(self, 'plot_widget') and self.plot_widget:
            self.plot_widget.get_tk_widget().destroy()