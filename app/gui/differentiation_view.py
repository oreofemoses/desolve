import tkinter as tk
from tkinter import ttk
import sympy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app.gui.error_view import ErrorView
from core.differentiation.finite_difference import forward_diff


class DifferentiationView(ttk.Frame):
    """
    Panel for numerical differentiation input and display.
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

        # Point of Evaluation Input
        ttk.Label(self, text="Point of Evaluation (x0)", style="TLabel").grid(row=1, column=0, sticky=tk.W)
        self.x0_entry = ttk.Spinbox(self, from_=-100, to=100, width=10, style="TSpinbox")
        self.x0_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Step Size Input
        ttk.Label(self, text="Step Size (h)", style="TLabel").grid(row=2, column=0, sticky=tk.W)
        self.h_entry = ttk.Spinbox(self, from_=0.0001, to=1, increment=0.01, width=10, style="TSpinbox")
        self.h_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Derivative Order Input
        ttk.Label(self, text="Derivative Order", style="TLabel").grid(row=3, column=0, sticky=tk.W)
        self.order_combobox = ttk.Combobox(self, values=[1, 2, 3], width=10, style="TCombobox")
        self.order_combobox.set(1)
        self.order_combobox.grid(row=3, column=1, sticky=tk.W, padx=5)

        # Number of Terms Input
        ttk.Label(self, text="Number of Terms", style="TLabel").grid(row=4, column=0, sticky=tk.W)
        self.terms_entry = ttk.Spinbox(self, from_=1, to=10, width=10, style="TSpinbox")
        self.terms_entry.grid(row=4, column=1, sticky=tk.W, padx=5)

        # Calculate Button
        self.calculate_button = ttk.Button(self, text="Calculate", command=self.calculate_derivative, style="TButton")
        self.calculate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Output Display
        ttk.Label(self, text="Result:", style="TLabel").grid(row=6, column=0, sticky=tk.W)
        self.output_label = ttk.Label(self, text="", style="TLabel")
        self.output_label.grid(row=6, column=1, sticky=tk.W, padx=5)

        # Plot area
        self.plot_frame = ttk.Frame(self, style="my.TFrame")
        self.plot_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky=tk.NSEW)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        # Clear Button
        self.clear_button = ttk.Button(self, text="Clear", command=self.clear_inputs, style="TButton")
        self.clear_button.grid(row=7, column=0, columnspan=2, pady=10)

    def calculate_derivative(self):
        try:
            function_str = self.function_entry.get()
            x0 = float(self.x0_entry.get())
            h = float(self.h_entry.get())
            order = int(self.order_combobox.get())
            terms = int(self.terms_entry.get())

            x_sym = sympy.Symbol('x')
            f_sym = sympy.sympify(function_str)
            f = sympy.lambdify(x_sym, f_sym, "numpy")

            result = forward_diff(f, x0, h, order, terms)
            self.output_label.config(text=result)
            self.plot_graph(f, x0, h, order, terms)
        except (ValueError, TypeError, sympy.SympifyError) as e:
            self.error_view.show_error(str(e))

    def clear_inputs(self):
        self.function_entry.delete(0, tk.END)
        self.x0_entry.delete(0, tk.END)
        self.h_entry.delete(0, tk.END)
        self.order_combobox.set(1)
        self.terms_entry.delete(0, tk.END)
        self.output_label.config(text="")
        self.clear_plot()

    def plot_graph(self, f, x0, h, order, terms):
        self.clear_plot()
        x_range = 2 * h if abs(x0) > 1e-3 else 1
        x_vals = np.linspace(x0 - x_range, x0 + x_range, 500)
        f_values = np.array([f(x) for x in x_vals])
        fig = plt.Figure(figsize=(4, 3), dpi=100, facecolor="#B1DDC6")
        ax = fig.add_subplot(111, facecolor="#B1DDC6")
        ax.plot(x_vals, f_values, label=f"f(x) = {self.function_entry.get()}", color="#007bff")
        ax.plot(x0, f(x0), 'ro', label="Derivative point")
        ax.set_xlabel("x", color="#333")
        ax.set_ylabel("f(x)", color="#333")
        ax.set_title(f"Numerical Differentiation (order: {order}, terms: {terms})", color="#333")
        ax.tick_params(colors="#333")
        ax.legend(facecolor="#B1DDC6", edgecolor="white", labelcolor="#333")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

        self.plot_widget = canvas

    def clear_plot(self):
        if hasattr(self, 'plot_widget') and self.plot_widget:
            self.plot_widget.get_tk_widget().destroy()