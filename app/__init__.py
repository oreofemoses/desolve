# def plot_graph(self, f, x0, h, order, terms):
#     self.clear_plot()
#     x_range = 2 / h if abs(x0) > 1e-3 else 1
#     x_vals = np.linspace(x0 - x_range, x0 + x_range, 100)
#     f_values = np.array([forward_diff(f, x, h, order) for x in x_vals])
#     for x in range(len(x_vals)):
#         print(f"{x_vals[x]} : {f_values[x]}\n")
#     fig = plt.Figure(figsize=(4, 3), dpi=100)
#     ax = fig.add_subplot(111)
#     ax.plot(x_vals, f_values, label=f"f(x) = {self.function_entry.get()}")
#     ax.plot(x0, f(x0), 'ro', label="Derivative point")
#     ax.set_xlabel("x")
#     ax.set_ylabel("f(x)")
#     ax.set_title(f"Numerical Differentiation (order: {order}, terms: {terms})")
#     ax.legend()
#
#     canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
#     canvas.draw()
#     canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
#
#     self.plot_widget = canvas
