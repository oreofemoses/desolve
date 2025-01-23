import tkinter as tk
from tkinter import ttk
from app.gui.differentiation_view import DifferentiationView
from app.gui.integration_view import IntegrationView


class MainWindow(tk.Tk):
    """
    Main application window for DEsolve numerical methods tool.
    """

    def __init__(self):
        super().__init__()
        self.title("DEsolve: Numerical Methods Tool")
        self.geometry("1400x800")  # Increased window size
        self.configure(bg="#B1DDC6")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        self.style.configure('TButton', font=("Arial", 10, "bold"), foreground="#fff", background="#007bff",
                             borderwidth=0, relief="flat", padding=5)
        self.style.map('TButton',
                       foreground=[('active', '#fff')],
                       background=[('active', '#0056b3')]
                       )
        self.style.configure('TLabel', font=("Arial", 10), foreground="#333")
        self.style.configure('TCombobox', font=("Arial", 10), foreground="#333", borderwidth=0)
        self.style.configure('TSpinbox', font=("Arial", 10), foreground="#333", borderwidth=0)
        self.style.configure('TEntry', font=("Arial", 10), foreground="#333", borderwidth=0)

        self.differentiation_view = DifferentiationView(self)
        self.differentiation_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.integration_view = IntegrationView(self)
        self.integration_view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()