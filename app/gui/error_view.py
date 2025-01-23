import tkinter as tk
from tkinter import ttk


class ErrorView(tk.Toplevel):
    """
    Pop up window to display errors to the user.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Error")
        self.geometry("300x100")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")

        self.error_label = tk.Label(self, text="", wraplength=280, bg="#f0f0f0", font=("Arial", 10), foreground="#333")
        self.error_label.pack(pady=20)

    def show_error(self, message):
        self.error_label.config(text=message)
        self.grab_set()
        self.wait_window()