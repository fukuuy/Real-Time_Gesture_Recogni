import tkinter as tk
from tkinter import ttk


class GUI:
    def __init__(self, save_callback):
        self.save_callback = save_callback
        self.should_exit = False
        self.should_save = False

        self.root = tk.Tk()
        self.root.title("数据采集")

        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack()

        label_frame = ttk.LabelFrame(control_frame, text="手势标签", padding="10")
        label_frame.pack(pady=5, fill=tk.X)
        self.label_entry = ttk.Entry(label_frame)
        self.label_entry.pack(pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)

        self.save_button = ttk.Button(
            button_frame,
            text="保存",
            command=self.on_save,
            width=15
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = ttk.Button(
            button_frame,
            text="退出",
            command=self.on_exit,
            width=15
        )
        self.exit_button.pack(side=tk.LEFT, padx=5)
        self.root.focus_force()

    def on_save(self):
        self.should_save = True

    def on_exit(self):
        self.should_exit = True
        self.root.destroy()

    def get_label(self):
        return self.label_entry.get()

    def reset_save_flag(self):
        self.should_save = False

    def update(self):
        self.root.update_idletasks()
        self.root.update()