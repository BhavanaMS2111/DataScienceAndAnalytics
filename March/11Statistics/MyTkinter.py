# import tkinter as tk

# root = tk.Tk()  # Corrected 'TK' to 'Tk'
# root.title('Sample TKINTER APP')
# root.geometry("200x100")  # Fixed the syntax for dimensions

# def say_hello():
#     print("Hello World")

# hello_button = tk.Button(root, text='Click Me', command=say_hello)  # Fixed 'CLick Me' to 'Click Me'
# hello_button.pack(pady=20)

# root.mainloop()


# from tkinter import *
# from tkinter import ttk
# root = Tk()
# frm = ttk.Frame(root, padding=10)
# frm.grid()
# ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
# ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
# root.mainloop()

import tkinter as tk
from tkinter import messagebox

# Create the main application window
root = tk.Tk()
root.title("Greeting App")
root.geometry("300x200")  # Set the size of the window

# Function to display the greeting
def greet():
    name = name_entry.get()
    if name:
        messagebox.showinfo("Greeting", f"Hello, {name}!")
    else:
        messagebox.showwarning("Input Error", "Please enter your name.")

# Create a label
name_label = tk.Label(root, text="Enter your name:")
name_label.pack(pady=10)  # Padding around the widget

# Create an entry box for user input
name_entry = tk.Entry(root)
name_entry.pack(pady=10)

# Create a button that triggers the greet function
greet_button = tk.Button(root, text="Greet", command=greet)
greet_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()