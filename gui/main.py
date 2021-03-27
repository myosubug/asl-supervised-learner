import tkinter as tk

WIDTH = 500
HEIGHT = 500

count  = 0

# Increments counter by 1
def buttonCallback():
    global count
    count += 1
    message.configure(text=f'{count} clicks!')

# Initialize gui
root = tk.Tk()
root.geometry(f'{WIDTH}x{HEIGHT}')
root.title("ASL Detection")

message = tk.Label(root)
message.configure(text=f'{count} clicks!')
message.pack()

button = tk.Button(text="Click me!", command=buttonCallback)
button.pack()

# Runs the gui
root.mainloop()
