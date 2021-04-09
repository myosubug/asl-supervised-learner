'''
This is another GUI application as proof of concept
Our model will predict an ASL alphabet from randomly picked images from 27 classes

Below are all the libraries that requires to run this GUI.
Please make sure that you have all of these in your Anaconda Environment
Also, especially for opencv library ("cv2"), you must have import from Spyder console by
typing follow lines and hit enter

pip install opencv-python

'''


import tkinter as tk
import cv2
from PIL import Image, ImageTk
import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from skimage.transform import resize
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.svm import SVC
from skimage.transform import resize
from skimage.color import rgb2grey
import random


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
    def show(self):
        self.lift()
        
class Page1(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.img = None
        self.randomly_picked = None
        self.pic_list = ["A_test","B_test","C_test","D_test","E_test","F_test","G_test","H_test","I_test","J_test","K_test","L_test","M_test","N_test","O_test","P_test","Q_test","R_test","S_test","T_test","U_test","V_test","W_test","X_test","Y_test","Z_test","nothing_test"]
        self.text = tk.StringVar()
        self.text.set("ASL Webcam Image Classification")
        self.canvas = tk.Canvas(self, width = 400, height= 400)
        self.canvas.pack()
        self.btn_pick = tk.Button(self, text="Pick", width=30, bg="blue",activebackground = "red",command = self.pick)
        self.btn_predcit = tk.Button(self, text="Predict", width=30, bg="blue",activebackground = "red",command = self.predict)
        self.result = tk.Label(self, textvariable=self.text, font = 45, bg="black",fg="white").pack(side="bottom", fill="both", expand=True)
        self.btn_pick.pack(anchor="center", expand=True)
        self.btn_predcit.pack(anchor="center", expand=True)
        self.update()  
    
    def predict(self):
        snapped = []
        prediction = []
        self.text.set("The prediction is.....")

    def pick(self):
        self.randomly_picked = random.choice(self.pic_list)
        print("randomly picked.. " + self.randomly_picked)
        self.img = ImageTk.PhotoImage(Image.open("data/"+self.randomly_picked+".jpeg"))
        self.canvas.create_image(110,50, anchor="nw", image=self.img)
        self.update()
        
    def show(self):
        self.lift()



class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)
        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p1.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ASL Webcam Image Classification")
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("800x800")
    root.mainloop()
