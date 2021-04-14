'''
Below are all the libraries that requires to run this GUI.
Please make sure that you have all of these in your Anaconda Environment
Also, especially for opencv library ("cv2"), you must have import from Spyder console by
typing follow lines and hit enter

pip install opencv-python

'''

import tkinter as tk
import cv2
from PIL import Image, ImageTk


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
    def show(self):
        self.lift()
        
        
class Page1(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.videosource = 0
        self.text = tk.StringVar()
        self.text.set("ASL Webcam Image Classification")
        self.vid = MyVideoCapture(self.videosource)
        self.label = tk.Label(self,text="Pose your hand to camera and Click Capture button to see the result", font = 50, bg="black",fg="white").pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self, width = 400, height= 400)
        self.canvas.pack() 
        self.btn_snapshot = tk.Button(self, text="Capture", width=30, bg="blue",activebackground = "red",command = self.snapshot)
        self.result = tk.Label(self, textvariable=self.text, font = 45, bg="black",fg="white").pack(side="bottom", fill="both", expand=True)
        self.btn_snapshot.pack(anchor="center", expand=True)
        self.update()
    
    def snapshot(self):
        check,frame = self.vid.getFrame()
        resize = cv2.resize(frame, (360, 240))
        snapped = []
        prediction = []
        if check:
            image = "image.jpeg"
            cv2.imwrite(image, cv2.cvtColor(resize, cv2.COLOR_BGRA2RGB))
            self.text.set("The image has been saved!, The prediction is...")
        else:
            self.text.set("Failed to take capture, please try again.")
                

    def update(self):
        isTrue, frame = self.vid.getFrame()
        if isTrue:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0,0,image = self.photo, anchor = "nw")
        self.after(15,self.update)
        
    def show(self):
        self.lift()


class MyVideoCapture:
    def __init__(self, videosource=0):
        self.vid = cv2.VideoCapture(videosource)
        if not self.vid.isOpened():
            raise ValueError("Unable to access camera")
           
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def getFrame(self):
        if self.vid.isOpened():
            isTrue, frame = self.vid.read()

            if isTrue:
                return(isTrue,cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB))
            else:
                return (isTrue,None)
        else:
            return(isTrue,None)
    
    def __delattr__(self):
        if self.vid.isOpened():
            sefd.vid.release()
            


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
