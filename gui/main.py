from tkinter import *
import cv2
from PIL import Image, ImageTk

import time


class App:
    def __init__(self,videosource=0):
        self.appName = "ASL"
        self.window = Tk()
        self.window.title = "ASL"
        self.window.resizable(0,0)
        self.window['bg'] = 'gray'
        self.videosource = videosource
    
        self.vid = MyVideoCapture(self.videosource)
        self.label = Label(self.window,text=self.appName, font = 15, bg="black",fg="white").pack(side =TOP, fill= BOTH)

        self.canvas = Canvas(self.window,width = self.vid.width, height= self.vid.height)

        self.canvas.pack()

        self.btn_snapshot = Button(self.window, text="Capture", width=30, bg="blue",activebackground = "red",command = self.snapshot)

        self.btn_snapshot.pack(anchor=CENTER, expand=True)
        self.update()
        self.window.mainloop()
    
    def snapshot(self):
        check,frame = self.vid.getFrame()
        if check:
            # image = "IMG-"+ time.strftime("%H-%M-%S-%d-%m") + ".png"
            image = "image.png"
            cv2.imwrite(image, cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
            msg = Label(self.window, text='image saved', bg= 'black', fg='green').place(x=430,y=510)

    def update(self):
        isTrue,frame = self.vid.getFrame()

        if isTrue:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0,0,image = self.photo, anchor = NW)

        self.window.after(15,self.update)


class MyVideoCapture:
    def __init__(self, videosource=0):
        self.vid = cv2.VideoCapture(videosource)
        if not self.vid.isOpened():
            raise ValueError("Unable to access camera")

        #get vid source width and height

        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)

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


if (__name__ == "__main__"):
    App()