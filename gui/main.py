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
        self.vid = MyVideoCapture(self.videosource)
        self.label = tk.Label(self,text="testing", font = 15, bg="black",fg="white").pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(self, width = 400, height= 400)
        self.canvas.pack() 
        self.btn_snapshot = tk.Button(self, text="Capture", width=30, bg="blue",activebackground = "red",command = self.snapshot)
        self.btn_snapshot.pack(anchor="center", expand=True)
        self.update()
    
    def snapshot(self):
        check,frame = self.vid.getFrame()
        if check:
            # image = "IMG-"+ time.strftime("%H-%M-%S-%d-%m") + ".png"
            image = "image.png"
            cv2.imwrite(image, cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
            msg = tk.Label(self, text='image saved', bg= 'black', fg='green').place(x=430,y=510)

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
            
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
            

class Page2(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 2")
       label.pack(side="top", fill="both", expand=True)


class Page3(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 3")
       label.pack(side="top", fill="both", expand=True)


class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)
        p3 = Page3(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="Page 1", command=p1.lift)
        b2 = tk.Button(buttonframe, text="Page 2", command=p2.lift)
        b3 = tk.Button(buttonframe, text="Page 3", command=p3.lift)

        b1.pack(side="left")
        b2.pack(side="left")
        b3.pack(side="left")

        p1.show()

if __name__ == "__main__":
    root = tk.Tk()
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("800x800")
    root.mainloop()
