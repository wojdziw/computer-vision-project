from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk
import cv2
import numpy as np

def indicateLocation(image):
    root = Tk()

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    rgbImage = np.zeros(image.shape, np.uint8)
    rgbImage[:,:,0] = image[:,:,2]
    rgbImage[:,:,2] = image[:,:,0]
    rgbImage[:,:,1] = image[:,:,1]

    image = rgbImage

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    canvas.config(width=imageWidth, height=imageHeight)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    # File = askopenfilename(parent=root, initialdir="~",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.fromarray(image, 'RGB'))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    def printcoords(event):
        global a
        global b
        a = event.x
        b = event.y

        root.destroy()

    canvas.bind("<Button 1>",printcoords)

    root.mainloop()

    return a, b