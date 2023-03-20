import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
# root = Tk()
# img = ImageTk.PhotoImage(Image.open("logo.png"))
# panel = Label(root, image = img)
# panel.pack(side = "bottom", fill = "both", expand = "yes")
# root.mainloop()
root=tk.Tk()   
img = ImageTk.PhotoImage(Image.open("logo.png"))
heading = Label(root,image=img,bg='black')

heading.pack() 
heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 

heading2.pack()
lmain = tk.Label(master=root,padx=50,bd=10)
lmain2 = tk.Label(master=root,bd=10)

lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
lmain.pack(side=LEFT)
lmain.place(x=50,y=250)
lmain3.pack()
lmain3.place(x=960,y=250)
lmain2.pack(side=RIGHT)
lmain2.place(x=900,y=350)
    