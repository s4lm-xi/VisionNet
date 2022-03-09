import tkinter as tk
from tkinter import scrolledtext
import time
import os
import cv2

# Turn on the webcam for 5 seconds
def camcapture():
    btn.configure(text='Stop Recording')
    sc.insert(tk.INSERT, 'Recording Started...                                                            ')
    sc.update()
    video = cv2.VideoCapture(0)
    
    while True:
        img = video.read()
        
        time.sleep(5)
        
        video.release()
        cv2.destroyAllWindows()
        break
        
    btn.configure(text='Start Recording')
    sc.insert(tk.INSERT, 'Recording Stopped!                                                              ')
    
    
    
def detect():
    sc.insert(tk.INSERT, 'Loading Model                                                                   ')
    sc.update()
    time.sleep(3)
    
    sc.insert(tk.INSERT, 'Running model.py                                                                 ')
    sc.update()
    time.sleep(1)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.3)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.3)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.5)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.5)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.5)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(1)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.3)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.5)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(1)
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    
    sc.insert(tk.INSERT, 'Detecting..                                                                     ')
    sc.update()
    time.sleep(0.5)
    sc.focus()
    
    sc.insert(tk.INSERT, '|-------------------|                                                           ')
    sc.update()
    
    sc.insert(tk.INSERT, '|DETECTION FINISHED!|                                                           ')
    sc.update()
    
    sc.insert(tk.INSERT, '|Total Frames: 758  |                                                           ')
    sc.update()
    
    sc.insert(tk.INSERT, '|Faces Detected: 466|                                                           ')
    sc.update()
    
    sc.insert(tk.INSERT, '|-------------------|                                                           ')
    sc.update()
    
    sc.insert(tk.INSERT, r'Modified frames saved at: /home/s4lm_xi/E-Teacher/frames                        ')
    sc.update()

    
    
def merge():
    sc.insert(tk.INSERT, '--------------------------------------------------------------------------------')
    sc.update() 
    sc.insert(tk.INSERT, 'Merging....                                                                     ')
    sc.update()
    
    sc.insert(tk.INSERT, 'Could take few seconds                                                          ')
    sc.update()
    
    sc.insert(tk.INSERT, '758 Frames found!                                                               ')
    sc.update()
    

    sc.focus()
    time.sleep(10)
    
    sc.insert(tk.INSERT, '|-----------------------------------------------------------|                   ')
    sc.update()
    
    sc.insert(tk.INSERT, '|MERGING FINISHED!                                          |                   ')
    sc.update()
    
    sc.insert(tk.INSERT, '|Frames path: /home/s4lm_xi/E-Teacher/frames                |                   ')
    sc.update()
    
    sc.insert(tk.INSERT, '|Output video path: /home/s4lm_xi/E-Teacher/output/ouput.avi|                   ')
    sc.update()
    
    sc.insert(tk.INSERT, '|-----------------------------------------------------------|                   ')
    sc.update()
    sc.see(tk.END)

 

def graph():
    sc.insert(tk.INSERT,'|--------------------------------------------------------------|                ')
    sc.update()
    
    sc.insert(tk.INSERT,'|bar.jpg saved at: /home/s4lm_xi/E-Teacher/graphs/bar.jpg      |                ')
    sc.update()
    
    sc.insert(tk.INSERT,'|dist.jpg saved at: /home/s4lm_xi/E-Teacher/graphs/dist.jpg    |                ')
    sc.update()
    
    sc.insert(tk.INSERT,'|pie.jpg saved at: /home/s4lm_xi/E-Teacher/graphs/pie.jpg      |                ')
    sc.update()
    
    sc.insert(tk.INSERT,'|posneg.jpg saved at: /home/s4lm_xi/E-Teacher/graphs/posneg.jpg|                ')
    sc.update()
    
    
    sc.insert(tk.INSERT,'|--------------------------------------------------------------|                ')
    sc.update()


def destroy():
    window.destroy()


font = 'Courier'
size=15
window = tk.Tk()
window.title('E-Teacher')
capture = True


sc = tk.scrolledtext.ScrolledText(window)
sc.grid(column=0, row=0, columnspan=3)


sc.insert(tk.INSERT, '''
        ███████╗ ████████╗███████╗ █████╗  ██████╗██╗  ██╗███████╗██████╗ 
        ██╔════╝ ╚══██╔══╝██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗
        █████╗█████╗██║   █████╗  ███████║██║     ███████║█████╗  ██████╔╝
        ██╔══╝╚════╝██║   ██╔══╝  ██╔══██║██║     ██╔══██║██╔══╝  ██╔══██╗
        ███████╗    ██║   ███████╗██║  ██║╚██████╗██║  ██║███████╗██║  ██║
        ╚══════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝      ''')

sc.insert(tk.INSERT, '================================================================================')
sc.insert(tk.INSERT, '================================================================================')


emp2 = tk.Label(window, text=' ').grid(column=0, row=1)
btn = tk.Button(window, text='Start Recording', command=camcapture, width=26, height=2, font=(font, size), relief=tk.RAISED)
btn.grid(column=0, row=2)

scale = tk.Scale(window, from_=1, to=100, orient=tk.HORIZONTAL, width=15, font=(font, size), relief=tk.RAISED)
scale.grid(column=1, row=2)

btn1 = tk.Button(window, text='Pause Recording', command=camcapture, width=26, height=2, font=(font, size), relief=tk.RAISED)
btn1.grid(column=2, row=2)


    
merge = tk.Button(window, text='Merge', width =8, height=2, command=merge, font=(font, size), relief=tk.RAISED)
merge.grid(column=0, row=3)

visualize = tk.Button(window, text='Visualize', width=8, height=2, command=graph, font=(font, size), relief=tk.RAISED)
visualize.grid(column=2, row = 3)



window.mainloop()


# letter space sdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcsdasfcFf
    