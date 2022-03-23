import os, shutil
from keras.models import load_model
import cv2
import pandas as pd
import subprocess as sa
import os
import random as rn
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tsmoothie.smoother import *
import warnings 
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

print('[!] Imported')

model = load_model('weights/ResNet-50.h5')
video = cv2.VideoCapture('input/input.mp4')
path = 'xml/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(path)
print('[!] Model Loaded')

text = ''
found = 0
index = 0
data = {
    'frame' : [],
    'class' : []
}

cap=cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    if _:
         # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3)

        if len(faces) > 0:
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                # Resized roi 
                res = resize(roi, (197,197,3))
                res = np.expand_dims(res, 0)

                # Feed the model the roi image
                pred = np.argmax(model.predict(res))
                if pred == 0 or pred == 3:
                    text = 'Satisfied'
                elif pred == 1 or pred == 4:
                    text = 'Unsatisfied'
                elif pred == 2 or pred == 5:
                    text = 'Amazed'
                else:
                    text = 'Engaged'

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                found+=1

                data['frame'].append(found)
                data['class'].append(text)

                if str(found)[-1] == '0':
                    print(f'[!] {found} faces detected so far!')
                    
        cv2.imshow('Results', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release() 
            False

        #cv2.imwrite(f'frames/{str(index)}.jpg', frame)
        index +=1
        
    else:
        break
    
    


cv2.destroyAllWindows()
print(f'Total Frames: {index}')
print(f'{found} faces found!')
df = pd.DataFrame(data, columns=['frame', 'class'])
df.to_csv('results.csv', index=False, header=True)