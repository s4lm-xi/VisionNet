import os, shutil
from keras.models import load_model
import cv2
import pandas as pd
import subprocess as s
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
import sys
import argparse
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# name of the python file with graph functions
import utils

# Taking arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', help='directory to the input video', type=str)
parser.add_argument('-a', '--alert', help='Alert when the whole operation is done', default=0, type=int)
args = parser.parse_args()


if not os.path.exists(str(args.input)):
    print('No such file or directory!')
    sys.exit()


folder = 'frames'
print('[!] Deleting contents of frame/..')
time.sleep(2)

for filename in tqdm(os.listdir(folder)):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))



# Detection
print('[!] Detecting')

model = load_model('weights/ResNet-50.h5')
video = cv2.VideoCapture('input/input.mp4')
path = 'xml/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(path)

text = ''
found = 0
index = 0
data = {
    'frame' : [],
    'class' : []
}


while True:
    _, frame = video.read()
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
                if pred == 0:
                    text = 'Upset'
                if pred == 1:
                    text = 'Annoyed'
                if pred == 2:
                    text = 'Distress'
                if pred == 3:
                    text = 'Satisfied'
                if pred == 4:
                    text = 'Dissatisfied'
                if pred == 5:
                    text = 'Amazed'
                else:
                    text = 'Neutral'

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                found+=1

                data['frame'].append(found)
                data['class'].append(text)

                if str(found)[-1] == '0':
                    print(f'[!] {found} faces detected so far!')

        cv2.imwrite(f'frames/{str(index)}.jpg', frame)
        index +=1
        
    else:
        break
    
    



print(f'Total Frames: {index}')
print(f'{found} faces found!')
df = pd.DataFrame(data, columns=['frame', 'class'])
df.to_csv('results.csv', index=False, header=True)


# Graphs data

utils.graph()


# New graphs
utils.new_graphs()


# Convert detected frames to mp4 output
utils.to_mp4()




