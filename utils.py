import os, shutil
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tsmoothie.smoother import *
from tqdm import tqdm
import warnings 

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

              
              
def new_graphs():
    # First graph
    data = pd.read_csv('results.csv')

    frame_nox = [i for i in range(data.shape[0])]
    frame_no = len(frame_nox)
    emotions = [i for i in data.iloc[:, 1]]

    emotion_names = [i for i in data['class'].unique()]
    frequency_all = [i for i in data['class'].value_counts()]

    plt.bar(emotion_names, frequency_all)
    plt.savefig('graphs/1.jpg')
        
    # Second graph
    emotion_occurunce_frameno = []
    for i in tqdm(emotions):
        if i == 'Upset':
            emotion_occurunce_frameno.append(1)
        elif i == 'Annoyed':
            emotion_occurunce_frameno.append(2)
        elif i == 'Distress':
            emotion_occurunce_frameno.append(3)
        elif i == 'Satisfied':
            emotion_occurunce_frameno.append(4)
        elif i == 'Dissatisfied':
            emotion_occurunce_frameno.append(5)
        elif i == 'Neutral':
            emotion_occurunce_frameno.append(6)
        elif i == 'Amazed':
            emotion_occurunce_frameno.append(7)

        else:
            print(i)
    plt.scatter(frame_nox, emotion_occurunce_frameno)
    plt.savefig('graphs/2.jpg')
    
    
    # Third graph
    # amaze occurence at which frame, with cumulative plot
    fig, axes = plt.subplots(7, 1)

    upset_count = 1
    annoyed_count = 1
    distress_count = 1
    satisfied_count = 1
    dissatisfied_count = 1
    neutral_count = 1
    amazed_count = 1

    c_upset = []
    c_annoyed = []
    c_distress = []
    c_satisfied = []
    c_dissatisfied = []
    c_neutral = []
    c_amazed = []
    
    time.sleep(0.3)
    for i in tqdm(emotions):
        if i == 'Upset':
            upset_count += 1
            c_upset.append(upset_count)
        else:
            c_upset.append(upset_count)

        if i == 'Annoyed':
            annoyed_count += 1
            c_annoyed.append(annoyed_count)
        else:
            c_annoyed.append(annoyed_count)

        if i == 'Distress':
            distress_count += 1
            c_distress.append(distress_count)
        else:
            c_distress.append(distress_count)

        if i == 'Satisfied':
            satisfied_count += 1
            c_satisfied.append(satisfied_count)
        else:
            c_satisfied.append(satisfied_count)

        if i == 'Dissatisfied':
            dissatisfied_count += 1
            c_dissatisfied.append(dissatisfied_count)
        else:
            c_dissatisfied.append(dissatisfied_count)

        if i == 'Amazed':
            amazed_count += 1
            c_amazed.append(dissatisfied_count)
        else:
            c_amazed.append(dissatisfied_count)

        if i == 'Neutral':
            neutral_count += 1
            c_neutral.append(neutral_count)
        else:
            c_neutral.append(neutral_count)

    #change in emotions
    l = 0
    time.sleep(0.3)
    for k in range(100, frame_no, 100):
        change = (c_upset[k] - c_upset[l])/100
        if change >= 0.1:
            print('increase in Upset emotions between', l, 'and', k)
        l = k

    l = 0
    for k in range(100, frame_no, 100):
        change = (c_annoyed[k] - c_annoyed[l])/100
        if change >= 0.1:
            print('increase in Annoyed emotions between', l, 'and', k)
        l = k

    l = 0
    for k in range(100, frame_no, 100):
        change = (c_distress[k] - c_distress[l])/100
        if change >= 0.1:
            print('increase in Distress emotions between', l, 'and', k)
        l = k

    l = 0
    for k in range(100, frame_no, 100):
        change = (c_satisfied[k] - c_satisfied[l])/100
        if change >= 0.1:
            print('increase in Satisfied emotions between', l, 'and', k)
        l = k

    ########################################################################################3    
    l = 0
    for k in range(100, frame_no, 100):
        change = (c_dissatisfied[k] - c_dissatisfied[l])/100
        if change >= 0.1:
            print('increase in Dissatisfied emotions between', l, 'and', k)
        l = k

    l = 0
    for k in range(100, frame_no, 100):
        change = (c_neutral[k] - c_neutral[l])/100
        if change >= 0.1:
            print('increase in Neutral emotions between', l, 'and', k)
        l = k

    # cumulative plot
    #j = 0
    #c_engaged = []
    # for i in emotions:
    #    if i == 'Engaged':
    #        j += 1
    #        c_engaged.append(j)
    #    else:
    #        c_engaged.append(j)

    axes[0].plot(frame_nox, c_upset, 'r')
    axes[1].plot(frame_nox, c_annoyed, 'g')
    axes[2].plot(frame_nox, c_distress, 'b')
    axes[3].plot(frame_nox, c_satisfied, 'y')
    axes[3].plot(frame_nox, c_dissatisfied, 'p')
    axes[3].plot(frame_nox, c_neutral, 'm')
    axes[3].plot(frame_nox, c_amazed, 'o')

    plt.savefig('graphs/3.jpg')
        
def graph():
    print('Generating graphs.....')
    time.sleep(1)
    # Creating graphs
    df = pd.read_csv('results.csv')
    categories = []
    size = []
    if len(df['class']) != 0:
        unique = df['class'].unique()

        for value in range(len(df['class'].value_counts())):
            size.append(df['class'].value_counts()[value])

        for value in range(len(unique)):
            categories.append(unique[value])
    # Pie chart
    plt.figure(figsize=(15,13))
    plt.title('Percentage of emotions')
    plt.pie(size, labels=categories, autopct='%1.1f%%')
    plt.savefig('graphs/pie.jpg')

    # Barplot
    plt.figure(figsize=(15,13))
    sns.barplot(categories, df['class'].value_counts(), palette='Greens_d')
    plt.xlabel('Emotions')
    plt.ylabel('Emotion')
    plt.title('Value Count of emotions')
    plt.savefig('bar.jpg')

    # basic changes
    #Creating a new column
    df['ranking'] = np.zeros((df.shape[0],1))
    time.sleep(0.3)
    for i in tqdm(range(df.shape[0])):
        category = df['class'][i]

        if category == 'Amazed':
            df['ranking'][i] = 0.9
        elif category == 'Satisfied':
            df['ranking'][i] = 0.7
        elif category == 'Neutral':
            df['ranking'][i] = 0.5
        else:
            df['ranking'][i] = 0.3

    smoother = ConvolutionSmoother(window_len=30, window_type='ones')
    smoother.smooth(df['ranking'])

    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)

    # plot the smoothed timeseries with intervals
    plt.figure(figsize=(15,13))
    plt.plot(smoother.data[0], color='orange')
    plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
    plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Satisfaction')
    plt.title('Satisfaction rate over time')
    plt.savefig('graphs/dist.jpg')


    # Basic changes
    df['posneg'] = np.zeros((df.shape[0],1))
    time.sleep(0.3)
    for i in tqdm(range(df.shape[0])):
        if df['class'][i] =='Amazed' or df['class'][i] == 'Satisfied' or df['class'][i] == 'Neutral':
            df['posneg'][i] = 1
        else:
            df['posneg'][i] = 0


    if len(df['posneg'].unique()) > 1:
        y = [df['posneg'].value_counts()[0], df['posneg'].value_counts()[1]]

        plt.figure(figsize=(15,13))
        sns.barplot(['Negative','Positive'],y, alpha=0.9)
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Value Count of Positive and Negative emotions')
        plt.savefig('graphs/posneg.jpg')

    

        
def to_mp4():
    print()
    print()
    
    # Converting detected frames to an output video
    fps = 30
    path = 'frames/'
    name = 'output/output.mp4'
    img_file = [path+str(i)+'.jpg' for i in range(len(os.listdir(path)))]
    img = cv2.imread(img_file[0])
    height, width, channel = img.shape

    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    print('[!] Converting to MP4...')
    time.sleep(2)

    for image in tqdm(img_file):
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


    print('[!] Done!')
              
