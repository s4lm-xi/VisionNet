import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

emotions_names = ['Upset', 'Annoyed', 'Distress', 'Satisfied', 'Dissatisfied', 'Amazed', 'Neutral']
emotions = []
with open('results.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    frame_no = 0
    frame_nox = []
    for line in csv_reader:
        frame_no += 1
        frame_nox.append(frame_no)
        emotions.append(line["class"])

# bar plot, plotting the frequency of each emotion
upset_frequency = emotions.count('Upset')
annoyed_frequency = emotions.count('Annoyed')
distress_frequency = emotions.count('Distress')
satisfied_frequency = emotions.count('Satisfied')
dissatisfied_frequency = emotions.count('Dissatisfied')
neutral_frequency = emotions.count('Neutral')
amazed_frequency = emotions.count('Amazed')


frequency_all = [upset_frequency, annoyed_frequency, distress_frequency, satisfied_frequency, dissatisfied_frequency, neutral_frequency, amazed_frequency]

plt.bar(emotions_names, frequency_all)
plt.savefig('1.jpg')

# Emotions with reference to frame number

emotion_occurunce_frameno = []
for i in emotions:
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
    else:
        print(i)
        
plt.scatter(frame_nox, emotion_occurunce_frameno)
plt.savefig('2.jpg')

# amaze occurence at which frame, with cumulative plot
fig, axes = plt.subplots(7, 1)

upset_count = 1
annoyed_count = 1
distress_count = 1
satisfied_count = 1
dissatisfied_count = 1
neutral_count = 1


c_upset = []
c_annoyed = []
c_distress = []
c_satisfied = []
c_dissatisfied = []
c_neutral = []

for i in emotions:
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
    
    if i == 'Neutral':
        neutral_count += 1
        c_neutral.append(neutral_count)
    else:
        c_neutral.append(neutral_count)

#change in emotions
l = 0
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


axes[0].plot(frame_nox, c_upset, 'g')
axes[1].plot(frame_nox, c_annoyed, 'm')
axes[2].plot(frame_nox, c_distress, 'r')
axes[3].plot(frame_nox, c_dissatisfied, 'y')
axes[3].plot(frame_nox, c_neutral, 'l')

plt.savefig('3.jpg')


# I need to write a code that detects if there is no change for 300 frames to notify the user
