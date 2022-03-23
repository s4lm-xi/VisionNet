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
plt.show()

# Emotions with reference to frame number

emotion_occurunce_frameno = []
for i in emotions:
    if i == 'Upset
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
        
plt.scatter(frame_nox, emotion_occurunce_frameno)
plt.show()

# amaze occurence at which frame, with cumulative plot
fig, axes = plt.subplots(4, 1)

amazed_count = 1
engaged_count = 1
satisfied_count = 1
unsatisfied_count = 1


c_amazed = []
c_engaged = []
c_satisfied = []
c_unsatisfied = []
for i in emotions:
    if i == 'Amazed':
        amazed_count += 1
        c_amazed.append(amazed_count)
    else:
        c_amazed.append(amazed_count)

    if i == 'Engaged':
        engaged_count += 1
        c_engaged.append(engaged_count)
    else:
        c_engaged.append(engaged_count)

    if i == 'Satisfied':
        satisfied_count += 1
        c_satisfied.append(satisfied_count)
    else:
        c_satisfied.append(satisfied_count)

    if i == 'Unsatisfied':
        unsatisfied_count += 1
        c_unsatisfied.append(unsatisfied_count)
    else:
        c_unsatisfied.append(unsatisfied_count)

#change in emotions
l = 0
for k in range(100, frame_no, 100):
    change = (c_amazed[k] - c_amazed[l])/100
    if change >= 0.1:
        print('increase in amazed emotions between', l, 'and', k)
    l = k

l = 0
for k in range(100, frame_no, 100):
    change = (c_engaged[k] - c_engaged[l])/100
    if change >= 0.1:
        print('increase in engaged emotions between', l, 'and', k)
    l = k

l = 0
for k in range(100, frame_no, 100):
    change = (c_satisfied[k] - c_satisfied[l])/100
    if change >= 0.1:
        print('increase in satisfied emotions between', l, 'and', k)
    l = k

l = 0
for k in range(100, frame_no, 100):
    change = (c_unsatisfied[k] - c_unsatisfied[l])/100
    if change >= 0.1:
        print('increase in unsatisfied emotions between', l, 'and', k)
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


axes[0].plot(frame_nox, c_amazed, 'g')
axes[1].plot(frame_nox, c_engaged, 'm')
axes[2].plot(frame_nox, c_satisfied, 'r')
axes[3].plot(frame_nox, c_unsatisfied, 'y')
plt.show()


# I need to write a code that detects if there is no change for 300 frames to notify the user
