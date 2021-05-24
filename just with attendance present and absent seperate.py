# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.


# 1. load the training data (numpy arrays of all the persons)
# x- values are stored in the numpy arrays
# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np
import os
from datetime import datetime
import time


########## KNN CODE ############
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1 - v2) ** 2).sum())


def markAttendence(name):
    with open('present.csv', 'r+') as f:
        total_student_in_class = f.readline()
        print(total_student_in_class)
        nameList = []
        absstuds = []

        for line in total_student_in_class:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\nthe present students are : \n{name},{dtString}')


def maarkattndnce(namees):
    with open('absent.csv', 'r+') as f:
        absstuds = []

        for nam in total_student_in_class:
            if nam not in class_total_present:
                entry = nam.split(',')
                absstuds.append(entry[0])
        if namees not in absstuds:
            f.writelines(f'\nabsent students are : \n{absstuds}')

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


################################


# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = "C:/Users/Samarth/Desktop/knn/data/"

face_data = []
number = []
labels = []

class_id = 0  # Labels for the given file
names = {}  # Mapping btw id - name
phone_number = {}

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1

        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Testing
attn = []
appn = []
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if (len(faces) == 0):
        continue

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Predicted Label (out)
        out = knn(trainset, face_section.flatten())

        # Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if pred_name not in attn:
            attn.append(pred_name)
        else:
            continue
        markAttendence(pred_name)

    cv2.imshow("Faces", frame)

    path = "C:/Users/Samarth/Desktop/knn/data/"
    images = []  # LIST CONTAINING ALL THE IMAGES
    className = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])

    total_student_in_class = list(className)  ###the toatl students in this class
    print(total_student_in_class)

    class_total_present = list(attn)
    #print(attn)

    res_list = []
    for i in total_student_in_class:
        if i not in class_total_present:
            res_list.append(i)
    time.sleep(10)
    print(res_list)

    maarkattndnce(i)

   # ai = tuple(total_student_in_class)               #name of all the students as a tuple
   #print(ai)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()