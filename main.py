from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

df = pd.read_csv('data.csv')

desc_list = df.to_records(index=False)

def rec_the_man_from_list(face_descriptor):
    min_dist=1
    name='not found'
    for row in desc_list:
        desc = list(row)[5:]
        dist = distance.euclidean(face_descriptor, desc)
        if (dist<min_dist) and (dist<0.57):
            min_dist=dist
            name=list(row)[3]
    return name

font = cv2.FONT_HERSHEY_DUPLEX

# cap = cv2.VideoCapture('20170921_170814.mp4')
cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)


    for k,d in enumerate(rects):


        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        cv2.rectangle(frame,(d.left(), d.bottom()),(d.right(),d.top()),(255,0,0),2)

        shape = sp(gray, d)
        face_descriptor = list(facerec.compute_face_descriptor(frame, shape))
        name = rec_the_man_from_list(face_descriptor)


        cv2.putText(frame, name, (d.left() + 6, d.bottom() +16), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break