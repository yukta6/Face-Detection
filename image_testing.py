# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:48:28 2021

@author: yukta
"""

import cv2
import os
import numpy as np
import face_recognition as fr

test_img = cv2.imread('C:/Users/yukta/Desktop/Face detection/Test image/test.jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print('face detected:', faces_detected)

# comment these lines when you are running the code from the second time.
faces, faceID = fr.labels_for_training_images('C:/Users/yukta/Desktop/Face detection/Training images')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write('C:/Users/yukta/Desktop/Face detection/traningData.yml')

# uncomment these lines when you are running the code from the second time onwards.
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('C:/Users/yukta/Desktop/Face detection/traningData.yml')

name = {0: 'Manoj'} 

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    
    if (confidence>35):
        continue
    
    fr.put_text(test_img, predicted_name, x, y)
    print('Confidence:', confidence)
    print('Label', label)
    resized_img =cv2.resize(test_img,(500,700))
    cv2.imshow('face detection',resized_img)     
    cv2.waitKey(0)
    cv2.destroyAllWindows