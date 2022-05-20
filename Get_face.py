import cv2
import math
import numpy as np

from tensorflow.keras import models
from libs.utils.utils import show_bboxes
from libs.utils.align import get_aligned_faces
from libs.mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

font_letter = cv2.FONT_HERSHEY_PLAIN
Emotion_model = load_model(r'C:\Users\Kopila\Downloads\Sajon Clean\Clean\Models\threecalss.h5')
Gender_model = load_model(r'C:\Users\Kopila\Downloads\Sajon Clean\Clean\Models\Genderv2.h5')
age_model = load_model(r'C:\Users\Kopila\Downloads\Sajon Clean\Clean\Models\Age_final.h5')
detector =MTCNN()

cap = cv2.VideoCapture(0)


def emotion_detection(image):
    category = {'angry': 0, 'happy': 1, 'neutral': 2}
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_grayscale = cv2.resize(grayscale, (48,48), interpolation = cv2.INTER_AREA)
    resized = cropped_grayscale.reshape((1, 48, 48, 1)).astype(np.float32) / 255.
    opt = Emotion_model.predict(resized)
    idx = np.argmax(opt)
    emotion_class = list(category.keys())[list(category.values()).index(idx)]
    return emotion_class

def age_detection(image):
    labels =["CHILD",  # index 0
        "YOUTH",      # index 1
        "ADULT",     # index 2 
        "MIDDLEAGE",        # index 3 
        "OLD",         # index 4
        ]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_grayscale = cv2.resize(grayscale, (32,32), interpolation = cv2.INTER_AREA)
    resized = cropped_grayscale.reshape((1, 32, 32, 3)).astype(np.float32) / 255.
    print(resized.shape)
    age = age_model.predict(resized)
    y = np.argmax(age)
    x = labels[y]
    ax = list(age.flatten())[y]
    if  ax > 0.4:

        return x
    
    else:
        return 'UNK'

def gender_detection(image):
    category = {'female': 1, 'male': 0}
    cropped = cv2.resize(image, (96,96)) 
    resized = cropped.reshape((1,96,96,3)).astype(np.float32) / 255.
    opt = Gender_model.predict(resized)
    idx = np.argmax(opt)
    ax = list(opt.flatten())[idx]



    if ax > 0.6:
        gender_class = list(category.keys())[list(category.values()).index(idx)]
    else:
        gender_class = "UNK"
    return gender_class
    

#########################FPS#################
import datetime
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
############################################
while cap.isOpened():
    emotion = ''
    gender = ''
    age = ''
    output = np.zeros((800,1000,3), dtype="uint8")

    ret, frame = cap.read()
    frame  = cv2.flip(frame,1)
    landmarks, bboxs = detector(frame)
    faces = get_aligned_faces(frame, bboxs, landmarks)
    frame = show_bboxes(frame , bboxs , landmarks,'.')
    # print(frame.shape)# 480/640
    try:
  
        emotion = emotion_detection(faces[0])
        gender = gender_detection(faces[0])
        age = age_detection(faces[0])
    except:
        print('No Face Detected')
    
    #################################FPS##########################################################
    total_frames = total_frames + 1
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
    ###############################################################################################
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.putText(frame, emotion, (50,100 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    
    
    
    
    
    
    
    
    cv2.line(output, (333,0), (333,250), (0,255,0), 1)
    cv2.line(output, (666,0), (666  ,250), (0,255,0), 1)
    
    cv2.putText(output,"Emotion",(100,30), font_letter,2, (255,255,51),2)
    cv2.putText(output,"Gender",(400,30), font_letter,2, (255,255,51),2)
    cv2.putText(output,"Age",(750,30), font_letter,2, (255,255,51),2)
    
    # Conditional 
    cv2.putText(output,str(emotion),(100,150), font_letter,2, (0,0,255),2)
    cv2.putText(output,str(gender),(450,150), font_letter,2, (0,0,255),2)
    cv2.putText(output,str(age),(700,150), font_letter,2, (0,0,255),2)
    


    output[320:800, 180:820] = frame
    cv2.imshow("Frame", output)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()