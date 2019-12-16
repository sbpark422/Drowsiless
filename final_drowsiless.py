import cv2, dlib
import numpy as np
import keras
from imutils import face_utils
from keras.models import load_model,Sequential
from keras.layers import Dense
from IPython import display
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils.generic_utils import CustomObjectScope
from keras.applications import mobilenet
from keras.preprocessing import image

import sys
import argparse

#Regarding Alarm
import pygame
#from playsound import playsound
import time

#IMG_SIZE = (32, 32)
IMG_SIZE = (34, 26)
MOUTH_SIZE = (224, 224)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Loading Model
model = load_model('models/final_model.h5')
model.summary()

with CustomObjectScope({'relu6': keras.layers.ReLU(6.)}):
    mouth_model = load_model('models/mobileTrain_100.h5', custom_objects ={'relu6': keras.layers.ReLU(6.)})
mouth_model.summary()

pose_model = load_model('models/model.h5')

#Counting Frames && Thresholds
EYE_AR_CONSEC_FRAMES = 6
YAWN_FRAMES = 5
DTR_FRAMES = 5
COUNTER = 0
COUNTER_YAWN = 0
COUNTER_DTR = 0
ALARM_ON = False
ALARM_ON_Yawn = False
ALARM_ON_Dtr = False
which = 0
a_played = False
t_played = False
w_played = False


#std = StandardScaler()

pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")
window = pygame.mixer.Sound("window.wav")
tired = pygame.mixer.Sound("tired.wav")
front = pygame.mixer.Sound("front.wav")
drowsy = pygame.mixer.Sound("drowsy.wav")

##################

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments

# On versions of L4T previous to L4T 28.1, flip-method=2
# Use the Jetson onboard camera
def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    return cv2.VideoCapture(device_number)
  
##################
  
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

def crop_mouth_one(img, mouth_points):
  x1, y1 = np.amin(mouth_points, axis=0)
  x2, y2 = np.amax(mouth_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  mouth_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  mouth_img = gray[mouth_rect[1]:mouth_rect[3], mouth_rect[0]:mouth_rect[2]]

  return mouth_img, mouth_rect

def crop_mouth_two(img, mouth_points):
  x1, y1 = np.amin(mouth_points, axis=0)
  x2, y2 = np.amax(mouth_points, axis=0)

  pad = 10

  mouth_rect = np.rint([x1-pad, y1-pad, x2+pad, y2+pad]).astype(np.int)

  mouth_img = gray[mouth_rect[1]:mouth_rect[3], mouth_rect[0]:mouth_rect[2]]

  return mouth_img, mouth_rect

def process_image_Mobilenet(img): 
  #img = image.load_img(target_size=(224, 224)) 
  img_array = image.img_to_array(img) 
  #img_array = np.expand_dims(img_array, axis=0) 
  img_array = np.empty(shape=(1,224,224,3))
  #img_array = np.expand_dims(img_array, axis=0) 
  processedImg = mobilenet.preprocess_input(img_array) 
  #processedImg = image[:,:,::-1]
  #processedImg = cv2.cvtColor(processedImg, cv2.COLOR_
  return processedImg

# main
#cap = cv2.VideoCapture('videos/2.mp4')

if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:",arguments.video_device)
    if arguments.video_device==0:
      video_capture=open_onboard_camera()
    else:
      video_capture=open_camera_device(arguments.video_device)
    prevTime = 0

    while video_capture.isOpened():
        ret, img_ori = video_capture.read()

        if not ret:
            break
    
        img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
        
        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        faces = detector(gray)

        curTime = time.time()
        sec = curTime - prevTime 
        prevTime = curTime
        fps = 1/(sec)
        print("Time {0} ".format(sec))
        print("Estimated fps {0} ".format(fps))
        str = "FPS : %0.1f" % fps
        #cv2.putText(img, str, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #face_points = detect_face_points(img)      
    
        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            #for roll,pitch,yaw
            features = []
            for i in range(68):
                for j in range(i+1, 68):
                    features.append(np.linalg.norm(shapes[i]-shapes[j]))

            features = np.array(features).reshape(1, -1)

            y_pred = pose_model.predict(features)

            roll_pred, pitch_pred, yaw_pred = y_pred[0]
            roll_pred = roll_pred / 100
            pitch_pred = pitch_pred / 100
            yaw_pred = yaw_pred / 100

            cv2.putText(img, ' Roll: {:.2f}'.format(roll_pred), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, ' Pitch: {:.2f}'.format(pitch_pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, ' Yaw: {:.2f}'.format(yaw_pred), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
            
            #mouth_img, mouth_rect = crop_mouth_one(gray, mouth_points=shapes[48:67])            
            mouth_img, mouth_rect = crop_mouth_two(gray, mouth_points=shapes[48:67]) 

            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)
            
            mouth_img = cv2.resize(mouth_img, dsize=MOUTH_SIZE)         

            cv2.imshow('l', eye_img_l)
            cv2.imshow('r', eye_img_r)

            cv2.imshow('m',mouth_img)
            
            eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

            #mouth_input = mouth_img.copy().reshape((1, MOUTH_SIZE[1], MOUTH_SIZE[0], 1)).astype(np.float32) / 255.
            mouth_img = process_image_Mobilenet(mouth_img)

            pred_l = model.predict(eye_input_l)
            pred_r = model.predict(eye_input_r)

            pred_m = mouth_model.predict(mouth_img)
            
            # visualize
            #state_l = 'O %.1f' if pred_l > 0.1 else '- %.2f'
            #state_r = 'O %.1f' if pred_r > 0.1 else '- %.2f'

            state_l = 'open' if pred_l > 0.1 else 'closed'
            state_r = 'open' if pred_r > 0.1 else 'closed'

            #state_m = '%.1f' if pred_m > 0.1 else '- %.2f'
            #state_m = 'yawn' if pred_m >= 1.0 else 'normal'
            state_m = '%.4f'

            eye_color = (255,0,0)
            eye_color = (255,0,0) if pred_l > 0.1 else (0,0,255)
            mouth_color = (0,255,0)
            mouth_color = (0,0,255) if pred_m >= 1.0 else (0,255,0)
                    
            state_l = state_l % pred_l
            state_r = state_r % pred_r

            state_m = state_m % pred_m
                            
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=eye_color, thickness=2)
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=eye_color, thickness=2)

            cv2.rectangle(img, pt1=tuple(mouth_rect[0:2]), pt2=tuple(mouth_rect[2:4]), color=mouth_color, thickness=2)
                        
            cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
            cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
            
            cv2.putText(img, state_m, tuple(mouth_rect[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)


            if pred_l < 0.1 and pred_r < 0.1:
                cv2.putText(img, "Closed Eyes", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                                ALARM_ON = True
                        cv2.putText(img, "Drowsiness!!", (230,330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if a_played == False:
                           drowsy.play()
                           alarm.play()
                        a_played = True
            else:
                cv2.putText(img, "Open Eyes", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                COUNTER = 0
                ALARM_ON = False
                alarm.fadeout(3000)
                a_played = False

            if pred_m >= 1.0:
                cv2.putText(img, "Yawning Detected", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                COUNTER_YAWN += 1
                if COUNTER_YAWN >= YAWN_FRAMES:
                   if not ALARM_ON_Yawn:
                      ALARM_ON_Yawn = True 
                   if t_played == False: 
                      if which == 0 : 
                         tired.play()
                         which = 1
                      elif which == 1 :
                         window.play()
                         which = 0 
                      t_played = True
            else :
                COUNTER_YAWN = 0
                ALARM_ON_Yawn = False
                t_played = False

            if pitch_pred < 0.6:
                cv2.putText(img, "Driver Distraction", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                COUNTER_DTR += 1
                if COUNTER_DTR >= DTR_FRAMES:
                   if not ALARM_ON_Dtr:
                      ALARM_ON_Dtr = True
                   if w_played == False:
                      front.play()
                   w_played = True
            else :
               COUNTER_DTR = 0
               ALARM_ON_Dtr = False
               w_played = False


        cv2.imshow('Result', img)
        if cv2.waitKey(1) == ord('q'):
            break
                
    video_capture.release()
    cv2.destroyAllWindows()
