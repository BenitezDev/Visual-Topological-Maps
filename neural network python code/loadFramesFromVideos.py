import random
import numpy as np
import cv2
from PIL import Image
import pickle
from Network import Normalize

# Funciones de carga de datos
output_dimension = (128,128)

video_route = "./landmarks_videos/landmarks_{:}.avi"
video = cv2.VideoCapture(video_route)

video_route_test = "./landmarks_videos/final_test_video.mp4"
video_test = cv2.VideoCapture(video_route_test)

# se definen las funciones de carga de frames del conjunto del testeo
def getFrameNumber():
    return int(video_test.get(cv2.CAP_PROP_FRAME_COUNT))
    
def getFrames(number):
    frames=[]
    for i in range(number):
        ret, frame = video_test.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(output_dimension, Image.ANTIALIAS)
        frame = np.asarray( frame, dtype="float32" )
        frame = Normalize(frame)
        frames.append(frame)
    return frames
        
# se definen las funciones de carga de grames del conjunto del entrenamiento y development
def generateRandomNumbers(end, number):
    res = []
    for i in range(number):
        num = random.randint(0, end-1)
        while num in res:
            num = random.randint(0, end-1)
        res.append(num)
    res.sort()
    
    return res

def getImagesFromTest(rands):
    frames =[]
    video_route = "./landmarks_videos/final_test_video.mp4"
    video = cv2.VideoCapture(video_route)    
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        if i in rands:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(output_dimension, Image.ANTIALIAS)
            frame = np.asarray( frame, dtype="float32" )
            frame = Normalize(frame)            
            frames.append(frame)
    return frames

def developmentTest(number):
    gt = pickle.load(open("clasificacion_frames.sav", 'rb'))
    labels = []
    rands = generateRandomNumbers(len(gt), number)
    print(rands)
    dev = getImagesFromTest(rands)
    for i in rands:
        labels.append(int(gt[i]))
    return dev, labels

    
