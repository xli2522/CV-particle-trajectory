# the following code assumes low frame numbers (~20,000); 
# --> simplicity > efficiency (speed); 
# may be optimized for higher frame numbers
from __future__ import print_function
from operator import index

#CV and Math
import cv2                      #openCV; Powerful computer vision
import trackpy as tp            #trackPy
#import custom_imvideo as imv   #may require unpublished custom_imvideo for memory management and acceleration (version dev 0.0.3 and up)
import imvideo as imv           #imvideo; quick video construction (published version 0.0.2)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  

#Utilities
import time
from tqdm import tqdm           #tqdm; progress bar
import warnings
import os.path
import glob
from pathlib import Path
import shutil

from trackpy.plots import make_axes

def main():
    '''Main testing function'''
    processedVideo, framesPath = prepareVideo('video\concave1.mp4', savePic=True, skip=3)
    processedVideoSpecs(str(processedVideo))

    return

def prepareVideo(video, gray=False, savePic=False, sample=(700, 5000), skip=1, downsize=0.4):
    '''Function reads video into grayscale frames and reduce excessive frames for tracking and analysis.
    Imput:
            video               (string)            video file location
            gray=False          (boolean)           grayscale video
            savePic=False       (boolean)           save processed frames
            sample=1000         (int)               number of frames to be processed
            skip=1              (int)               skip a number of frames in between two processed prames
            downsize            (float)             fraction of the original frame size
    Notes:
            1. Will improve memory management for longer video construction
            2. Will provide GPU acceleration
            3. Will provide Multi-thread acceleration
            4. imvideo 0.0.3 update will provide batch function and GPU acceleration
    '''
    cap = cv2.VideoCapture(video)       #capture video
    ret, frame = cap.read()             #read frame information
    if not ret:
        raise Exception('Unable to read video... Video file broken? Check path/type...')

    #script_dir = os.path.dirname(__file__)                 #current directory
    savePath = Path('frames/')    
    try:                                                    #trash the picture directory if exists
        shutil.rmtree(savePath)
    except OSError as e:
        pass
    savePath.mkdir(exist_ok=True)                           #make directory frames/ under the current parent directory for savePic

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #total video frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))                  #video fps
    #frames = [[] for _ in range(length)]                   # $$ Faster implementation? $$ empty 'array' for frame storage np.array_like
    height, width, channels = frame.shape                   #frame size
    size = (width, height)
    print('Total # frames: '+str(length))
    print('FPS: '+str(fps))
    print('Size: '+str(size))
    
    size = (int(downsize*width), int(downsize*height))
    smaller = min([length, int(sample[1])])                 #compare video length and sample length
    for i in tqdm(range(int(smaller))):
        if i <= int(sample[0]) or i>= int(sample[1]):
            pass
        else:
            if i%skip == 0:                                 #skip a number of frames in between two processed prames
                if not gray:                                #if not in grayscale compute grayscale
                    frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                     # $$ FIX $$        

                cv2.imwrite(os.path.join(savePath, str(i) + '.png'), frame)             #save processed frames and make video
        ret, frame = cap.read()                             #retrieve next frame

    warnings.warn('Currently requires dedicated RAM >= 5GB (system RAM ~>= 8GB); close other programes if possible... This will be fixed soon...')
    if input('y to accept the warning and continue; any key to exit...') == 'y':
        pass 
    else:
        raise Exception('Program interrupted by user...')

    processedVideoTime = time.time()
    processedVideoTitle = str(processedVideoTime)+'grayscale_processed.avi'
    imv.local.timelapse(str(processedVideoTitle), 15,  str(savePath), inspect=False)       #construct video from processed frames $$ FIX $$ grayscale in cv2

    if not savePic:                                         
        try:                                                #trash the picture directory 
            shutil.rmtree(savePath)
            return processedVideoTitle, None

        except OSError as e:
            print(f'Error: {savePath} : {e.strerror}')

    return processedVideoTitle, savePath

def identify(frame):
    '''Identify features in a frame with trackpy'''

    if isinstance(frame, str):
        frame = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    f = tp.locate(frame, 11, percentile=99, invert=True)
    f.head()
    tp.annotate(f, frame, color='r')
    
    return 

def stealthAnnotate(centroids, image, circle_size=21, color=0, thickness=2):
    '''custom annotation'''
    positionX, positionY = centroids['x'].tolist(), centroids['y'].tolist()

    for i in range(len(positionX)):
        #position approximated to the nearest integer for drawing
        image = cv2.circle(image, (int(positionX[i]), int(positionY[i])), circle_size, color, thickness)
    
    return image

def processedVideoSpecs(video):
        '''Function finds out the specs of the processed video
        Imput:
                video               (string)            video file location           
        '''
        cap = cv2.VideoCapture(video)                           #capture video
        ret, frame = cap.read()                                 #read frame information
        if not ret:
            raise Exception('Unable to read video... Video file broken? Check path/type...')

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #total video frames
        fps = float(cap.get(cv2.CAP_PROP_FPS))                  #video fps

        height, width, channels = frame.shape                   #frame size
        size = (width, height)
        print('Total # frames: '+str(length))
        print('FPS: '+str(fps))
        print('Size: '+str(size))

        print('Inspect feature selection...')
        frame0 = frame.copy()
        frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # $$ FIX $$ correct grayscale parameter in cv2, remove this line
        identify(frame0)                                        # identify cells of interest present in frame 0

        return

if __name__ == "__main__":
    main()
