# the following code assumes low frame numbers (~20,000); 
# --> simplicity > efficiency (speed); 
# may be optimized for higher frame numbers

#CV and Math
import cv2                      #openCV; Powerful computer vision
import trackpy as tp            #trackPy
#import custom_imvideo as imv   #may require unpublished custom_imvideo for batch and GPU acceleration (version dev 0.0.3 and up)
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

#set styles
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

def main():
    '''Main testing function'''
    processedVideo, framesPath = track.prepareVideo('video\concave1.mp4', savePic=True, skip=5)
    track.trackProcessedVideo(str(processedVideo))
    return

class track:

    def prepareVideo(video, gray=False, savePic=False, sample=1000, skip=1):
        '''Function reads video into grayscale frames and reduce excessive frames for tracking and analysis.
        Imput:
                video               (string)            video file location
                gray=False          (boolean)           grayscale video
                savePic=False       (boolean)           save processed frames
                sample=1000         (int)               number of frames to be processed
                skip=1              (int)               skip a number of frames in between two processed prames
        
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
        savePath.mkdir(exist_ok=True)       #make directory frames/ under the current parent directory for savePic

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #total video frames
        fps = float(cap.get(cv2.CAP_PROP_FPS))                  #video fps
        frames = [[] for _ in range(length)]                    # $$ Faster implementation? $$ empty 'array' for frame storage
        print('Total # frames: '+str(length))
        print('FPS: '+str(fps))

        smaller = min([length, sample])                         #compare video length and sample length
        for i in tqdm(range(int(smaller))):
            if i%skip == 0:                                     #skip a number of frames in between two processed prames
                if not gray:                                    #if not in grayscale compute grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # $$ FIX $$        

                cv2.imwrite(os.path.join(savePath, str(i) + '.png'), frame)             #save processed frames and make video
            ret, frame = cap.read()                             #retrieve next frame

        warnings.warn('Currently requires dedicated RAM >= 5GB (system RAM ~>= 8GB); close other programes if possible... This will be fixed soon...')
        if input('y to accept the warning and continue; any key to exit...') == 'y':
            pass 
        else:
            raise Exception('Program interrupted by user...')

        processedVideoTime = time.time()
        processedVideoTitle = str(processedVideoTime)+'grayscale_processed.avi'
        imv.local.timelapse(str(processedVideoTitle), 15,  str(savePath))       #construct video from processed frames $$ FIX $$ grayscale in cv2

        if not savePic:                                         
            try:                                                #trash the picture directory 
                shutil.rmtree(savePath)
                return processedVideoTitle, None

            except OSError as e:
                print(f'Error: {savePath} : {e.strerror}')

        return processedVideoTitle, savePath

    def trackProcessedVideo(video):
        '''Function performs cell tracking and position logging
        Imput:
                video               (string)            video file location
        Output:
                dataFile
        '''
        cap = cv2.VideoCapture(video)       #capture video
        ret, frame = cap.read()             #read frame information
        if not ret:
            raise Exception('Unable to read video... Video file broken? Check path/type...')

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #total video frames
        fps = float(cap.get(cv2.CAP_PROP_FPS))                  #video fps
        frames = [[] for _ in range(length)]                    # $$ Faster implementation? $$ empty 'array' for frame storage
        print('Total # frames: '+str(length))
        print('FPS: '+str(fps))

        print('Inspect feature selection...')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # $$ FIX $$ correct grayscale parameter in cv2, remove this line
        track.identify(frame)                                   # identify cells of interest present in frame 0

        user = input('y to continue; any key to quit program...')           #continue if satisfied with cell selection
        if user != 'y':
            raise Exception('Program interrupted by user...')               #interrupt if not

        for i in tqdm(range(length)):
            ret, frame = cap.read()             #go through each frame

        return
    
    def trackCameraFeed(gray=False, savePic=False, skip=1):
        '''
        Compute particle trajectory and analysis on the fly!
        Function reads live video into grayscale frames and performs tracking and data logging.
        Imput:
                gray=False          (boolean)           grayscale video
                savePic=False       (boolean)           save processed frames
                skip=1              (int)               skip a number of frames in between two processed prames
        Notes:
                Live video structure only... Will update when trak and analysis are done...
        '''
        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Monitor', grayscale)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        return

    def identify(frame):
        '''Dummy Testing Function'''
        f = tp.locate(frame, 51, invert=True)
        f.head()
        tp.annotate(f, frame)

        return 

class analysis:

    def meanPosition():

        return

if __name__ == "__main__":
    main()
