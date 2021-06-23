# the following code assumes low frame numbers (~20,000); 
# --> simplicity > efficiency (speed); 
# may be optimized for higher frame numbers

#CV and Math
import cv2                      #openCV; Powerful computer vision
import trackpy as tp            #trackPy
import imvideo as imv           #imvideo; quick video construction
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  

#Utilities
import datetime
from tqdm import tqdm           #tqdm; progress bar
import os.path
import glob
from pathlib import Path
import shutil

#set styles
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

def main():
    '''Main testing function'''
    track.trackCell('video\concave1.mp4', savePic=True, skip=5)

    return

class track:

    def trackCell(video, gray=False, savePic=False, sample=1000, skip=1):
        '''Function reads video into grayscale frames and performs tracking and data logging.
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

        print('Inspect feature selection...')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        track.identify(frame)                                   # identify cells of interest present in frame 0

        user = input('y to continue; any key to quit program...')           #continue if satisfied with cell selection
        #if user != 'y':
            #raise Exception('Program interrupted by user...')               #interrupt if not

        smaller = min([length, sample])                         #compare video length and sample length
        for i in tqdm(range(int(smaller))):
            if i%skip == 0:                                     #skip a number of frames in between two processed prames
                if not gray:                                    #if not in grayscale compute grayscale
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # $$ FIX $$
                    pass
                #cells = position(frame)
                #analysis.log(cells)
                if savePic:                                     #save processed frames and make video
                    cv2.imwrite(os.path.join(savePath, str(i) + '.png'), frame)

            ret, frame = cap.read()                             #retrieve next frame

        if savePic:
            imv.local.timelapse('trackCell_processed.avi', 10,  str(savePath))       #construct video from processed frames

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
