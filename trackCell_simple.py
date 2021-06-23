# the following code assumes low frame numbers (~20,000); 
# --> simplicity > efficiency (speed); 
# may be optimized for higher frame numbers

#import libraries
    #CV and Math
import cv2      #openCV; Powerful computer vision
import trackpy as tp        #trackPy
import imvideo as imv       #imvideo; quick video construction
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

    #Utilities
import datetime
from tqdm import tqdm       #tqdm; progress bar
import os.path
import glob
from pathlib import Path
import shutil

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

def main():

    track.trackCell('video\concave1.mp4', skip=5)

    return

class track:
    def trackCell(video, gray=False, savePic=False, sample=1000, skip=1):
        '''Function reads video into gray frames and performs tracking and data logging
        Imput:
                video
                gray=False
                savePic=False
                sample=1000
                skip=1
        Output:
                datafile
        '''
        cap = cv2.VideoCapture(video)       #capture video
        ret, frame = cap.read()             #read frame information
        if not ret:
            raise Exception('Unable to read video... Video file broken? Check path/type...')

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #total number of frames
        fps = float(cap.get(cv2.CAP_PROP_FPS))                  #video fps
        frames = [[] for _ in range(length)]                    # $$ Faster implementation? $$ empty 'array' for frame storage
        print('Total # frames: '+str(length))
        print('FPS: '+str(fps))
        print('Inspect feature selection...')
        track.identify(frame)
        user = input('y to continue; any key to quit program...')
        if user != 'y':
            raise Exception('Program interrupted by user...')

        smaller = min([length, sample])
        for i in tqdm(range(int(smaller))):
            if i%skip == 0:
                if not gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #cells = position(frame)
                    #analysis.log(cells)
                
                else:
                    #cells = position(frame)
                    #analysis.log(cells)
                    
                    pass

            ret, frame = cap.read()

        return
    
    def identify(frame):
        '''Dummy Testing'''
        f = tp.locate(frame, 29, invert=True)
        f.head()
        tp.annotate(f, frame)

        return 

class analysis:
    def meanPosition():

        return

if __name__ == "__main__":
    main()
