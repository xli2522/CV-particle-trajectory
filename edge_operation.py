import cv2
import numpy as np
import datetime
time = datetime.datetime.now()
import os.path
import glob
from pathlib import Path
import shutil

"""
    Created in June 2019 by Xiyuan Li
        Log: 
        Beta Version 0.1
        Modified on 19 June 2019 by Xiyuan Li'''
"""


# video-image saver saves input video as modified pictures and labels them for image processing

# ********remember to change these two addresses*********** and your video file name
savePathi = '/your/address/PycharmProjects/objectCapture/isave/'
savePatho = '/your/address/PycharmProjects/objectCapture/osave/'
videoName = 'edge_detected' + str(time) + '.mp4'

class viSaver:

    def viSave(self, video):
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        imname = 0
        o = Path('osave/')
        o.mkdir(exist_ok=True)
        i = Path('isave/')
        i.mkdir(exist_ok=True)
        while ret:
            original = frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            canny = cv2.Canny(frame, 100, 150)

            ret, inverted = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            cv2.putText(original,str(imname),(0, 17),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2,cv2.LINE_AA)
            cv2.putText(inverted,str(imname),(0, 17),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2,cv2.LINE_AA)

            # completeNameO = os.path.join(savePath, 'O'+ str(imname) + '.png')
            # completeNameI = os.path.join(savePath, 'I' + str(imname) + '.png')

            cv2.imwrite(os.path.join(savePatho + 'O' + str(imname) + '.png'), original)
            cv2.imwrite(os.path.join(savePathi + 'I' + str(imname) + '.png'), inverted)
            ret, frame = cap.read()
            imname = imname + 1
            print(imname)

        print("inverting process finished.\n Next Step: video reconstruction.")
        cap.release()
        cv2.destroyAllWindows()


class viProcessor:

    def processor(self):

        savePathi = '/your/address/PycharmProjects/objectCapture/isave'
        savePatho = '/your/address/PycharmProjects/objectCapture/osave'

        imageFolderi = savePathi
        imageFoldero = savePatho
        # imagesi = [img for img in os.listdir(imageFolderi)]
        # imageso = [img for img in os.listdir(imageFoldero)]

        imagesi = []
        imageso = []
        filenamei = [img for img in glob.glob("/your/address/PycharmProjects/objectCapture/isave/*.png")]
        num = len(filenamei)

        print(num)

        '''Sorted name list for inverted images'''
        sortnamei = []
        i = 0
        while i < num:
            item = "/your/address/PycharmProjects/objectCapture/isave/I" + str(i) + ".png"

            sortnamei.append(item)
            i = i + 1

        for image in sortnamei:
            img = cv2.imread(str(image))
            if img is not None:
                imagesi.append(img)
        #print(imagesi)

        '''Sorted name list for original images'''
        sortnameo = []
        i = 0
        while i < num:
            item = "/your/address/PycharmProjects/objectCapture/osave/O" + str(i) + ".png"

            sortnameo.append(item)
            i = i + 1
        #print(sortnameo)
        for image in sortnameo:
            # print(image)
            imgo = cv2.imread(str(image))
            # print(image)
            if imgo is not None:
                imageso.append(imgo)
        #print(imageso)
        frame_spec = imagesi[0]
        height, width, layers = frame_spec.shape

        print("images saved to lists. \n Now constructing video.")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(videoName, fourcc, 25, (width, height))  # note: second per frame

        i = 0
        while i < len(imageso):
            imageo = imageso[i]
            imagei = imagesi[i]
            result = cv2.addWeighted(imageo, 0.5, imagei, 0.5, 0)
            print('frame' + str(i))
            video.write(result)
            i = i + 1
            cv2.imshow('edge_detected', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        video.release()

class lkof_motion:
    def motion(self):
        cap = cv2.VideoCapture(str(videoName))
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        ##Imwrite
        video_name = 'lkof_detected'+ str(time) + '.mp4'

        height, width, layers = old_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name,fourcc, 25, (width, height))


        while cap.isOpened():
            ret,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)


            cv2.imshow('frame',img)


            video.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)


        cv2.destroyAllWindows()
        video.release()
        cap.release()


def main():

    save = viSaver()
    vp = viProcessor()
    lkof = lkof_motion()

    #save.viSave('wave6.mp4')
    #vp.processor()
    lkof.motion()

    trash_diri = '/your/address/PycharmProjects/objectCapture/osave/'
    trash_diro = '/your/address/PycharmProjects/objectCapture/isave/'

    try:
        shutil.rmtree(trash_diri)
        shutil.rmtree(trash_diro)
    except OSError as e:
        print(f'Error: {trash_diri} : {e.strerror}')
        print(f'Error: {trash_diro} : {e.strerror}')


if __name__ == "__main__":
    main()

print("Completed.")
