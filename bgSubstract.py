import numpy as np
import cv2
import sys
from utils import smooth, sm2d

bgPts = np.load('./data/points/video5_points.npy')

print bgPts.shape
#bg = bg[:-23,:]

ks = 9
for i in range(bgPts.shape[1]):
    bgPts[:,i] = sm2d(bgPts[:,i], ks)

seconds_start = 0
seconds_end = -1


cap = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball5.mov')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

seconds_start = 0

codec =  cv2.VideoWriter_fourcc(*'XVID')
bg = cv2.imread('output/background5.jpg')
frame_start = int(frameFPS * seconds_start)
out_filename = 'output/substracted.avi'

out = cv2.VideoWriter(out_filename, codec, 24, (frame_height*3, frame_width))


paused = False

bgstart = bgPts[frame_start]
prevPts = bgPts[0]
ret, firstFrame = cap.read()
frame_index = 1
frame_count = 1

print frame_height, frame_width
while(ret and frame_index < bgPts.shape[0]):
    if paused == False:
        # print frame_index, cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        res, frame = cap.read()
        # Get the anchor points for the current frame
        pts = bgPts[frame_index]
        # Compute the homography induced by current frame and the  background
        H = cv2.findHomography(np.array(bgstart, dtype='float'), np.array(pts, dtype='float'))[0]
        # Warp the current frame with the homoraphy
        warpedFrame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

        # Warp the background, substract it from the framse
        bgWarped = cv2.warpPerspective(bg, H, (frame.shape[1], frame.shape[0]))
        substracted = np.float32(frame) - bgWarped

        frame = np.concatenate((frame, substracted, bgWarped), axis = 0)
        cv2.imshow('img', cv2.convertScaleAbs(frame))
        resized_image = cv2.resize(cv2.convertScaleAbs(substracted), (frame_height*3, frame_width))
        out.write(resized_image)

        prevPts = pts
        frame_index += 1
        frame_count += 1
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        paused = not paused
    elif k & 0xFF == ord('q'):
        break

print "Done"
out.release()
cap.release()
