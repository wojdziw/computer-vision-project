import numpy as np
import cv2
import sys
from utils import smooth, sm2d

bg = np.load('./data/points/video5_points.npy')

print bg.shape
#bg = bg[:-23,:]

ks = 9
for i in range(bg.shape[1]):
    bg[:,i] = sm2d(bg[:,i], ks)

seconds_start = 0
seconds_end = -1


cap = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball5.mov')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(cap.get(cv2.CAP_PROP_FPS))
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


frame_start = int(frameFPS * seconds_start)
frames_end = frameCount - 1
if seconds_end > 0:
    frames_end = max(frameCount -1, int(seconds_end * frameFPS))


paused = False
userQuit = False

cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * seconds_start)
prevPts = 0
firstPts = bg[frame_start]
ret, firstFrame = cap.read()
avgImg = np.float32(firstFrame)
frame_index = frame_start
frame_count = 1
while(ret and frame_index < frames_end and frame_index < bg.shape[0]):
    if paused == False:
        # print frame_index, cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        # print cap.get(cv.CV_CAP_PROP_POS_MSEC)
        ret, frame = cap.read()

        # Get the anchor points for the current frame
        pts = bg[frame_index]
        # Compute the homography induced by current frame and the first ones
        H = cv2.findHomography(pts, np.array(firstPts, dtype='float'))[0]
        # Warp the current frame with the homoraphy
        warpedFrame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

        # The produced image does not occupy the whole frame. In order not to
        # get darker patches when we average the background away, we want to
        # only use the non-black part(the actual warped image) to compute the
        # average
        sums = np.float32(np.sum(warpedFrame, axis=2) == 0)
        black = np.zeros([frame_height, frame_width, 3])
        black[:,:,0] = np.array(sums)
        black[:,:,1] = np.array(sums)
        black[:,:,2] = np.array(sums)

        # Backgound averaging iteration
        alpha = 1 / float(frame_count)
        prevAvg = np.array(avgImg)
        cv2.accumulateWeighted(warpedFrame, avgImg, alpha)
        # Use the pixels from the previous iteration of averaging for the parts
        # of the frame which are not covered by the warped image
        avgImg = black * prevAvg + (1 - black) * avgImg

        frame = np.concatenate((frame, avgImg, warpedFrame), axis = 0)
        cv2.imshow('warp', cv2.convertScaleAbs(frame))
        prevPts = pts
        frame_index += 1
        frame_count += 1
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        paused = not paused
    elif k & 0xFF == ord('q'):
        userQuit = True
        break

if not userQuit:

    cv2.imwrite('output/background5.jpg', avgImg)
else:
    print "You pressed Q - nothing will be written"
cap.release()
