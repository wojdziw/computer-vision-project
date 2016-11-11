import numpy as np
import cv2
import cv2.cv as cv
import sys
from utils import *

if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Usage: substractBackground.py <video_number> [visualize - 0 or 1 - default -> 0] [seconds_start, default -> first frame] [seconds_ref, default -> seconds_start]"
    print "[seconds_start] - the offset from the beginning of the video at which the background is projected"
    sys.exit(1)


video_number = int(sys.argv[1])
seconds_start = 0
visualize = False
if len(sys.argv) > 2:
    visualize = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    seconds_start = float(sys.argv[3])
seconds_ref = seconds_start
if len(sys.argv) > 4:
    seconds_ref = float(sys.argv[4])


points_file, source_video_filename, background_filename, _, background_substracted_video_filename = getFilenamesForIndex(video_number)


cap = cv2.VideoCapture(source_video_filename)
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
codec =  cv.CV_FOURCC('F', 'L', 'V', '1')

bg = cv2.imread(background_filename)

print "Output will be written to ", background_substracted_video_filename
out = cv2.VideoWriter(background_substracted_video_filename, codec, fps, (frame_width, frame_height))

frame_start = int(fps * seconds_start)
frame_ref = int(fps * seconds_ref)

print 'Loading points, computing homographies...'
points = np.load(points_file)
points = smoothPointsArray(points)
fbf = getFrameToFrameHomographies(points)
homographies = generateHomographiesFromPoints(points, frame_ref, fbf)

uMinG, uMaxG, vMinG, vMaxG = getMinMaxWidthHeight(frame_width, frame_height, homographies)

paused = False

ret, firstFrame = cap.read()
frame_index = 1
frame_count = 1

print "Processing video... press Q to quit and P to pause"
while(ret and frame_index < homographies.shape[0]):
    if paused == False:
        # print frame_index, cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        res, frame = cap.read()
        # Get the anchor points for the current frame
        # Compute the homography induced by current frame and the  background
        H = np.linalg.inv(np.matrix(homographies[frame_index]))
        t = [uMinG,vMinG]
        Ht = np.matrix(([[1,0,t[0]],[0,1,t[1]],[0,0,1]]))

        # Warp the background, substract it from the framse
        bgWarped = cv2.warpPerspective(bg, H * Ht, (frame.shape[1], frame.shape[0]))
        substracted = np.float32(frame) - bgWarped

        frame = np.concatenate((frame, substracted, bgWarped), axis = 0)

        out.write(cv2.convertScaleAbs(substracted))

        frame_index += 1
        frame_count += 1

        if visualize:
            cv2.imshow('Frame, Substracted, Warped frame', cv2.convertScaleAbs(frame))
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        if visualize:
            paused = not paused
    elif k & 0xFF == ord('q'):
        break

print "Done"
out.release()
cap.release()
