import numpy as np
import cv2
import cv2.cv as cv
import sys
from utils import *

if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Usage: extractBackgound.py <video_number> [visualize - 0 or 1 - default -> 0] [seconds_start, default -> start of video] [seconds_ref, default -> seconds_start] [seconds_end, default -> end of video] "
    sys.exit(1)

# Process command line arguments
video_number = int(sys.argv[1])
seconds_start = 0
seconds_ref = 0
seconds_end = -1
visualize = False
if len(sys.argv) > 2:
    visualize = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    seconds_start = float(sys.argv[3])

if len(sys.argv) > 4:
    seconds_ref = float(sys.argv[4])
else:
    seconds_ref = seconds_start

if len(sys.argv) > 5:
    seconds_end = float(sys.argv[5])

# Get the filenames we need
points_file, source_video_filename, background_filename, _, _ = getFilenamesForIndex(video_number)

cap = cv2.VideoCapture(source_video_filename)
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

# We want to know from which to which frame to start extracting and in which frame's
# reference frame(hah) we should project
frame_start = int(fps * seconds_start)
frame_ref = int(fps * seconds_ref)
frames_end = frame_count - 1
if seconds_end > 0:
    frames_end = max(frame_count -1, int(seconds_end * fps))

print 'Loading points, computing homographies...'
points = np.load(points_file)
# Remove noise
points = smoothPointsArray(points)
# Get frame-to-frame homographies(used in the next step)
ftf = getFrameToFrameHomographies(points)
# Get homographies w.r.t the reference frame
homographies = generateHomographiesFromPoints(points, frame_ref, ftf)

# Calculate the dimensions of the resulting image
uMinG, uMaxG, vMinG, vMaxG = getMinMaxWidthHeight(frame_width, frame_height, homographies)
max_width = uMaxG - uMinG
max_height = vMaxG - vMinG

paused = False
userQuit = False

cap.set(cv.CV_CAP_PROP_POS_MSEC, 1000 * seconds_start)
ret, firstFrame = cap.read()

# We will accumulate the background in avgImg
avgImg = np.zeros([max_height, max_width, 3], dtype='float')
avgImg[-vMinG:frame_height - vMinG, -uMinG:frame_width - uMinG] = firstFrame

# px_active_counts[x, y] -> in how many frames has x,y been in the bounds of the
# warped frame
px_active_counts = np.zeros([max_height, max_width])
vis = np.float32(np.sum(avgImg, axis=2) != 0)
px_active_counts += vis

frame_index = frame_start
frame_count = 1

print "Processing video... press Q to quit and P to pause"
while(ret and frame_index < frames_end and frame_index < homographies.shape[0]):
    if paused == False:
        ret, frame = cap.read()
        if ret:
            # Get the homography induced by current frame and the reference one
            H = np.matrix(homographies[frame_index])
            # We center our resulting image around the reference frame, so we
            # need to translate the warped frame
            t = [-uMinG,-vMinG]
            Ht = np.matrix(([[1,0,t[0]],[0,1,t[1]],[0,0,1]]))
            # Warp the current frame with the homoraphy
            warpedFrame = cv2.warpPerspective(frame, Ht * H, (max_width, max_height))

            # The produced image does not occupy the whole frame. In order not to
            # get darker patches when we average the background away, we want to
            # only use the non-black part(the actual warped image) to compute the
            # average
            vis = np.float32(np.sum(warpedFrame, axis=2) != 0)
            inv = np.float32(np.sum(warpedFrame, axis=2) == 0)
            px_active_counts += vis

            weights_mask = np.zeros([max_height, max_width, 3], dtype='float')
            black = np.zeros([max_height, max_width, 3])
            for c in range(3):
                weights_mask[:,:,c] = np.array(px_active_counts, dtype='float')
                black[:,:,c] = np.array(inv)

            # Backgound averaging iteration
            prevAvg = np.array(avgImg)
            # Use the pixels from the previous iteration of averaging for the parts
            # of the frame which are not covered by the warped image
            quot = 1.0 / weights_mask
            quot[quot == np.inf] = 0
            # We weight each pixel in the warped frame by 1 over the number of
            # times it has been a part of a warped frame so far - this is how
            # we calculate the running average
            avgImg = (1 - quot) * avgImg + quot * warpedFrame
            # We only add to the running average for the bounds of the current
            # warped frames
            avgImg = black * prevAvg + (1 - black) * avgImg

            frame_index += 1
            frame_count += 1
            if visualize:
                frame = np.concatenate((avgImg, warpedFrame), axis = 0)
                cv2.imshow('Average image, warped frame', cv2.convertScaleAbs(frame))
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        if visualize:
            paused = not paused
    elif k & 0xFF == ord('q'):
        userQuit = True
        break

if not userQuit:
    print "Writing background to ", background_filename
    cv2.imwrite(background_filename, avgImg)
else:
    print "You pressed Q - nothing will be written"
cap.release()
