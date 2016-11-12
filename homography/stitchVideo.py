import numpy as np
import cv2
import cv2.cv as cv
import sys
from utils import *

if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Usage: stitchVideo.py <video_number> [visualize - 0 or 1 - default -> 0]"
    sys.exit(1)

# Process command line arguments

video_number = int(sys.argv[1])
visualize = False
if len(sys.argv) > 2:
    visualize = bool(int(sys.argv[2]))

# Get the filenames we need
points_file, source_video_filename, background_filename, stitched_video_filename, _ = getFilenamesForIndex(video_number)
print "Output will be written to ", stitched_video_filename

print 'Loading points, computing homographies...'
points = np.load(points_file)
# Remove noise
points = smoothPointsArray(points)
# Get frame-to-frame homographies(used in the next step)
ftf = getFrameToFrameHomographies(points)
# Get homographies w.r.t the reference frame
homographies = generateHomographiesFromPoints(points, 0, ftf)

background = cv2.imread(background_filename)

filename = sys.argv[1]
cap = cv2.VideoCapture(source_video_filename)
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

# Calculate the dimensions of the resulting video
uMinG, uMaxG, vMinG, vMaxG = getMinMaxWidthHeight(frame_width, frame_height, homographies)
max_width = uMaxG - uMinG
max_height = vMaxG - vMinG

codec =  cv.CV_FOURCC('F', 'L', 'V', '1')
out = cv2.VideoWriter(stitched_video_filename, codec, fps, (max_width, max_height))

paused = False
frame_index = 0
ret, _ = cap.read()
frame_index += 1

print "Processing video... press Q to quit and P to pause"
while(ret and frame_index < homographies.shape[0]):
    if paused == False:
        ret, frame = cap.read()

        H = np.matrix(homographies[frame_index])
        # We center our resulting frames the reference frame, so we
        # need to translate the warped frame
        t = [-uMinG,-vMinG]
        Ht = np.matrix(([[1,0,t[0]],[0,1,t[1]],[0,0,1]]))

        frame_warped = cv2.warpPerspective(frame, Ht * H, (max_width, max_height))

        # Which pixels are withing the warped image?
        new_mask = np.float32(np.sum(frame_warped, axis=2) == 0)
        m = np.empty([new_mask.shape[0], new_mask.shape[1], 3])
        m[:, :, 0] = np.array(new_mask)
        m[:, :, 1] = np.array(new_mask)
        m[:, :, 2] = np.array(new_mask)

        # We dim the parts of the background outwith the bounds of the warped frame
        current = cv2.add(np.float32(background) * np.float32(m) * 0.8,  np.float32(frame_warped))
        out.write(cv2.convertScaleAbs(current))
        frame_index += 1
        if visualize:
            cv2.imshow('Stitched', cv2.convertScaleAbs(current))
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        if visualize:
            paused = not paused
    elif k & 0xFF == ord('q'):
        break

print 'Done'
cap.release()
out.release()
