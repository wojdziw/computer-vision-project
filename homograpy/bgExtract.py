import numpy as np
import cv2
import cv2.cv as cv
import sys


if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Usage: bgExctract.py <path_to_video_file> [seconds_start, default -> start of video] [seconds_end, default -> end of video]"
    sys.exit(1)

bl = np.load('./data/coords_video1/1st_coords.npy')
br = np.load('./data/coords_video1/2nd_coords.npy')
pl = np.load('./data/coords_video1/3rd_coords.npy')
pr = np.load('./data/coords_video1/4th_coords.npy')

filename = sys.argv[1]
seconds_start = 0
seconds_end = -1

if len(sys.argv) > 2:
    seconds_start = float(sys.argv[2])

if len(sys.argv) > 3:
    seconds_end = float(sys.argv[3])
    if seconds_end <= seconds_start:
        print "Erorr: seconds_end should be > than seconds_start"
        print "Usage: bgExctract.py <path_to_video_file> [seconds_start, default -> start of video] [seconds_end, default -> end of video]"
        sys.exit(1)


cap = cv2.VideoCapture(filename)
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

frame_start = int(fps * seconds_start)
frames_end = frame_count - 1
if seconds_end > 0:
    frames_end = max(frame_count -1, int(seconds_end * fps))


paused = False

cap.set(cv.CV_CAP_PROP_POS_MSEC, 1000 * seconds_start)
prevPts = [bl[frame_start], br[frame_start], pl[frame_start], pr[frame_start]]
firstPts = [bl[frame_start], br[frame_start], pl[frame_start], pr[frame_start]]
ret, firstFrame = cap.read()
avgImg = np.float32(firstFrame)
frame_index = frame_start + 1
frame_count = 1
while(ret and frame_index < frames_end):
    if paused == False:
        # print frame_index, cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        # print cap.get(cv.CV_CAP_PROP_POS_MSEC)
        ret, frame = cap.read()

        # Get the anchor points for the current frame
        pts = [bl[frame_index], br[frame_index], pl[frame_index], pr[frame_index]]
        # Compute the homography induced by current frame and the first ones
        H = cv2.findHomography(np.array(pts, dtype='float'), np.array(firstPts, dtype='float'))[0]
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
        break

noExt = filename[:filename.rfind('.')]
sec_end_str = str(seconds_end)
if seconds_end == -1:
    sec_end_str = 'end'
background_filename = noExt + '_bg_from_' + str(seconds_start) + '_to_' + sec_end_str + '.jpg'
print background_filename
cv2.imwrite(background_filename, avgImg)
cap.release()
