import cv2
# import cv2.cv as cv
import numpy as np
import sys

if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Specify input file in command line arguments, please"
    sys.exit(1)


'''
==============================
'''
def segmentMoving(bg, i_1, i_2, i_3, threshold, threshold_m):

    pixel_count = bg.shape[0] * bg.shape[1]

    d_b = abs(np.float32(i_2) - np.float32(bg))
    d_1 = np.float32(i_2) - np.float32(i_1)
    d_2 = np.float32(i_2) - np.float32(i_3)

    # apply thresholding
    d_b = np.uint(d_b > threshold)
    d_1 = np.uint(d_1 > threshold)
    d_2 = np.uint(d_2 > threshold)

    print np.sum(d_b), ' ', np.sum(d_1), ' ', np.sum(d_2)

    # Generate motion masks
    m_1 = d_b * d_1 # logical AND
    m_2 = d_b * d_2 # logical AND
    m = np.uint((m_1 + m_2) > 0) # logical OR

    nm = np.sum(m)
    nm_i = nm / float(pixel_count)

    # B(x,y) = i_2(x,y) if nm_i < treshold_m and B(x,y) otherwise
    new_bg = False # Should we update the background
    if nm_i < threshold_m:
        new_bg = True

    return m, new_bg
'''
=================================================
'''


filename = sys.argv[1]
cap = cv2.VideoCapture(filename)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canny_lower = 0
canny_upper = 0
canny = False
morph_close = False
morph_open = False
paused = False
p1 = np.zeros([frame_height, frame_width]) # previous frame
p2 = np.zeros([frame_height, frame_width]) # 2 frames back

windowName = filename.split('/')[-1]
cv2.namedWindow(windowName)

# ret, f = cap.read()
# bg = f
ret, frame = cap.read()
ret, frame1 = cap.read()
bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while ret:
    if not paused:
        ret, frame2 = cap.read()
        col = np.array(frame)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if ret:
            # res = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - np.float32(p1))
            res, change = segmentMoving(bg, frame, frame1, frame2, 0.3 * 255, 0.01)
            res *= 255 # we oly get 0s and 1s, so we multiply them to get a b/w image
            res = np.float32(res)
            # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones([10, 10]))
            # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones([2, 2]))
            if change:
                bg = frame1
            # mask the color img with the resulting mask
            # for i in range(3):
            #     col[:, :, i] = col[:, :, i] * res
            cv2.imshow(windowName, cv2.convertScaleAbs(res))
        frame = np.array(frame1)
        frame1 = np.array(frame2)

    k = cv2.waitKey(1000 / fps)
    if k & 0xFF == ord('q'):
        break
    elif k & 0xFF == ord('p'):
        paused = not paused
