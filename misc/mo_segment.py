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
    d_b1 = np.uint(d_b > threshold)
    d_11 = np.uint(d_1 > threshold)
    d_21 = np.uint(d_2 > threshold)

    # Generate motion masks
    m_1 = d_b1 * d_11 # logical AND
    m_2 = d_b1 * d_21 # logical AND
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
no_ext = filename[:filename.rfind('.')]
ext = filename[filename.rfind('.'):]
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(-1)

if(cv2.__version__[0] == '3'):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
else: # version 2
    fps = int(cap.get(cv2.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CV_CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CV_CAP_PROP_FOURCC))


out = cv2.VideoWriter(no_ext + "_mo_mask" + ext, codec, fps, (frame_width, frame_height))


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


c1 = 0.27 # diff threshold
c2 = 0.04 # bg change threshold

def onCoef1(val):
    global c1
    c1 = val / 100.0

def onCoef2(val):
    global c2
    c2 = val / 1000.0

def onBlurStr(val):
    global blurSize
    blurSize = max(1, val)

blurSize = 3

cv2.createTrackbar("Coef1", windowName, int(c1 * 100), 100, onCoef1)
cv2.createTrackbar("Coef2", windowName, int(c2 * 1000), 1000, onCoef2)
cv2.createTrackbar("Blur str", windowName, blurSize, 10, onBlurStr)

ret, bg = cap.read()
ret, frame1 = cap.read()
ret, frame2 = cap.read()
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
frame1 = cv2.cvtColor(cv2.blur(frame1, (blurSize,blurSize)), cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(cv2.blur(frame2, (blurSize,blurSize)), cv2.COLOR_BGR2GRAY)
while ret:
    if not paused:
        ret, frame3 = cap.read()
        frame3 = cv2.cvtColor(cv2.blur(frame3,(blurSize,blurSize)), cv2.COLOR_BGR2GRAY)
        print(np.sum(np.abs(np.float32(frame3) - np.float32(frame2))))
        if ret:
            res, change = segmentMoving(bg, frame1, frame2, frame3, c1 * 255, c2)
            res *= 255 # we oly get 0s and 1s, so we multiply them to get a b/w image

            # write to the outFile
            o = np.empty([res.shape[0], res.shape[1], 3])
            o[:, :, 0] = res
            o[:, :, 1] = res
            o[:, :, 2] = res
            out.write(cv2.convertScaleAbs(o))

            # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones([2, 2]))
            # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones([3, 3]))
            # res = cv2.erode(res, np.ones([2,2]))
            if change:
                bg = frame2

            res = np.float32(res)
            img = np.concatenate((frame2, res), axis=0)
            cv2.imshow(windowName, cv2.convertScaleAbs(img))
        frame1 = np.array(frame2)
        frame2 = np.array(frame3)

    k = cv2.waitKey(1000 / fps)
    # k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break
    elif k & 0xFF == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
