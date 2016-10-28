import cv2
# import cv2.cv as cv
import numpy as np
import sys

if len(sys.argv) < 2:
    print "Not enough aguments"
    print "Specify input file in command line arguments, please"
    sys.exit(1)

def nothing(_):
    pass

def posTrackbarChanged(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    readAndDisplayFrame()
    pass

openF = np.array([[0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0]])

def readAndDisplayFrame():
    ret, frame = cap.read();
    if ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        global canny
        if canny:
            global morph_close
            global morph_open
            edges = cv2.Canny(frame, canny_lower, canny_upper)
            if morph_close:
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones([10, 3]))
            if morph_open:
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones([3, 4]))
            cv2.imshow(windowName, edges)
        else:
            cv2.imshow(windowName, frame)


# def readAndDisplayFrame():
#     global p1
#     global p2
#     ret, frame = cap.read()
#     if ret:
#         res = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - np.float32(p1))
#         p2 = np.array(p1)
#         p1 = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#         cv2.imshow(windowName, cv2.convertScaleAbs(res))


def onCannyLowerChanged(val):
    global canny_lower
    canny_lower = val
    readAndDisplayFrame();
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

def onCannyUpperChanged(val):
    global canny_upper
    canny_upper = val
    readAndDisplayFrame();
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

def onCannySwitch(val):
    global canny
    canny = bool(val)
    readAndDisplayFrame();
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

def onCloseSwitch(val):
    global morph_close
    morph_close = bool(val)
    readAndDisplayFrame();
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

def onOpenSwitch(val):
    global morph_open
    morph_open = bool(val)
    readAndDisplayFrame();
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

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
cv2.createTrackbar("Frame", windowName, 0, frame_count, posTrackbarChanged)

cv2.createTrackbar("Canny lower", windowName, 0, 1000, onCannyLowerChanged)
cv2.createTrackbar("Canny upper", windowName, 0, 1000, onCannyUpperChanged)
cv2.createTrackbar("Canny switch", windowName, 0, 1, onCannySwitch)
cv2.createTrackbar("Close switch", windowName, 0, 1, onCloseSwitch)
cv2.createTrackbar("Open switch", windowName, 0, 1, onOpenSwitch)



while(True):
    if not paused:
        readAndDisplayFrame()
        cv2.setTrackbarPos("Frame", windowName, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    k = cv2.waitKey(1000 / fps)
    if k & 0xFF == ord('q'):
        break
    elif k & 0xFF == ord('p'):
        paused = not paused

cap.release()
