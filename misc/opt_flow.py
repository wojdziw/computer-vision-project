import numpy as np
import cv2
import cv2.cv as cv


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

 
cap = cv2.VideoCapture('beachVolleyball4.mov')

 
 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                        qualityLevel = 0.01,
                        minDistance = 3,
                        blockSize = 7 )
 
 # Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
 # Create some random colors
color = np.random.randint(0,255,(1000,3))
 
 # Take first frame and find corners in it


for i in range(1):
   ret, old_frame = cap.read() 

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

 
 # Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

ret, frame = cap.read()
ret, frame1 = cap.read()
bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


ret, frame2 = cap.read()
col = np.array(frame)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
if ret:
         # res = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) - np.float32(p1))
    old_gray, change = segmentMoving(bg, frame, frame1, frame2, 0.3 * 255, 0.01)
    old_gray *= 255 # we oly get 0s and 1s, so we multiply them to get a b/w image
    old_gray = np.float32(old_gray)
        # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones([10, 10]))
        # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones([2, 2]))
    if change:
        bg = frame1
            # mask the color img with the resulting mask
            # for i in range(3):
            #     col[:, :, i] = col[:, :, i] * res
    #cv2.imshow(windowName, cv2.convertScaleAbs(old_gray))
    old_gray = cv2.convertScaleAbs(old_gray)
 
while(1):

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
        #cv2.imshow("res", cv2.convertScaleAbs(res))

    res = cv2.convertScaleAbs(res)

    frame = np.array(frame1)
    frame1 = np.array(frame2)


    mask = np.zeros_like(old_gray)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #ret,frame = cap.read()

    #frame_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
 
     # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, res, p0, None, **lk_params)
 
     # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
 
    # draw the tracks

    avMovX = 0
    avSumX = 0
    avSumY = 0
    avMovY = 0
    count = 0

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        count = count + 1
        avSumX = avSumX + a - c
        avSumY = avSumY + b - d

    avMovY = abs((int)(avSumY/count) )+ 1
    avMovX = abs((int)(avSumX/count)) + 1

    ball_candidates = list()

    print(str(avMovX)+" "+str(avMovY))
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        if(abs(a-c)>abs(avMovX)*2 or abs(b-d)>abs(avMovY)*2):
            cv2.line(mask, (a,b),(c,d), color[0].tolist(), 2)
            cv2.circle(res,(a,b),5,color[0].tolist(),-1)
            ball_candidates.append((a,b))

    cv2.imshow("f", mask)

    img = cv2.add(res,mask)

    #old_gray = res
 

    cv2.imshow('frame',img)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
 
     # Now update the previous frame and previous points
    old_gray = res.copy()
    p0 = good_new.reshape(-1,1,2)
 
cv2.destroyAllWindows()
cap.release()