import numpy as np
import cv2
import cv2.cv as cv
from collections import defaultdict
 
cap = cv2.VideoCapture('beachVolleyball3.mov')

 
 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                        qualityLevel = 0.01,
                        minDistance = 5,
                        blockSize = 5 )
 
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
 
while(1):

    mask = np.zeros_like(old_frame)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    ret,frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
     # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
 
     # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
 
    # draw the tracks

    avMovX = 0
    avSumX = 0
    avSumY = 0
    avMovY = 0
    count = 0

    mvs = defaultdict(lambda: defaultdict(int))

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        count = count + 1
        avSumX = avSumX + a - c
        avSumY = avSumY + b - d
        mvs[(int)(a-c)][(int)(b-d)] = mvs[(int)(a-c)][(int)(b-d)] + 1
        

    avMovY = abs((int)(avSumY/count))+ 1
    avMovX = abs((int)(avSumX/count)) + 1

    objects = list()
    objects2 = list()



    #print(str(avMovX)+" "+str(avMovY))

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        if(True or abs(a-c)>abs(avMovX)*2 or abs(b-d)>abs(avMovY)*2):
            #print(mvs[(int)(a-c)][(int)(b-d)])
            if(mvs[(int)(a-c)][(int)(b-d)]<10):
                #print(mvs[(int)(a-c)][(int)(b-d)])
                cv2.line(frame, (a,b),(c,d), color[0].tolist(), 2)

                #cv2.circle(mask,(a,b),20,color[0].tolist(),-1)
            #if(mvs[(int)(a-c)][(int)(b-d)]>4):
                cv2.circle(frame,(a,b),20,color[1].tolist(),-1)
                objects.append((a,b))
                objects2.append((a,b))


        #for (x, y) in objects:
               #x , y = 482.571, 186.582
        #        candidate = 0
        #        for(x1, y1) in objects2:
                    #print(str(x)+" "+str(y)+" vs "+str(x1)+" "+str(y1))
        #            if(x1 != x and y1!=y and abs(y1-y)<1000 and abs(x1-x)<1000):
                        #print("elim")
        #                candidate = candidate + 2

        #        if candidate < 2:
        #            print("passed ! "+str(x) +" "+str(y))
        #            cv2.circle(mask,(x,y),20,color[5].tolist(),-1)




    cv2.imshow("f", mask)

    img = cv2.add(frame,mask)

    old_gray = frame_gray
 

    cv2.imshow('frame',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
 
     # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
 
cv2.destroyAllWindows()
cap.release()