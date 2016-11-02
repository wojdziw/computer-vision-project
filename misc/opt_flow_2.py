import numpy as np
import cv2
import cv2.cv as cv
from collections import defaultdict
 
cap = cv2.VideoCapture('../ballDetection/preprocessed/beachVolleyball3_2x_mo_mask.mov')

 
 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                        qualityLevel = 0.1,
                        minDistance = 5,
                        blockSize = 5 )
 
 # Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (10,10),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
 # Create some random colors
color = np.random.randint(0,255,(1000,3))
 
 # Take first frame and find corners in it
pos1 = np.load('../positionArrays/positions6_redDown.npy')
pos2 = np.load('../positionArrays/positions6_redUp.npy')
pos3 = np.load('../positionArrays/positions6_whiteRight.npy')
pos4 = np.load('../positionArrays/positions6_whiteLeft.npy')

ballLastPos = (0,0)
ballCandidatesPrev = list()

for i in range(1):
   ret, old_frame = cap.read() 

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

 
 # Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
ite = 0
while(1):
    mask = np.zeros_like(old_frame)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # frame_gray = cv2.dilate(frame_gray, np.ones((1, 1)))
    #frame_gray = cv2.erode(frame_gray, np.ones((2, 2)))
    #frame_gray = cv2.dilate(frame_gray, np.ones((10, 10)))

    frdisp = frame_gray.copy()
    cv2.imshow("before",old_gray)
    #old_gray = cv2.dilate(old_gray, np.ones((12, 12))) 
    cv2.imshow("after",old_gray)

    if(p0 != None):
        
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
        count = 1

        mvs = defaultdict(lambda: defaultdict(int))
        angles = list()
        for i in range(360):
            angles.append(0)

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()


            if(a-c == 0):
                angle = 90/5
            else :
                angle = (int)(np.rad2deg(np.arctan((b-d)/(a-c)))/5)

            angles[angle] = angles[angle] + 1

            #print(str(angle))

            if( abs(a-c)<20 and abs(b-d)<20):
                count = count + 1
                avSumX = avSumX + a - c
                avSumY = avSumY + b - d
                mvs[(int)(a-c)][(int)(b-d)] = mvs[(int)(a-c)][(int)(b-d)] + 1               
            

        avMovY = abs((int)(avSumY/count))+ 1
        avMovX = abs((int)(avSumX/count)) + 1

        objects = list()
        objects2 = list()


        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #cv2.circle(frdisp,(a,b),5,color[0].tolist(),-1)
            if( True and abs(a-c)>2*abs(avMovX) or abs(b-d)>2*abs(avMovY)):
                if(a-c == 0):
                    angle = 90 / 5
                else :
                    angle = (int)(np.rad2deg(np.arctan((b-d)/(a-c)))/5)
                #print(mvs[(int)(a-c)][(int)(b-d)])
                if(True and angles[angle] < 10 and angles[angle]>0 and abs(a-c)<20 and abs(b-d)<20 and (abs(a-c)>1 or abs(b-d)>1)):
                    #print(mvs[(int)(a-c)][(int)(b-d)])
                    #cv2.line(frame, (a,b),(c,d), color[i].tolist(), 2)
                    #cv2.circle(frame,(a,b),3,color[i].tolist(),-1)
                    #cv2.circle(mask,(a,b),10,color[0].tolist(),-1)
                #if(mvs[(int)(a-c)][(int)(b-d)]>4):
                    #cv2.circle(frame,(a,b),20,color[1].tolist(),-1)
                    objects.append(((int)(a),(int)(b)))
                #objects2.append((a,b))



            p1pos = (pos1[ite,1],pos1[ite,0])
            p2pos = (pos2[ite,1],pos2[ite,0])
            p3pos = (pos3[ite,1],pos3[ite,0])
            p4pos = (pos4[ite,1],pos4[ite,0])

            cv2.circle(frdisp,p1pos,40,color[20].tolist(),-1)
            cv2.circle(frdisp,p2pos,40,color[20].tolist(),-1)
            cv2.circle(frdisp,p3pos,40,color[20].tolist(),-1)
            cv2.circle(frdisp,p4pos,40,color[20].tolist(),-1)


        ballCandidates = list()

        for (x,y) in objects:
            candidate = 0
            for(x1,y1) in objects:
                    if abs(x1-x)<40 and abs(y1-y)<40:
                        candidate = candidate+1
            if candidate>2:
                if (abs(p1pos[0]-x)>50 or abs(p1pos[1]-y)>50) and (abs(p2pos[0]-x)>50 or abs(p2pos[1]-y)>50 ) and (abs(p3pos[0]-x)>50 or abs(p3pos[1]-y)>50 ) and (abs(p4pos[0]-x)>50 or abs(p4pos[1]-y)>50):
                    ballCandidates.append((x,y))
                    cv2.circle(frame,(x,y),20,color[10].tolist(),-1)
                    


        if(len(ballCandidates)>0):
            ballCandidatesPrev = ballCandidates


        print(ballCandidates)            
        p0 = good_new.reshape(-1,1,2)

        cv2.imshow("f", mask)

        img = cv2.add(frame,mask)
        cv2.imshow("after",frame_gray)
        cv2.imshow('frame',img)

    ite = ite +1
    old_gray = frame_gray
    cv2.waitKey(0)
      
     
         # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

 
cv2.destroyAllWindows()
cap.release()