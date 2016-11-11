import numpy as np
import cv2
import cv2.cv as cv
from collections import defaultdict
from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk
import cv2
import numpy as np
import sys


def sig(i):
    if i<0:
        return 0
    else:
        return 1

def display(image):
    root = Tk()

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    rgbImage = np.zeros(image.shape, np.uint8)
    rgbImage[:,:,0] = image[:,:,2]
    rgbImage[:,:,1] = image[:,:,1]
    rgbImage[:,:,2] = image[:,:,0]
    image = rgbImage

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN, cursor="tcross")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    canvas.config(width=imageWidth, height=imageHeight)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    # File = askopenfilename(parent=root, initialdir="~",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.fromarray(image, 'RGB'))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    def go(event):
        #print("event")
        root.destroy()
    def exit():
        #print("event")
        root.destroy()

    canvas.after(600,exit)
    canvas.bind("<Button 1>",go)

    root.mainloop()

def indicateLocation(image):
    root = Tk()

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    rgbImage = np.zeros(image.shape, np.uint8)
    rgbImage[:,:,0] = image[:,:,2]
    rgbImage[:,:,1] = image[:,:,1]
    rgbImage[:,:,2] = image[:,:,0]
    image = rgbImage

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN, cursor="tcross")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    canvas.config(width=imageWidth, height=imageHeight)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    # File = askopenfilename(parent=root, initialdir="~",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.fromarray(image, 'RGB'))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    def printcoords(event):
        global a
        global b
        a = event.x
        b = event.y

        root.destroy()

    canvas.bind("<Button 1>",printcoords)

    root.mainloop()

    return a, b

arguments = sys.argv

if len(sys.argv)<2:
    print("Usage : <Video num>")
    sys.exit(0)
video = sys.argv[1]
videoFile = "../data/videos/beachVolleyball"+video+".mov"
outputFile = "../data/ballPositions/ballPos"+video+".npy"
cap = cv2.VideoCapture(videoFile)

frCount = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
 
 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 5000,
                        qualityLevel = 0.1,
                        minDistance = 5,
                        blockSize = 5 )
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (20,20),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Create some random colors
color = np.random.randint(0,255,(1000,3))


ballPositions = list()

ret, old_frame = cap.read() 

height, width = old_frame.shape[0], old_frame.shape[1]

framePoint = old_frame.copy()

old_frame = cv2.resize(old_frame, (width*1, (int)(height*1)) )

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

pts = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

p0 = np.zeros((5,2), np.float32)
kernelBig = np.ones((3,3))
kernelBig[0,0] = 0
kernelBig[0,2] = 0
kernelBig[2,0] = 0
kernelBig[2,2] = 0

old_frame = cv2.dilate(old_frame, np.ones((10, 10)))
old_frame = cv2.erode(old_frame, np.ones((2, 2)))
old_frame = cv2.dilate(old_frame, np.ones((10, 10)))

old_gray = cv2.dilate(old_gray, np.ones((10, 10)))
old_gray = cv2.erode(old_gray, np.ones((5, 5)))
old_gray = cv2.dilate(old_gray, np.ones((15, 15)))
#print(pts)
for [[u, v]] in pts:
    cv2.circle(framePoint,(u,v),3,color[7].tolist(),-1)

ballx,bally = 0,0
ballPositions.append([ballx, bally, 0, 0])
ctr = 0
for [[u, v]] in pts:
    if abs(u-ballx)<2000 and abs(v-bally)<2000:
        ctr = ctr+1
p0 = np.zeros((ctr,1,2), np.float32)
ctr = 0
for [[u, v]] in pts:
    if abs(u-ballx)<2000 and abs(v-bally)<2000:
        p0[ctr] = [[u, v]]
        ctr = ctr+1

#print(p0)

B1 = 50
G1 = 50
R1 = 50
B2 = 255
G2 = 255
R2 = 255
color1 = np.array([B1, G1, R1])
color2 = np.array([B2, G2, R2])


#pts = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)


 # Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
ite = 0
fourcc = cv.CV_FOURCC('F', 'L', 'V', '1')
video = cv2.VideoWriter('balltrack.avi',fourcc,24,(old_gray.shape[1],old_gray.shape[0]))

for ite in range((int)(frCount) - 1):
    mask = np.zeros_like(old_frame)
    ret,frame = cap.read()
    frame = cv2.resize(frame, (width*1,(int)(height*1)))

    #frame_gray = cv2.inRange(frame, color1, color2)
    cv2.imshow("converted ", frame)
    #cv2.waitKey(0)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #frame_gray = cv2.blur(frame_gray,(20,20))
    frame_gray = cv2.dilate(frame_gray, np.ones((10, 10)),None, iterations= 1)
    frame_gray = cv2.erode(frame_gray, np.ones((2, 2)), None, iterations= 3)
    frame_gray = cv2.dilate(frame_gray, np.ones((10, 10)), None, iterations= 1)


    ptot0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    #cv2.imshow('msk', frame_gray)
    #cap.read()

    frdisp = frame_gray.copy()
    #print(ite)
    if(p0 != None):
        
         # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        ptot1, sttot, errtot = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, ptot0, None, **lk_params)

        #print(st)
        #print(p0)
        #print(p0)
         # Select good points
        if p1!=None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            good_newtot = ptot1[sttot==1]
            good_oldtot = ptot0[sttot==1]

       # print(good_old)
        #print(good_new)

        if False and (len(good_new)<4 or len(good_old)<4 or p1 == None):

            pts = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            for [[u, v]] in pts:
                cv2.circle(frame,(u,v),3,color[i].tolist(),-1)

            #x,y = indicateLocation(frame)
            ctr = 0

            for [[u, v]] in pts:
                if abs(u-x)<15 and abs(v-y)<15:
                    ctr = ctr+1
            p0 = np.zeros((ctr,1,2), np.float32)
            ctr = 0
            for [[u, v]] in pts:
                if abs(u-x)<15 and abs(v-y)<15:
                    p0[ctr] = [[u, v]]
                    ctr = ctr+1
            good_new = p0
            good_old = p0
            #print(p0)
     
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


        for i,(new,old) in enumerate(zip(good_newtot,good_oldtot)):
            a,b = new.ravel()
            c,d = old.ravel()

            if(a-c == 0):
                angle = 90/5
            else :
                angle = (int)(np.rad2deg(np.arctan((b-d)/(a-c)))/5)

            angles[angle] = angles[angle] + 1


            if( abs(a-c)<100 and abs(b-d)<100):
                count = count + 1
                avSumX = avSumX + a - c
                avSumY = avSumY + b - d
                mvs[(int)(a-c)][(int)(b-d)] = mvs[(int)(a-c)][(int)(b-d)] + 1               
            

        avMovY = (int)(avSumY/count)
        avMovX = (int)(avSumX/count)

        if avMovX == 0:
            avMovX = 1

        if avMovY == 0:
            avMovY = 1


        #print("frames0min "+str(frames0min))
        candidates = list()


       # print("mov "+str(avMovX)+" "+str(avMovY))
        final = good_new
        todelete = list()
        for i,(new,old) in enumerate(zip(good_newtot, good_oldtot)):
            a,b = new.ravel()
            c,d = old.ravel()
            #cv2.circle(frdisp,(a,b),5,color[0].tolist(),-1)
            if True and ( a-c < -(abs(avMovX)+1)*2 or b-d < -(abs(avMovY)+1)*2  or a-c > (abs(avMovX)+1)*2 or b-d > (abs(avMovY)+1)*2  or (sig(avMovX)!=sig(a-c) and avMovX>2) or (sig(avMovY)!=sig(b-d) and avMovY>2)):
                if(a-c == 0):
                    angle = 90 / 5
                else :
                    angle = (int)(np.rad2deg(np.arctan((b-d)/(a-c)))/5)
                #print(mvs[(int)(a-c)][(int)(b-d)])
                if(True or angles[angle] < 10 and (abs(a-c)>=3 or abs(b-d)>=3)  and (abs(a-c)<20 and abs(b-d)<20)):
                    #print(mvs[(int)(a-c)][(int)(b-d)])
                    cv2.line(frame, (a,b),(c,d), color[0].tolist(), 2)
                    cv2.circle(frame,(a,b),3,color[0].tolist(),-1)
                    candidates.append([[a,b, a-c, b-d]])
                    candidates

                    #print("delete "+ str((int)(abs(a-c)))+" "+str((int)(abs(b-d))))
                    #todelete.append(i)
              
                    #cv2.circle(mask,(a,b),10,color[0].tolist(),-1)
                #if(mvs[(int)(a-c)][(int)(b-d)]>4):
                    #cv2.circle(frame,(,b),20,color[1].tolist(),-1)
                #objects2.append((a,b))
                                                                                           
            realCandidates = list()
            bestCandidate = (0,0)
            bestCandidateCount = 0
            for [[a,b, x, y]] in candidates:
                sins = list()

                for [[a1,b1, x1, y1]] in candidates:
                    if(abs(abs(x)-abs(x1)) < 10 and abs(abs(y)-abs(y1))<10 and abs(a1-a)<50 and abs(b-b1)<50 and (abs(a1-a)>8  or abs(b-b1)>8)):
                        #sins = sins + 1
                        sins.append((a1,b1))
                        cv2.circle(frame,(a,b),10,color[7].tolist(),-1)  
                if sins > bestCandidateCount and len(sins)>1:
                    bestCandidateCount = len(sins)
                    sins.append((a,b))

                    xCount = 0;
                    yCount = 0;

                    for (u,v) in sins:
                        xCount = xCount + u / len(sins)
                        yCount = yCount + v / len(sins)


                    bestCandidate = (xCount,yCount)
                    #realCandidates.append([[a,b]])
                    #                       cv2.circle(frame,(a,b),10,color[7].tolist(),-1)        
           # print(str(bestCandidateCount))

            #cv2.line(frame, (a,b),(c,d), color[10].tolist(), 2)
            #cv2.circle(frame,(a,b),3,color[10].tolist(),-1)
            #print("delete everything")      
        good_new = np.delete(good_new, todelete, axis = 0)
        p0 = good_newtot.reshape(-1,1,2)

        xCount = 0;
        yCount = 0;

        #for [[u,v]] in realCandidates:
        #    xCount = xCount + u / len(realCandidates)
        #    yCount = yCount + v / len(realCandidates)

        #if len(realCandidates)>1:
        (a,b) = bestCandidate

        if a>0 or b>0:
            ballx, bally = a, b

        ballPositions.append([ballx, bally, avMovX, avMovY])

        cv2.circle(frame,((int)(ballx),(int)(bally)),20,color[1].tolist(),-1)

        #cv2.imshow("f", mask)

        img = cv2.add(frame,mask)
        #display(img)
        cv2.imshow("after",frame_gray)
        cv2.imshow('frame',img)
        video.write(img)

    ite = ite +1
    old_gray = frame_gray
    cv2.waitKey(20)
      
     
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

np.save(outputFile, ballPositions)
cv2.destroyAllWindows()
video.release()
cap.release()