import numpy as np
import cv2
import cv2.cv as cv
from collections import defaultdict
from Tkinter import *
import matplotlib.pyplot as plt
from tkFileDialog import askopenfilename
import Image, ImageTk
import cv2
import numpy as np

def applyPoly(coeff, x):
	return x*x*coeff[0] + x*coeff[1] + coeff[2]

def applyParabolicInterpolation(startPointComp, endPointComp, startPointInt, endPointInt, ballPos, maskLogo):
	xs = [x[0] for x in ballPos[startPointComp:endPointComp] if x[1]>maskLogo*60  and x[0]>0]
	ys = [x[1] for x in ballPos[startPointComp:endPointComp] if x[1]>maskLogo*60  and x[0]>0]

	nb = endPointComp - startPointComp
	#correct the points according to camera movement
	cumulX = 0
	cumulY = 0
	#for i in range(startPointComp, endPointComp):
	#	cumulX = cumulX + ballPos[i][2]
	#	cumulY = cumulY + ballPos[i][3]
	#	xs[i-startPointComp] = xs[i-startPointComp] - cumulX
	#	ys[i-startPointComp] = ys[i-startPointComp] - cumulY

	print(xs)


	sortedX = sorted(xs)

	start, end = (int)(ballPos[startPointComp][0]), (int)(ballPos[endPointComp][0])
	startY  = (int)(ballPos[startPointComp][1])
	curvX = list()            
	curvY = list()

	res = np.polyfit(xs, ys, 2)


	#compute the parabolic curve
	delta =  (end - start) / nb +1
	print("delta "+ str(delta))
	startX = start - (startPointComp-startPointInt)*delta
	cumulX = 0
	cumulY = 0
	for i in range(endPointInt-startPointInt):
		#cumulX = cumulX + ballPos[i + startPointComp][2]
		#cumulY = cumulY + ballPos[i + startPointComp][3]
		x = startX + i*delta
		r = applyPoly(res, x)
		ec = startY - r
		print(ec)
		final = r
		#if(invert):
		#	final = final + 2*ec
		ballPos[startPointInt + i] = [x - cumulX, final-cumulY, 0, 0]
		curvX.append(x-cumulX)
		curvY.append(final-cumulY)

	print(res)

	plt.scatter(curvX, curvY)            
	plt.scatter(xs, ys)
	plt.show()
#video 3 : 
# compute para from 115 to 155 and apply from 88 to 169
#video 1

if len(sys.argv)<4:
    print("Usage : <VideoFile> <BallPosFile> <OutputFile> <ParaboleFile>")
    sys.exit(0)

videoFile = sys.argv[1]
ballPosFile = sys.argv[2]
outputFile = sys.argv[3]
paraboles = None
if len(sys.argv)>4:
	parabolFile = sys.argv[4]
	paraboles = np.load(parabolFile)
	maskLogo = (int)(sys.argv[5])

ballPos = np.load(ballPosFile)
ballPosOr = ballPos.copy()
cap = cv2.VideoCapture(videoFile)

frCount = (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
[prevX, prevY, avMovX, avMovY]= ballPos[0]

if paraboles != None:
	#video7
	print(str(len(paraboles))+" paraboles detected")
	for [f1,f2,f3,f4] in paraboles:
		applyParabolicInterpolation(f1, f2, f3, f4, ballPos, maskLogo)
	

ret, frame = cap.read()
fourcc = cv.CV_FOURCC('F', 'L', 'V', '1')
video = cv2.VideoWriter('../data/videos/balltrackEnhanced.avi',fourcc,24,(frame.shape[1],frame.shape[0])) 
print("sizes ")
print(frame.shape)
#play video
for i in range(frCount-1):
	ret, frame = cap.read()

	[a, b,x, y] = ballPos[i]
	[c, d, x1, y1] = ballPosOr[i]
	cv2.circle(frame,((int)(a),(int)(b)),10,[255,255,0],-1)
	#cv2.circle(frame,((int)(c),(int)(d)),10,[255,0,0],-1)
	#cv2.line(frame, (100,100),(100+(int)(x),100+(int)(y)), [255,0,0], 2)
	video.write(frame)
	cv2.imshow("frame", frame)
	cv2.waitKey(10)
	print(i)

np.save(outputFile, ballPos)
cap.release()
video.release()