#!/usr/bin/env python 

import numpy as np
import cv2
import math
import sys
from functions import *

# Choose which video to process 
if (sys.argv < 1):
	print 'Usage: main.py <VideoIndex>'
else:
	try:
		vidNr = str(sys.argv[1])
	except:
		print 'Usage: main.py <VideoIndex>'

# Where to start look
indLoc = np.zeros(2, int)
indLoc[0] = 235 #uint8(argv[2])
indLoc[1] = 496 #uint8(argv[3])
rad = 10

print indLoc

vidObj = cv2.VideoCapture('vids/beachVolleyball'+vidNr+'.mov')

frWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
frHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
frFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
frCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outObj = cv2.VideoWriter('output/vid'+vidNr+'3.avi',fourcc, frFPS, (frWidth, frHeight))

# Read first frame
_,img = vidObj.read()
outImg = np.zeros(img.shape, np.uint8)
#cv2.imwrite('vid1.jpg', img)


# Define the region mask where to look
mask = np.zeros(img.shape[:2], np.uint8)
mask[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad] = 255
cv2.imwrite('mask.jpg', mask)

test = np.zeros(img.shape, np.uint8)
test[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad] = img[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad]
cv2.imwrite('test.jpg', test)
#print mask.shape

# Use gaussian blur
#grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gauss = gauss_kernels(5,1)
#img = MyConvolve(grayImg, gauss)
#cv2.imwrite('img.jpg', img)



# Create SIFT and find the best features in the first frame
# Could probably use our own detector (Harris/Tomasi) and just use this as compute descriptor
# We would need to have the input as cv.KeyPoints in that case! 
sift = cv2.xfeatures2d.SIFT_create()				# Look for features
(kp1, desc1) = sift.detectAndCompute(img, mask)		# Look only in mask 

# Convert to 2D point and find the bottom-most keypoint!
pts = np.zeros([len(kp1), 2])
btm, foot = 0, 0
firstPt = np.zeros(2, int)

for idx in range(len(kp1)):
	pts[idx] = kp1[idx].pt # [x, y]
	if (pts[idx][1]>btm):
		foot = idx
		btm = pts[idx][1]

desc1 = np.matrix(desc1[foot])
firstPt = [int(pts[foot][1]), int(pts[foot][0])] #change [x, y] to [row, col] 
print desc1
print firstPt
print len(kp1)
print len(desc1)

# Draw first frame

outImg = img
#cv2.drawKeypoints(img, kp1, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
mask[firstPt[0]-5:firstPt[0]+5, firstPt[1]-5:firstPt[1]+5] = 255 
mask[firstPt[0]-3:firstPt[0]+3, firstPt[1]-3:firstPt[1]+3] = 0
outImg[mask] = [255, 0, 0] 
outObj.write(outImg)

for fr in range(1, frCount):

	_,nextFr = vidObj.read()

	# Use gaussian blur
	#grayImg = cv2.cvtColor(nextFr, cv2.COLOR_BGR2GRAY)
	#gauss = gauss_kernels(5,1)
	#nextFr = MyConvolve(grayImg, gauss)

	# Find all desciptors in image, should use bigger mask only!?
	mask[firstPt[0]-2*rad:firstPt[0]+2*rad,firstPt[1]-2*rad:firstPt[1]+2*rad] = 255
	(kp2, desc2) = sift.detectAndCompute(nextFr, mask)


	# Define Brute Force matcher (Find the nearest descriptor)
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
	matches = bf.match(desc1, desc2) # Returns best result for the foot index

	# DMatch objects, (have distance, trainIdx, queryIdx (index of descriptors), imgIdx) 
	matches = sorted(matches, key = lambda x:x.distance)
	print len(matches)

	if(len(matches)>=1):
		newPt = kp2[matches[0].trainIdx].pt
		newPt = [int(newPt[1]), int(newPt[0])]
	
	#desc1 = desc2[matches[0].queryIdx]
	desc1 = desc2
	firstPt = newPt

	print newPt

	# Draw first frame
	outImg = nextFr
	#cv2.drawKeypoints(nextFr, kp2, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	mask[firstPt[0]-5:firstPt[0]+5, firstPt[1]-5:firstPt[1]+5] = 255 
	mask[firstPt[0]-3:firstPt[0]+3, firstPt[1]-3:firstPt[1]+3] = 0
	outImg[mask] = [255, 0, 0] 
	outObj.write(outImg)

vidObj.release()
outObj.release()
