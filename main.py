#!/usr/bin/env python 

import numpy as np
import cv2
import math
import sys

# Choose which video to process 
try:
	vidNr = str(sys.argv[1])
except:
	print 'Usage: main.py <VideoIndex>'

indLoc = np.zeros(2, np.uint8)
indLoc[0] = 227 #uint8(argv[2])
indLoc[1] = 494 #uint8(argv[3])

vidObj = cv2.VideoCapture('vids/beachVolleyball'+vidNr+'.mov')

frWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
frHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
frFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
frCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outObj = cv2.VideoWriter('output/vid'+vidNr+'2.avi',fourcc, frFPS, (frWidth, frHeight))

# Read first frame
_,img = vidObj.read()
outImg = np.zeros(img.shape, np.uint8)
#cv2.imwrite('vid1.jpg', img)

# Define the region mask (25x25) where to look
mask = np.zeros(img.shape[:2], np.uint8)
mask[indLoc[0]-24:indLoc[0]+24,indLoc[1]-24:indLoc[1]+24] = 255
#mask[:,:] = 2

# Create SIFT and find the best features in the first frame
# Could probably use our own detector (Harris/Tomasi) and just use this as compute descriptor
# We would need to have the input as cv.KeyPoints in that case! 
#grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()				# Look for features
(kp1, desc1) = sift.detectAndCompute(img, mask)		# Look only in mask 


for fr in range(1, frCount):

	_,nextFr = vidObj.read()

	(kp2, descs2) = sift.detectAndCompute(nextFr, None)

	# Find matches
	FLANN_INDEX_KDTREE = 0
   	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   	search_params = dict(checks = 50)
   	
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	 
	matches = flann.knnMatch(des1,des2,k=2)
	 
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)



	cv2.drawKeypoints(nextFr, kp2, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	outObj.write(outImg)


vidObj.release()
outObj.release()
