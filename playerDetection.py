import numpy as np
import cv2
import math
import sys
from functions import *


# Given a starting position it tracks using SIFT descriptors
def playerDetec(videoIdx, startPos, radius):

	vidNr = videoIdx

	# Where to start look
	indLoc = np.zeros(2, int)
	indLoc[0] = startPos[0]
	indLoc[1] = startPos[1]
	rad = radius

	# Read video object
	vidObj = cv2.VideoCapture('vids/beachVolleyball'+vidNr+'.mov')

	frWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
	frHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
	frCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outObj = cv2.VideoWriter('output/vid'+vidNr+'.avi',fourcc, frFPS, (frWidth, frHeight))

	# Read first frame
	_,img = vidObj.read()
	outImg = img
	# Print first frame to find where to start looking
	cv2.imwrite('output/detStart'+vidNr+'.jpg', img)

	# Initialize output
	pos = np.zeros([frCount, 2], int)

	# Define the region mask where to look
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad] = 255
	cv2.imwrite('output/mask.jpg', mask)

	# Test whether we found the right spot
	test = np.zeros(img.shape, np.uint8)
	test[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad] = img[indLoc[0]-rad:indLoc[0]+rad,indLoc[1]-rad:indLoc[1]+rad]
	cv2.imwrite('output/test.jpg', test)

	'''
	# Use gaussian blur
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gauss = gauss_kernels(5,1)
	img = MyConvolve(grayImg, gauss)
	cv2.imwrite('output/blurimg.jpg', img)
	'''

	# Create SIFT and find the best features in the first frame
	# Could probably use our own detector (e.g. Harris/Tomasi) and just use OpenCV to compute SIFT descriptor
	# We would need to have the input to sift.compute() as cv.KeyPoints in that case! 
	sift = cv2.xfeatures2d.SIFT_create()				# Create SIFT
	(kp1, desc1) = sift.detectAndCompute(img, mask)		# Look only in mask region 

	# Convert to 2D point and find the bottom-most keypoint!?
	# Find the one with strongest response instead!?
	# Or find the average? Combine with color?
	pts = np.zeros([len(kp1), 2])
	best, foot = 0, 0
	firstPt = np.zeros(2, int)
	'''
	# Find bottom-most keypoint
	for idx in range(len(kp1)):
		pts[idx] = kp1[idx].pt # [x, y]
		if (pts[idx]>btm):
			foot = idx
			btm = pts[idx][1]
	'''
	# Find keypoint with strongest response
	for idx in range(len(kp1)):
		if (kp1[idx].response>best):
			foot = idx
			best = kp1[idx].response


	# Store only the best/lowest result
	desc1 = np.matrix(desc1[foot]) 		# Comment if all descriptors are used

	#firstPt = [int(pts[foot][1]), int(pts[foot][0])] 		# Lowet desc, change [x, y] to [row, col] 
	firstPt = [int(kp1[foot].pt[1]), int(kp1[foot].pt[0])]	#Strongest response
	pos[0] = firstPt
	#print desc1
	print firstPt


	# Draw first frame
	#cv2.drawKeypoints(img, kp1, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	drawSq = np.zeros(mask.shape)
	drawSq[firstPt[0]-5:firstPt[0]+5, firstPt[1]-5:firstPt[1]+5] = 255 
	drawSq[firstPt[0]-3:firstPt[0]+3, firstPt[1]-3:firstPt[1]+3] = 0
	drawMask = (drawSq == 255)
	outImg[drawMask] = [0, 0, 255] 
	outObj.write(outImg)

	for fr in range(1, frCount):

		# Read next frame and prepare output image
		_,nextFr = vidObj.read()
		outImg = nextFr

		'''
		# Use gaussian blur
		grayImg = cv2.cvtColor(nextFr, cv2.COLOR_BGR2GRAY)
		gauss = gauss_kernels(5,1)
		grayImg = MyConvolve(grayImg, gauss)
		'''

		# Find all desciptors in an area around the first point 
		mask = np.zeros(mask.shape, np.uint8)
		mask[firstPt[0]-rad:firstPt[0]+rad,firstPt[1]-rad:firstPt[1]+rad] = 255
		(kp2, desc2) = sift.detectAndCompute(nextFr, mask)
		if desc2 is None: # Didn't find any descriptors, keep old values
			print "No detections"

		else: 

			# Define Brute Force matcher (Find the nearest matching descriptor)
			bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
			matches = bf.match(desc1, desc2) # Returns only the best result for every descriptor
			#matches = bf.match(desc1, desc2, 5) # Returns matches within Hamming distance (NOT pixels)

			# DMatch objects, (have distance, trainIdx, queryIdx (index of descriptors) & imgIdx) 
			matches = sorted(matches, key = lambda m:m.distance) # Sort by distance
			#print len(matches)
			
			if(len(matches)>=1):
				newPt = kp2[matches[0].trainIdx].pt 	# Copy the 2D coord from best descriptor
				newPt = [int(newPt[1]), int(newPt[0])]	# Convert to int and switch place
			
			# Replace old values
			desc1 = np.matrix(desc2[matches[0].trainIdx]) 	# Use only best descriptor
			#desc1 = desc2 									# Use all descriptors 
			firstPt = newPt

			print newPt

		


		# Draw red square around point
		#cv2.drawKeypoints(nextFr, kp2, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		drawSq = np.zeros(mask.shape)
		drawSq[newPt[0]-5:newPt[0]+5, newPt[1]-5:newPt[1]+5] = 255 
		drawSq[newPt[0]-3:newPt[0]+3, newPt[1]-3:newPt[1]+3] = 0
		drawMask = (drawSq == 255)
		outImg[drawMask] = [0, 0, 255]  
		outObj.write(outImg)

		pos[fr] = newPt

	vidObj.release()
	outObj.release()


	return pos
