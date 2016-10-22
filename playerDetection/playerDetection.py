import cv2
import numpy as np
from functions import *
from annotationFunctions import *


def playerDetection(videoNumber, indicatedLocation):

	vidObj = cv2.VideoCapture('beach-volleyball-films/beachVolleyball'+str(videoNumber)+'.mov')

	frameWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
	frameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outObj = cv2.VideoWriter('output/vid' + str(videoNumber) + '.avi',fourcc, frameFPS, (frameWidth, frameHeight))

	# Read first frame
	_,image = vidObj.read()

	# Set up output
	positions = np.zeros([frameCount, 2], int)
	positions[0] = indicatedLocation # TODO: indicate bottom-most position here instead as well!?
	jumps = 0

	# Cropping out a window around the indicated component to find its BGR colour
	startColour = patchColour(image, indicatedLocation)
	previousColour = startColour

	moveVec = indicatedLocation

	for fr in range(1,frameCount):
    	
		print str(fr) + " out of " + str(frameCount)
		_,image = vidObj.read()

		# Finding the centre of the chosen component (e.g. leg, torso) and its bottom point
		try:
			centreR, centreC, bottomestR, bottomestC, visited, previousColour = componentCoords(image, indicatedLocation, previousColour, startColour)
			moveVec = np.array(indicatedLocation) - np.array([centreR, centreC]) 
			indicatedLocation = [centreR, centreC]
		except:
			print "Crash!"

		# Adding the image features
		image = colourComponentBlack(image, visited)
		image = drawCrosses(image, centreR, centreC, bottomestR, bottomestC)

		# TODO: Detect jump 
		# Calc gradient from moveVec, when peak has been reached, jumps++
		#jumps += 1

		# Saving the frame to a jpg
		#cv2.imwrite('playerDetection/individual-frames/frame' + str(fr) + '.jpg', image)

		# Saving the image frame
		outObj.write(image)

		# Saving new position
		# TODO: Calculate position of feet! Take moveVec into account for jumps
		positions[fr] = [bottomestR, bottomestC] 
		

	vidObj.release()
	outObj.release()

	return positions, jumps

