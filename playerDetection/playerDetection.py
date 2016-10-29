import cv2
import numpy as np
from functions import *
from annotationFunctions import *


def playerDetection(videoNumber, indicatedLocation, startFr):

	vidObj = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

	frameWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
	frameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outObj = cv2.VideoWriter('output/vid' + str(videoNumber) + '.avi',fourcc, frameFPS, (frameWidth, frameHeight))

	# Read first frame
	for i in range(startFr):
		_,image = vidObj.read()

	# Set up output
	positions = np.zeros([frameCount, 2], int)
	positions[0] = indicatedLocation # TODO: indicate bottom-most position here instead as well!?
	areas = np.zeros(frameCount, int)

	# Cropping out a window around the indicated component to find its BGR colour
	startColour = patchColour(image, indicatedLocation)
	previousColour = startColour

	moveVec = [0, 0]

	for fr in range(startFr,frameCount):
    	
		print str(fr) + " out of " + str(frameCount)
		_,image = vidObj.read()

		# Finding the centre of the chosen component (e.g. leg, torso) and its bottom point
		try:
			centreR, centreC, bottomestR, bottomestC, visited, previousColour = componentCoords(image, indicatedLocation, previousColour, startColour, moveVec)
			moveVec = np.array([centreR, centreC]) - np.array(indicatedLocation) # Normalize?
			# print "moveVec" + str(moveVec)
			indicatedLocation = [centreR, centreC]
		except:
			print "Crash!"

		# Adding the image features
		image = colourComponentBlack(image, visited)
		image = drawCrosses(image, centreR, centreC, bottomestR, bottomestC)

		# Saving the frame to a jpg
		cv2.imwrite('playerDetection/individualFrames/frame' + str(fr) + '.jpg', image)

		# Saving the image frame
		outObj.write(image)

		# Saving new position
		positions[fr] = [bottomestR, bottomestC] 
		areas[fr] = calculateArea(visited)
		

	vidObj.release()
	outObj.release()

	return positions, areas

