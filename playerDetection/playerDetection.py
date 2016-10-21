import cv2
import numpy as np
from functions import *
from annotationFunctions import *


def playerDetection(videoNumber, indicatedLocation):

	video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

	frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameFPS = int(video.get(cv2.CAP_PROP_FPS))
	frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output' + str(videoNumber) + '.avi',fourcc, 40, (frameWidth, frameHeight))

	_,image = video.read()

	positions = np.zeros([frameCount, 2], int)
	positions[0] = indicatedLocation

	# Cropping out a window around the indicated component to find its colour
	previousColour = patchColour(image, indicatedLocation)

	for fr in range(1,frameCount):
		print str(fr) + " out of " + str(frameCount)
		# finding the centre of the component (e.g. leg) and its bottom point
		try:
			centreX, centreY, bottomestX, bottomestY, notSameColour, visited, previousColour = componentCoords(image, indicatedLocation, previousColour)
			indicatedLocation = [centreX, centreY]
		except:
			print "Crash!"

		positions[fr] = indicatedLocation

		# adding the image features
		image = colourComponentBlack(image, visited)
		image = drawCrosses(image, centreX, centreY, bottomestX, bottomestY)

		# saving the frame to a jpg
		cv2.imwrite('individualFrames/frame' + str(fr) + '.jpg', image)

		# saving the image frame
		out.write(image)
		_,image = video.read()

	video.release()
	out.release()

	return positions

