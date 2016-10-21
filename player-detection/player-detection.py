import cv2
import numpy as np
from functions import *
from annotationFunctions import *

# indicating the location
# indicatedLocation = [170,420]
indicatedLocation = [254,342]

# opening the video
# video = cv2.VideoCapture('../beach-volleyball-films/beachVolleyball2.mov') - working
video = cv2.VideoCapture('../beach-volleyball-films/beachVolleyball5.mov')

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(video.get(cv2.CAP_PROP_FPS))
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 40, (frameWidth, frameHeight))

_,image = video.read()

print(frameCount)

# Problem:
# Rewrite the findNewCentre function so that it finds the component from the previous frame better

# Cropping out a window around the indicated component to find its colour
previousColour = patchColour(image, indicatedLocation)

for fr in range(1,frameCount):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY, notSameColour, visited, previousColour = componentCoords(image, indicatedLocation, previousColour)
	indicatedLocation = [centreX, centreY]

	# adding the image features
	image = colourComponentBlack(image, visited)
	image = drawCrosses(image, centreX, centreY, bottomestX, bottomestY)

	# saving the frame to a jpg
	frameName = 'individual-frames/frame' + str(fr) + '.jpg'
	cv2.imwrite(frameName, image)

	# saving the image frame
	out.write(image)
	_,image = video.read()

video.release()
out.release()

