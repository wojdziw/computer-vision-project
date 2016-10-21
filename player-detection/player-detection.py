import cv2
# import cv2.cv as cv
import numpy as np
from functions import *
from annotationFunctions import *

# indicating the location
indicatedLocation = [170,420] # working 70 frames

# opening the video
video = cv2.VideoCapture('../beach-volleyball-films/beachVolleyball2.mov')

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(video.get(cv2.CAP_PROP_FPS))
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 40, (frameWidth, frameHeight))

_,image = video.read()

print(frameCount)

# PROBLEMS:
# 1) Colour might not be chosen well - maybe calculate the colour of the whole component 
# -- it might be okay enough, just make sure from frame to frame it doesn't change much
# 2) Point sometimes lies outside the component and can't detect anything
# -- if we search for the new colour patch well enough that shouldn't be a problem
# 3) The guy moves his leg too quickly and the algorithm starts from the sand
# -- same as above

# Cropping out a window around the indicated component to find its colour
previousColour = patchColour(image, indicatedLocation)

for fr in range(1,frameCount):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY, notSameColour, visited, previousColour = componentCoords(image, indicatedLocation, previousColour)
	indicatedLocation = [centreX, centreY]

	# ADDING IMAGE FEATURES
	# colouring the component black
	image = colourComponentBlack(image, visited)

	# drawing crosses
	image = drawCrosses(image, centreX, centreY, bottomestX, bottomestY)

	# saving the frame to a jpg
	frameName = 'individual-frames/frame' + str(fr) + '.jpg'
	cv2.imwrite(frameName, image)

	out.write(image)

	_,image = video.read()

video.release()
out.release()

