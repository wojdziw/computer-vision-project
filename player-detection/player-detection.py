import cv2
# import cv2.cv as cv
import numpy as np
from functions import *

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

previousVisited = np.zeros([image.shape[0], image.shape[1]], dtype=bool)

# Cropping out a window around the indicated component to find its colour
sample = image[indicatedLocation[0]-5:indicatedLocation[0]+5,indicatedLocation[1]-5:indicatedLocation[1]+5]
greenAvg = np.mean(sample[:,:,0])
blueAvg = np.mean(sample[:,:,1])
redAvg = np.mean(sample[:,:,2])
# Finding the average colour of the component
previousColour = [greenAvg,blueAvg,redAvg]

for fr in range(1,300):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY, notSameColour, visited, previousColour = componentCoords(image, indicatedLocation, previousColour)

	# if the point doesn't lie within the previous component area, don't update
	# I don't really think that is used much with the new "colour search" fixes
	if not previousVisited[centreX,centreY]:
		centreX = indicatedLocation[0]
		centreY = indicatedLocation[1]

	previousVisited = visited

	# # RESTRICTING THE POINTER MOVEMENT
	# # calculating the vector between the old location and the new location
	# vector = [centreX-indicatedLocation[0], centreY-indicatedLocation[1]]
	# vectorLen = np.sqrt(vector[0]*vector[0]+vector[1]*vector[1])

	# # restricting the component pointer movement - this criterion isn't very good for some reason
	# # the reason is that the indication of things going bad is not pointer moving far - it's the colours being messed up
	# if vectorLen > 15 and fr != 1:
	# 	print("nieeee")
	# 	centreX = indicatedLocation[0]
	# 	centreY = indicatedLocation[1]

	indicatedLocation = [centreX, centreY]

	# ADDING IMAGE FEATURES
	# colouring the component black
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if visited[i,j]:
				image[i,j] = [0,0,0]

	# drawing crosses
	image[bottomestX-7:bottomestX+7,bottomestY]=[0,0,255]
	image[bottomestX,bottomestY-7:bottomestY+7]=[0,0,255]
	image[centreX-7:centreX+7,centreY]=[0,0,255]
	image[centreX,centreY-7:centreY+7]=[0,0,255]

	# saving the frame to a jpg
	frameName = 'individual-frames/frame' + str(fr) + '.jpg'
	cv2.imwrite(frameName, image)

	out.write(image)

	_,image = video.read()

video.release()
out.release()