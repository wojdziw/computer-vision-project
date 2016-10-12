import cv2
import cv2.cv as cv
import numpy as np
from functions import *

# indicating the location
componentLocation = [170,420] # working 70 frames
# componentLocation = [88,427]
bottomestX = 225
bottomestY = 375

# opening the video
video = cv2.VideoCapture('beachVolleyball/beachVolleyball2.mov')
frameCount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
frameWidth = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 40, (frameWidth, frameHeight))

_,image = video.read()

# PROBLEMS:
# 1) Colour might not be chosen well - maybe calculate the colour of the whole component again
# 2) Point sometimes lies outside the component and can't detect anything

for fr in range(1,123):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY, notSameColour, visited = componentCoords(image, componentLocation)
	componentLocation = [centreX, centreY]

	# if the new starting point isn't within the component, find the closest point that is
	if not visited[centreX,centreY]:
		print "Nieee " + str(fr)
		while not visited[centreX,centreY]:
			centreX -= 1

	# drawing crosses
	image[bottomestX-7:bottomestX+7,bottomestY]=[0,0,255]
	image[bottomestX,bottomestY-7:bottomestY+7]=[0,0,255]
	image[centreX-7:centreX+7,centreY]=[0,0,255]
	image[centreX,centreY-7:centreY+7]=[0,0,255]

	# colouring the component black
	# for i in range(image.shape[0]):
	# 	for j in range(image.shape[1]):
	# 		if visited[i,j]:
	# 			image[i,j] = [0,0,0]

	# saving the frame to a jpg
	frameName = 'individualFrames/frame' + str(fr) + '.jpg'
	cv2.imwrite(frameName, image)

	out.write(image)

	_,image = video.read()

video.release()
out.release()




# # saving into a file
# cv2.imwrite('frame100_experiment.jpg', image)
