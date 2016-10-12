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

for fr in range(1,77):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY = componentCoords(image, componentLocation)
	componentLocation = [centreX, centreY]

	# drawing crosses
	image[bottomestX-7:bottomestX+7,bottomestY]=[0,0,255]
	image[bottomestX,bottomestY-7:bottomestY+7]=[0,0,255]
	image[centreX-7:centreX+7,centreY]=[0,0,255]
	image[centreX,centreY-7:centreY+7]=[0,0,255]

	out.write(image)

	_,image = video.read()

video.release()
out.release()




# # saving into a file
# cv2.imwrite('frame100_experiment.jpg', image)
