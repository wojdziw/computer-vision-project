import cv2
import numpy as np
from functions import *
from annotationFunctions import *

def markPositions(videoNumber, positions):

	vidObj = cv2.VideoCapture('../beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

	frameWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
	frameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outObj = cv2.VideoWriter('../output/positions' + str(videoNumber) + '.avi',fourcc, frameFPS, (frameWidth, frameHeight))

	# Read first frame
	_,image = vidObj.read()

	for fr in range(1,frameCount):

		_,image = vidObj.read()

		image = drawCrosses(image, positions[fr, 0], positions[fr, 1], 20,20)

		outObj.write(image)

	vidObj.release()
	outObj.release()




