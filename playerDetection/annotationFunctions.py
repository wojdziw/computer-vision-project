import cv2
import numpy as np

def colourComponentBlack(image, visited):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if visited[i,j]:
				image[i,j] = [0,0,0]
				
	return image

def drawCrosses(image, centreX, centreY, bottomestX=0, bottomestY=0):
	# -1 to prevent overflow
	# all these maxs and mins to prevent drawing outside of image
	imageHeight = image.shape[0]-1
	imageWidth = image.shape[1]-1

	image[max(0,bottomestX-7):min(imageHeight,bottomestX+7),min(bottomestY, imageWidth)]=[0,0,255]
	image[min(bottomestX,imageHeight),max(0,bottomestY-7):min(imageWidth,bottomestY+7)]=[0,0,255]
	image[max(0,centreX-7):min(imageHeight,centreX+7),min(centreY, imageWidth)]=[0,0,255]
	image[min(centreX, imageHeight),max(0,centreY-7):min(imageWidth,centreY+7)]=[0,0,255]

	return image

def markPositions(videoNumber, playerName, positions, jumps=np.zeros([1,1])):

	vidObj = cv2.VideoCapture('../beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

	frameWidth = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameFPS = int(vidObj.get(cv2.CAP_PROP_FPS))
	frameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))

	name = ""
	if jumps.shape[0] == 1:
		# just initialise the jumps to be false
		jumps = np.zeros(frameCount, bool)
		name = "feet"
	else:
		name = "jumps"

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outObj = cv2.VideoWriter('../output/' + name + str(videoNumber) + "_" + playerName + '.avi',fourcc, frameFPS, (frameWidth, frameHeight))

	# Read first frame
	_,image = vidObj.read()

	words = cv2.imread("jumpWords.jpg")

	for fr in range(1,frameCount):

		_,image = vidObj.read()

		if jumps[fr]:
			image[0:100,0:150] = words

		image = drawCrosses(image, positions[fr, 0], positions[fr, 1])

		outObj.write(image)

	vidObj.release()
	outObj.release()