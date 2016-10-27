import cv2

def colourComponentBlack(image, visited):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if visited[i,j]:
				image[i,j] = [0,0,0]
				
	return image

def drawCrosses(image, centreX, centreY, bottomestX, bottomestY):
	image[bottomestX-7:bottomestX+7,bottomestY]=[0,0,255]
	image[bottomestX,bottomestY-7:bottomestY+7]=[0,0,255]
	image[centreX-7:centreX+7,centreY]=[0,0,255]
	image[centreX,centreY-7:centreY+7]=[0,0,255]

	return image

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