import cv2
import numpy as np
from functions import *

# indicating the location
componentLocation = [170,420] # working 70 frames
# componentLocation = [88,427]
bottomestX = 225
bottomestY = 375

# opening the video
video = cv2.VideoCapture('vids/beachVolleyball2.mov')
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.mp4',fourcc, fps, (frameWidth, frameHeight))

_,image = video.read()

# PROBLEMS:
# 1) Colour might not be chosen well - maybe calculate the colour of the whole component again
# 2) Point sometimes lies outside the component and can't detect anything - THIS IS IMPORTANT

for fr in range(1,123):
	print fr
	# finding the centre of the component (e.g. leg) and its bottom point
	centreX, centreY, bottomestX, bottomestY, notSameColour, visited = componentCoords(image, componentLocation)

	# calculating the vector between the old location and the new location
	vector = [centreX-componentLocation[0], centreY - componentLocation[1]]
	vectorLen = np.sqrt(vector[0]*vector[0]+vector[1]*vector[1])
	
	centreX = int(centreX)
	centreY = int(centreY)

	print(vectorLen)

	# this criterion isn't very good for some reason
	if vectorLen > 15 and fr != 1:
		print("nieeee")
		centreX = componentLocation[0]
		centreY = componentLocation[1]

	componentLocation = [centreX, centreY]

	print(componentLocation)

	# if the new starting point isn't within the component, find the closest point that is
	# THIS DOESNT WORK - MIGHT HAVE BEEN VISITED BUT A DIFFERENT COLOUR
	# if not visited[centreX,centreY]:
	# 	print "Nieee " + str(fr)
	# 	while not visited[centreX,centreY]:
	# 		centreY += 1

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
