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