import cv2
import numpy as np

def MyConvolve(image, ff):
	result = np.zeros(image.shape)
	width = image.shape[0]
	height = image.shape[1]

	# Flip the filter
	ff = np.transpose(np.transpose(ff[::-1])[::-1])

	kernelSize = ff.shape[0]
	margin = kernelSize/2

	# Create a bigger matrix to pad with zeroes
	biggerResult = np.zeros([width+margin*2,height+margin*2])
	biggerResult[1:width+margin,1:height+margin] = image

	# Convolve
	for i in range(1,width+margin):
		for j in range(1,height+margin):
			multiplied = np.multiply(biggerResult[i-margin:i+margin+1,j-margin:j+margin+1],ff)
			summed = np.sum(multiplied)
			result[i-margin,j-margin] = summed

	return result

def nms(image):
	result = np.zeros(image.shape)
	edges = sobel(image)

	width = image.shape[0]
	height = image.shape[1]

	for i in range(width):
		for j in range(height):
			this = edges[i,j]

			# Save the neigbour values, make sure they are within the matrix (hence the mins and maxs)
			top = edges[max(i-1,0),j]
			bottom = edges[min(i+1,width-1),j]
			left = edges[i,max(j-1,0)]
			right = edges[i,min(j+1,height-1)]

			# Retain the edge if it's maximum in either axis
			if (max(top,bottom,this)==this) or (max(left,right,this)==this):
				result[i,j] = this

	return result

def sobel(image):
	result = np.zeros(image.shape)
	horizontal = np.zeros(image.shape)
	vertical = np.zeros(image.shape)

	horizontalFilter = np.matrix([[-1,0,1],[-2,0,2],[-1,0, 1]])
	verticalFilter = np.matrix([[1,2,1],[0,0,0],[-1,-2,-1]])

	# Find the horizontal and vertical edges first
	horizontal = MyConvolve(image, horizontalFilter)
	vertical = MyConvolve(image, verticalFilter)

	# Combine the two to get all edges
	result = np.sqrt(horizontal*horizontal+vertical*vertical)/4

	return result

def sobel_hor(image):
	result = np.zeros(image.shape)
	horizontal = np.zeros(image.shape)

	horizontalFilter = np.matrix([[1,2,1],[0,0,0],[-1,-2,-1]])

	# Find the horizontal edges first
	horizontal = MyConvolve(image, horizontalFilter)

	# Combine the two to get all edges
	result = horizontal/4

	return result

def sobel_ver(image):
	result = np.zeros(image.shape)
	vertical = np.zeros(image.shape)

	verticalFilter = np.matrix([[-1,0,1],[-2,0,2],[-1,0, 1]])

	# Find the vertical edges first
	vertical = MyConvolve(image, verticalFilter)

	# Combine the two to get all edges
	result = vertical/4

	return result

def gauss_kernels(size,sigma=1.0):
	## returns a 2d gaussian kernel
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	kernel_sum = kernel.sum()
	if not sum==0:
		kernel = kernel/kernel_sum
	return kernel

def corner_detector(image, imageRGB):

	gx = sobel_hor(image)
	gy = sobel_ver(image)

	Ixx = gx*gx
	Ixy = gx*gy
	Iyy = gy*gy

	kernel = gauss_kernels(3,1)

	Wxx = MyConvolve(Ixx, kernel)
	Wxy = MyConvolve(Ixy, kernel)
	Wyy = MyConvolve(Iyy, kernel)

	k = 0.06
	threshold = 0.1
	stepsize = 10

	responses = np.zeros(Wxx.shape)

	for i in range(0,Wxx.shape[0],stepsize):
		for j in range(0,Wxx.shape[1],stepsize):
			W = np.zeros([2,2])
			W[0,0] = Wxx[i,j]
			W[0,1] = Wxy[i,j]
			W[1,0] = Wxy[i,j]
			W[1,1] = Wyy[i,j]

			detW = W[0,0]*W[1,1]-W[0,1]*W[1,0]
			traceW = W[0,0]+W[1,1]

			response = detW-k*traceW*traceW

			responses[i,j] = response

	maxVal = np.max(responses)

	for i in range(4,Wxx.shape[0]-4):
		for j in range(4,Wxx.shape[1]-4):
			if responses[i,j] >= maxVal*threshold:
				imageRGB[i-4:i+4,j-4,0] = 0
				imageRGB[i-4:i+4,j-4,1] = 0
				imageRGB[i-4:i+4,j-4,2] = 255

				imageRGB[i-4:i+4,j+4,0] = 0
				imageRGB[i-4:i+4,j+4,1] = 0
				imageRGB[i-4:i+4,j+4,2] = 255

				imageRGB[i-4,j-4:j+4,0] = 0
				imageRGB[i-4,j-4:j+4,1] = 0
				imageRGB[i-4,j-4:j+4,2] = 255

				imageRGB[i+4,j-4:j+5,0] = 0
				imageRGB[i+4,j-4:j+5,1] = 0
				imageRGB[i+4,j-4:j+5,2] = 255



	return imageRGB

def componentCoords(image,indicatedLocation):
	# Cropping out a window around the indicated component to find its colour
	sample = image[indicatedLocation[0]-5:indicatedLocation[0]+5,indicatedLocation[1]-5:indicatedLocation[1]+5]
	greenAvg = np.mean(sample[:,:,0])
	blueAvg = np.mean(sample[:,:,1])
	redAvg = np.mean(sample[:,:,2])
	# finding the average colour of the component
	sampleColour = [greenAvg,blueAvg,redAvg]

	# traversing
	visited = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	notSameColour = np.zeros([image.shape[0], image.shape[1]], dtype=bool)

	sumX = 0
	sumY = 0
	total = 0.0
	visited, notSameColour, sumX, sumY, total = traverseOut(image, indicatedLocation[0], indicatedLocation[1], sampleColour, visited, notSameColour, sumX, sumY, total)

	bottomestX = 0
	bottomestY = 0

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if notSameColour[i,j]:
				if i > bottomestX:
					bottomestX = i
					bottomestY = j


	centreX = sumX/total
	centreY = sumY/total

	return centreX, centreY, bottomestX, bottomestY, notSameColour, visited
	
def traverseOut(image,i,j,sampleColour,visited, notSameColour, sumX, sumY, total):
	tolerance = 0.2

	visited[i,j] = True
	sameColour = True

	sampleGreen = sampleColour[0]
	sampleRed = sampleColour[1]
	sampleBlue = sampleColour[2]

	green = image[i,j,0]
	red = image[i,j,1]
	blue = image[i,j,2]

	# Check if the investigated coordinate is of the same colour
	if green < sampleGreen*(1-tolerance) or green > sampleGreen*(1+tolerance):
		sameColour = False

	if red < sampleRed*(1-tolerance) or red > sampleRed*(1+tolerance):
		sameColour = False

	if blue < sampleBlue*(1-tolerance) or blue > sampleBlue*(1+tolerance):
		sameColour = False

	if not sameColour:
		sumX += i
		sumY += j
		total += 1
		notSameColour[i,j] = True
		return visited, notSameColour, sumX, sumY, total

	# If we are still within the same sample, traverse out further
	if sameColour:
		if not visited[i-1,j-1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i-1,j-1,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i-1,j]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i-1,j,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i-1,j+1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i-1,j+1,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i,j+1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i,j+1,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i+1,j+1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i+1,j+1,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i+1,j]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i+1,j,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i+1,j-1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i+1,j-1,sampleColour,visited, notSameColour, sumX, sumY, total)
		if not visited[i,j-1]:
			visited, notSameColour, sumX, sumY, total = traverseOut(image,i,j-1,sampleColour,visited, notSameColour, sumX, sumY, total)

	return visited, notSameColour, sumX, sumY, total

