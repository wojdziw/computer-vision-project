import cv2
import numpy as np
import Queue

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

def patchColour(image,indicatedLocation):
	# Cropping out a window around the indicated component to find its colour	
	sample = image[indicatedLocation[0]-5:indicatedLocation[0]+5,indicatedLocation[1]-5:indicatedLocation[1]+5]
	greenAvg = np.mean(sample[:,:,0])
	blueAvg = np.mean(sample[:,:,1])
	redAvg = np.mean(sample[:,:,2])

	return [greenAvg,blueAvg,redAvg]

def colourDistance(colour1,colour2):
	return np.linalg.norm([colour1[0]-colour2[0], colour1[1]-colour2[1], colour1[2]-colour2[2]])

def componentCoords(image,indicatedLocation,previousColour):
	# Finding the average colour of the component
	sampleColour = patchColour(image,indicatedLocation)
	a = indicatedLocation

	# Checking how far the previous colour patch and the new one is
	sampleColourNorm = np.linalg.norm(sampleColour)
	previousColourNorm = np.linalg.norm(previousColour)

	# Calculating different measures of how different the previous and current colour patches are
	normDifference = int(100*np.abs(previousColourNorm-sampleColourNorm)/previousColourNorm)
	colourDistance = np.linalg.norm([sampleColour[0]-previousColour[0], sampleColour[1]-previousColour[1], sampleColour[2]-previousColour[2]])

	print("normDifference: " + str(normDifference)+"%")
	print("colour distance: " + str(colourDistance))

	# CORRECTING THE COMPONENT CENTRAL LOCATION

	# If the colour patches are very different - start looking for a new starting point
	# Trying to find the point in the neighborhood that is more similar to the previous patch
	threshold = 20

	# SOMETIMES THESE DIRECTIONS AREN'T ENOUGH!!!!
	if colourDistance > threshold:
		print("Colour distance bad")
		for i in range(1,10):

			nPatch = [indicatedLocation[0]-2*i,indicatedLocation[1]]
			nePatch = [indicatedLocation[0]-2*i,indicatedLocation[1]+2*i]
			ePatch = [indicatedLocation[0],indicatedLocation[1]+2*i]
			sePatch = [indicatedLocation[0]+2*i,indicatedLocation[1]+2*i]
			sPatch = [indicatedLocation[0]+2*i,indicatedLocation[1]]
			swPatch = [indicatedLocation[0]+2*i,indicatedLocation[1]-2*i]
			wPatch = [indicatedLocation[0],indicatedLocation[1]-2*i]
			nwPatch = [indicatedLocation[0]-2*i,indicatedLocation[1]-2*i]

			nPatchColour = patchColour(image, nPatch)
			nePatchColour = patchColour(image, nePatch)
			ePatchColour = patchColour(image, ePatch)
			sePatchColour = patchColour(image, sePatch)
			sPatchColour = patchColour(image, sPatch)
			swPatchColour = patchColour(image, swPatch)
			wPatchColour = patchColour(image, wPatch)
			nwPatchColour = patchColour(image, nwPatch)

			nColourDistance = np.linalg.norm([nPatchColour[0]-previousColour[0], nPatchColour[1]-previousColour[1], nPatchColour[2]-previousColour[2]])
			neColourDistance = np.linalg.norm([nePatchColour[0]-previousColour[0], nePatchColour[1]-previousColour[1], nePatchColour[2]-previousColour[2]])
			eColourDistance = np.linalg.norm([ePatchColour[0]-previousColour[0], ePatchColour[1]-previousColour[1], ePatchColour[2]-previousColour[2]])
			seColourDistance = np.linalg.norm([sePatchColour[0]-previousColour[0], sePatchColour[1]-previousColour[1], sePatchColour[2]-previousColour[2]])
			sColourDistance = np.linalg.norm([sPatchColour[0]-previousColour[0], sPatchColour[1]-previousColour[1], sPatchColour[2]-previousColour[2]])
			swColourDistance = np.linalg.norm([swPatchColour[0]-previousColour[0], swPatchColour[1]-previousColour[1], swPatchColour[2]-previousColour[2]])
			wColourDistance = np.linalg.norm([wPatchColour[0]-previousColour[0], wPatchColour[1]-previousColour[1], wPatchColour[2]-previousColour[2]])
			nwColourDistance = np.linalg.norm([nwPatchColour[0]-previousColour[0], nwPatchColour[1]-previousColour[1], nwPatchColour[2]-previousColour[2]])

			if nColourDistance < colourDistance:
				colourDistance = nColourDistance
				indicatedLocation = nPatch
			if neColourDistance < colourDistance:
				colourDistance = neColourDistance
				indicatedLocation = nePatch
			if eColourDistance < colourDistance:
				colourDistance = eColourDistance
				indicatedLocation = ePatch
			if seColourDistance < colourDistance:
				colourDistance = seColourDistance
				indicatedLocation = sePatch
			if sColourDistance < colourDistance:
				colourDistance = sColourDistance
				indicatedLocation = sPatch
			if swColourDistance < colourDistance:
				colourDistance = swColourDistance
				indicatedLocation = swPatch
			if wColourDistance < colourDistance:
				colourDistance = wColourDistance
				indicatedLocation = wPatch
			if nwColourDistance < colourDistance:
				colourDistance = nwColourDistance
				indicatedLocation = nwPatch

	sampleColour = patchColour(image, indicatedLocation)

	# Finding the edges
	region = image[indicatedLocation[0]-50:indicatedLocation[0]+50,indicatedLocation[1]-50:indicatedLocation[1]+50]
	regionGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
	regionEdges = sobel(regionGray)
	regionEdges[regionEdges<40] = 0
	regionEdges[regionEdges>=40] = 1
	
	regionEdges.astype(bool)

	imageEdges = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	imageEdges[indicatedLocation[0]-50:indicatedLocation[0]+50,indicatedLocation[1]-50:indicatedLocation[1]+50] = regionEdges

	# Traversing
	visited = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	toVisit = Queue.Queue()
	notSameColour = np.zeros([image.shape[0], image.shape[1]], dtype=bool)

	toVisit.put(indicatedLocation)

	sumX = 0
	sumY = 0
	total = 0.0

	visited[indicatedLocation[0],indicatedLocation[1]] = True
	
	while not toVisit.empty():
		currentX,currentY,visited,notSameColour,toVisit = traverseOut(image,sampleColour,visited,toVisit,notSameColour,imageEdges)
		sumX += currentX
		sumY += currentY
		total += 1

	bottomestX = 0
	bottomestY = 0

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if notSameColour[i,j]:
				if i > bottomestX:
					bottomestX = i
					bottomestY = j

	# Centre of the component is assumed to be at the average of all of its coordinates
	centreX = sumX/total
	centreY = sumY/total

	centreX = int(centreX)
	centreY = int(centreY)
	bottomestX = int(bottomestX)
	bottomestY = int(bottomestY)

	return centreX, centreY, bottomestX, bottomestY, notSameColour, visited, sampleColour
	
def traverseOut(image,sampleColour,visited,toVisit,notSameColour,imageEdges):
	tolerance = 0.2

	[i,j] = toVisit.get()

	sameColour = True

	[sampleGreen, sampleRed, sampleBlue] = sampleColour
	[green, red, blue] = image[i,j]

	# Check if the investigated coordinate is of the same colour
	# MAYBE CHANGE THAT TO THE EUCLIDEAN DISTANCE??
	if green < sampleGreen*(1-tolerance) or green > sampleGreen*(1+tolerance):
		sameColour = False

	if red < sampleRed*(1-tolerance) or red > sampleRed*(1+tolerance):
		sameColour = False

	if blue < sampleBlue*(1-tolerance) or blue > sampleBlue*(1+tolerance):
		sameColour = False

	if not sameColour:
		notSameColour[i,j] = True
		return i, j, visited, notSameColour, toVisit

	# If we are still within the same sample, traverse out further
	if sameColour:
		if not visited[i-1,j-1] and not imageEdges[i-1,j-1]:
			visited[i-1,j-1] = True
			toVisit.put([i-1,j-1])
		if not visited[i-1,j] and not imageEdges[i-1,j]:
			visited[i-1,j] = True
			toVisit.put([i-1,j])
		if not visited[i-1,j+1] and not imageEdges[i-1,j+1]:
			visited[i-1,j+1] = True
			toVisit.put([i-1,j+1])
		if not visited[i,j+1] and not imageEdges[i,j-1]:
			visited[i,j+1] = True
			toVisit.put([i,j+1])
		if not visited[i+1,j+1] and not imageEdges[i+1,j+1]:
			visited[i+1,j+1] = True
			toVisit.put([i+1,j+1])
		if not visited[i+1,j] and not imageEdges[i+1,j]:
			visited[i+1,j] = True
			toVisit.put([i+1,j])
		if not visited[i+1,j-1] and not imageEdges[i+1,j-1]:
			visited[i+1,j-1] = True
			toVisit.put([i+1,j-1])
		if not visited[i,j-1] and not imageEdges[i,j-1]:
			visited[i,j-1] = True
			toVisit.put([i,j-1])

	return i, j, visited, notSameColour, toVisit

