import cv2
import numpy as np
import Queue

# Set global constants, see README.md for usage
PATCHSIZE = 2
GRADDIV = 10
THRESHOLD = 7
RAYRANGE = 30

STARTPT_TH = 4
NEWCENTRE_TH = 1.5
TRAVERSE_TH = 2


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

def rgbToHue(colour):
	hue = 0

	b = colour[0]/255.0
	g = colour[1]/255.0
	r = colour[2]/255.0
	Cmax = max(b,g,r)
	Cmin = min(b,g,r)
	delta = Cmax-Cmin

	if delta == 0:
		hue = 0
	elif Cmax == r:
		hue = 60*(((g-b)/delta)%6)
	elif Cmax == g:
		hue = 60*(((b-r)/delta)+2)
	elif Cmax == b:
		hue = 60*(((r-g)/delta)+4)

	return hue

def rgbToSat(colour):
	sat = 0

	b = colour[0]/255.0
	g = colour[1]/255.0
	r = colour[2]/255.0
	Cmax = max(b,g,r)
	Cmin = min(b,g,r)
	delta = Cmax-Cmin

	if Cmax == 0:
		s = 0
	else:
		s = delta/Cmax

	return sat			

def rgbToBri(colour):
	bri = 0

	b = colour[0]/255.0
	g = colour[1]/255.0
	r = colour[2]/255.0
	Cmax = max(b,g,r)
	Cmin = min(b,g,r)
	delta = Cmax-Cmin

	bri = Cmax

	return bri

# Return average BGR colours from 2*PATCHSIZE around indicated location
def patchColour(image,indicatedLocation):
	size = PATCHSIZE
	# Cropping out a window around the indicated component to find its colour	
	sample = image[indicatedLocation[0]-size:indicatedLocation[0]+size,indicatedLocation[1]-size:indicatedLocation[1]+size]
	greenAvg = np.mean(sample[:,:,0])
	blueAvg = np.mean(sample[:,:,1])
	redAvg = np.mean(sample[:,:,2])

	return [greenAvg,blueAvg,redAvg]

def singleColour(image, indicatedLocation):

	G = image[indicatedLocation[0], indicatedLocation[1], 0]
	B = image[indicatedLocation[0], indicatedLocation[1], 1]
	R = image[indicatedLocation[0], indicatedLocation[1], 2]

	return [G, B, R]

def hueDistance(colour1,colour2):
	colour1Hue = rgbToHue(colour1)
	colour2Hue = rgbToHue(colour2)

	hueDist = min(abs(colour1Hue-colour2Hue), 360-abs(colour1Hue-colour2Hue));

	return hueDist

def rgbDistance(colour1,colour2):
	rgbDist = np.linalg.norm([colour1[0]-colour2[0], colour1[1]-colour2[1], colour1[2]-colour2[2]])

	return rgbDist

# Convert RGB to Lab using RGB -> XYZ -> Lab
def rgb2lab(rgb):
    	
	## Convert RGB to XYZ
	num = 0
	RGB = [0, 0, 0]

	for value in rgb :
	   value = float(value) / 255

	   if value > 0.04045 :
	       value = np.power(((value+0.055)/1.055), 2.4)
	   else :
	       value = value / 12.92

	   RGB[num] = value * 100
	   num = num + 1

	XYZ = [0, 0, 0,]

	X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
	Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
	Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505

	XYZ[0] = round(X, 4)
	XYZ[1] = round(Y, 4)
	XYZ[2] = round(Z, 4)

	## Convert XYZ to Lab

	XYZ[0] = float(XYZ[0]) / 95.047		# ref_X =  95.047   Observer= 2 degrees, Illuminant= D65
	XYZ[1] = float(XYZ[1]) / 100.0		# ref_Y = 100.000
	XYZ[2] = float(XYZ[2]) / 108.883	# ref_Z = 108.883

	num = 0
	for value in XYZ :

	   if value > 0.008856 :
	       value = np.power(value, float(1)/3)
	   else :
	       value = (7.787*value) + (16/116)

	   XYZ[num] = value
	   num = num + 1

	Lab = [0, 0, 0]

	L = (116 * XYZ[1]) - 16
	a = 500 * (XYZ[0] - XYZ[1])
	b = 200 * (XYZ[1] - XYZ[2])

	Lab [ 0 ] = round( L, 4 )
	Lab [ 1 ] = round( a, 4 )
	Lab [ 2 ] = round( b, 4 )

	return Lab


# Calculate the difference between two Lab colours, deltaE1976
def deltaE(lab1, lab2):

	L1, a1, b1 = lab1[0], lab1[1], lab1[2]
	L2, a2, b2 = lab2[0], lab2[1], lab2[2]

	diff = np.sqrt(np.power(L1-L2, 2) + np.power(a1-a2, 2) + np.power(b1-b2, 2))

	return diff

# Find distance between two Lab colours using deltaE
def labDistance(colour1,colour2):
    # Convert BGR to RGB
	col1 = np.zeros(3)
	col2 = np.zeros(3)
	col1[0] = colour1[2]	
	col1[1] = colour1[1]
	col1[2] = colour1[0]
	col2[0] = colour2[2]
	col2[1] = colour2[1]
	col2[2] = colour2[0]

	lab1 = rgb2lab(col1)
	lab2 = rgb2lab(col2)

	labDist = deltaE(lab1, lab2)

	return labDist

def findColourDistance(colour1,colour2):
	#colour1Hue = rgbToHue(colour1)
	#colour2Hue = rgbToHue(colour2)

	#hueDist = hueDistance(colour1, colour2)
	#rgbDist = rgbDistance(colour1, colour2)
	labDist = labDistance(colour1, colour2)

	return labDist

# Make starting colour a bit more similar to ending colour (factor 1/GradientDivide)
# TODO: Change so this is done in Lab instead? Now it's only BGR
def addGradient(startColour, endColour):
	newStart = startColour
	diff = np.zeros(3)
	diff[0] = endColour[0] - startColour[0]
	diff[1] = endColour[1] - startColour[1]
	diff[2] = endColour[2] - startColour[2]
	newStart = newStart + diff/GRADDIV
	
	return newStart

# Perform edge detection and return as boolean array
def findCloseEdges(image, indicatedLocation):
	region = image[indicatedLocation[0]-50:indicatedLocation[0]+50,indicatedLocation[1]-50:indicatedLocation[1]+50]
	regionGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
	regionEdges = sobel(regionGray)
	regionEdges[regionEdges<40] = 0
	regionEdges[regionEdges>=40] = 1
	
	regionEdges.astype(bool)

	imageEdges = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	imageEdges[indicatedLocation[0]-50:indicatedLocation[0]+50,indicatedLocation[1]-50:indicatedLocation[1]+50] = regionEdges
	
	return imageEdges

# Find the nearest pixels that's a good match for previous colour
def findNewCentre(image, indicatedLocation, previousColour, startColour, threshold):

	sampleColour = patchColour(image, indicatedLocation)
	#sampleColour = singleColour(image, indicatedLocation)

	colourDistance = findColourDistance(sampleColour, previousColour)

	# Set up parameters
	rayRange = RAYRANGE
	startPtThreshold = STARTPT_TH*threshold
	threshold = NEWCENTRE_TH*threshold
	found = 0

	directions = 8
	patch = np.zeros([directions, 2], int)
	newPatchColour = np.zeros([directions, 3], np.uint8)
	newColourDistance = np.zeros(directions)
	startColDist = np.zeros(directions)

	# TODO: Include edge detection somehow?
	#regionEdges = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	#regionEdges = findCloseEdges(image, indicatedLocation)

	print("Colour distance bad")
	for i in range(1,rayRange):

		# Extend in 8 directions
		patch[0] = [indicatedLocation[0]-2*i,indicatedLocation[1]]
		patch[1] = [indicatedLocation[0]-2*i,indicatedLocation[1]+2*i]
		patch[2] = [indicatedLocation[0],indicatedLocation[1]+2*i]
		patch[3] = [indicatedLocation[0]+2*i,indicatedLocation[1]+2*i]
		patch[4] = [indicatedLocation[0]+2*i,indicatedLocation[1]]
		patch[5] = [indicatedLocation[0]+2*i,indicatedLocation[1]-2*i]
		patch[6] = [indicatedLocation[0],indicatedLocation[1]-2*i]
		patch[7] = [indicatedLocation[0]-2*i,indicatedLocation[1]-2*i]

		for i in range(directions):
			# Take new colour sample
			newPatchColour[i] = patchColour(image, patch[i])
			#patchColour[i] = singleColour(image, patch[i])

			# Find distance to previous colour
			newColourDistance[i] = findColourDistance(newPatchColour[i],previousColour)

			# Find distance to starting colour
			startColDist[i] = findColourDistance(newPatchColour[i],startColour)

			# Save only if it's below threshold for previous AND starting colour and if it's the best match so far!
			if newColourDistance[i] < threshold and startColDist[i] < startPtThreshold and newColourDistance[i] < colourDistance:
				colourDistance = newColourDistance[i]
				indicatedLocation = patch[i]
				found = 1

		# If we found something this iteration, make starting colour a bit more similar and return the best result
		if(found == 1):
			endColour = patchColour(image, indicatedLocation)
			startColour = addGradient(startColour, endColour)
			return indicatedLocation, found, startColour

	# Nothing good was found, return initial values
	return indicatedLocation, found, startColour


def findBottomest(image,notSameColour):
	bottomestR = 0
	bottomestC = 0

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if notSameColour[i,j]:
				if i > bottomestR:
					bottomestR = i
					bottomestC = j

	return int(bottomestR), int(bottomestC)


def componentCoords(image, indicatedLocation, previousColour, startColour):
	# Finding the average colour of the component
	sampleColour = patchColour(image, indicatedLocation)
	#sampleColour = singleColour(image, indicatedLocation)

	# Calculating how different the previous and current colour patches are
	colourDistance = findColourDistance(sampleColour, previousColour)

	# Constant for how different colours we should accept (Lab colour space)
	threshold = THRESHOLD

	# If the colour patches are very different - find a better match in surrounding
	if colourDistance > threshold:
		indicatedLocation, found, startColour = findNewCentre(image, indicatedLocation, previousColour, startColour, threshold)
		
		if(found == 1):    		
			print "Found new"
			sampleColour = patchColour(image, indicatedLocation)
			#sampleColour = singleColour(image, indicatedLocation)
		
		else:	#If no new centre = we are lost => Stay put	
			# TODO: handle player disappearence outside frame different? Move in direction of previous moveVec?
			print "Nothing found"
			centreR = indicatedLocation[0]
			centreC = indicatedLocation[1]
			bottomestR = indicatedLocation[0]
			bottomestC = indicatedLocation[1]
			visited = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
			return centreR, centreC, bottomestR, bottomestC, visited, previousColour

	# Finding the edges
	imageEdges = findCloseEdges(image, indicatedLocation)

	# Traversing
	visited = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
	toVisit = Queue.Queue()
	notSameColour = np.zeros([image.shape[0], image.shape[1]], dtype=bool)

	toVisit.put(indicatedLocation)

	sumR = 0.0
	sumC = 0.0
	total = 0.0

	visited[indicatedLocation[0],indicatedLocation[1]] = True

	while not toVisit.empty():
		currentR,currentC,visited,notSameColour,toVisit = traverseOut(image,sampleColour,visited,toVisit,notSameColour,imageEdges,threshold)
		sumR += currentR
		sumC += currentC
		total += 1

	bottomestR, bottomestC = findBottomest(image, notSameColour)

	# Centre of the component is assumed to be at the average of all of its coordinates
	centreR = int(sumR/total)
	centreC = int(sumC/total)

	return centreR, centreC, bottomestR, bottomestC, visited, sampleColour

# Traverse all pixels with similar colour
def traverseOut(image,sampleColour,visited,toVisit,notSameColour,imageEdges, threshold):

	[i,j] = toVisit.get()

	sameColour = True

	currentColour = image[i,j]

	# Comparing colour based on Euclidean distance (deltaE1976, Lab colours)
	threshold = threshold+TRAVERSE_TH
	colourDistance = findColourDistance(sampleColour, currentColour)
	sameColour = colourDistance < threshold

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

def calculateArea(visited):
	area = 0

	for i in range(visited.shape[0]):
		for j in range(visited.shape[1]):
			if visited[i,j]:
				area += 1

	return area