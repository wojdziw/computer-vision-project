import numpy as np
import matplotlib.pyplot as plt
from annotationFunctions import *

def recomputePositions(playerPositions, jumps):

	smoothingFilter = gaussianFilter(2,5)
	playerPositions[:,0] = smoothArray(playerPositions[:,0], smoothingFilter)
	playerPositions[:,1] = smoothArray(playerPositions[:,1], smoothingFilter)

	latestBeforeJump = 0
	for i in range(playerPositions.shape[0]):
		if not jumps[i]:
			latestBeforeJump = playerPositions[i,0]
		else:
			playerPositions[i,0] = latestBeforeJump

	return playerPositions

def jumpDetection(playerPositions):

	jumpFrames = np.zeros(playerPositions.shape[0], bool)

	smoothingFilter = gaussianFilter(10,15)

	smoothedPositions = np.zeros(playerPositions.shape)
	smoothedPositions[:,0] = smoothArray(playerPositions[:,0], smoothingFilter)

	derivatives = fakeDerivative(smoothedPositions[:,0],2)
	smoothedDerivatives = smoothArray(derivatives, smoothingFilter)

	peakLengths = computeLongPeaks(smoothedDerivatives)

	for i, peakLength in enumerate(peakLengths):
		if peakLength>0:
			jumpFrames[i] = True

	noJumps = 0
	for i in range(len(jumpFrames)-1):
		if jumpFrames[i] and not jumpFrames[i+1]:
			noJumps += 1

	fig1 = plt.figure(1)
	plt.plot(range(playerPositions.shape[0]), smoothedPositions[:,0], 'r-')
	fig1.savefig('smoothed.png')

	fig2 = plt.figure(2)
	plt.plot(range(playerPositions.shape[0]), playerPositions[:,0], 'r-')
	fig2.savefig('notSmoothed.png')

	fig3 = plt.figure(3)
	plt.plot(range(playerPositions.shape[0]), smoothedDerivatives, 'r-')
	fig3.savefig('derivative.png')

	fig4 = plt.figure(4)
	plt.plot(range(playerPositions.shape[0]), peakLengths, 'r-')
	fig4.savefig('lengths.png')

	return jumpFrames, noJumps

def smoothArray(array, smoothingFilter):

	filterSize = len(smoothingFilter)
	appendSize = len(smoothingFilter)/2

	# don't pad with zeros!! That introduces sharp derivatives
	appendedArray = array
	appendedArray = np.append(appendedArray, np.full([appendSize], array[len(array)-1]))
	appendedArray = np.append(np.full([appendSize], array[0]), appendedArray)

	smoothedArray = appendedArray

	for i in range(appendSize, len(smoothedArray)-appendSize):
		extract = appendedArray[i-appendSize: i+appendSize+1]
		smoothedArray[i] = sum(extract*smoothingFilter)

	smoothedArray = smoothedArray[appendSize:len(smoothedArray)-appendSize]

	return smoothedArray

def fakeDerivative(array, step):
	derivatives = np.zeros(array.shape)

	for i in range(len(array)-step):
		derivatives[i] = np.abs(array[i]-array[i+step])

	return derivatives

def gaussianFilter(stdev, extent):
	stdev += 0.0
	filter = np.zeros(extent)
	centre = extent/2
	for i in range(extent):
		filter[i] = (1/(np.sqrt(2*np.pi)*stdev))*np.exp(-(i-centre)*(i-centre)*0.5*(1/stdev)*(1/stdev))

	total = sum(filter)

	for i in range(len(filter)):
		filter[i] = filter[i]/total

	return filter

def computePeakLengths(array):
	lengths = np.zeros(len(array))

	threshold = 2
	thresholdedArray = [(0,i)[i > threshold] for i in array]

	for i, point in enumerate(thresholdedArray):
		if point!=0 and i!=1:
			lengths[i] = lengths[i-1]+1

	for i in range(len(array)-2, -1, -1):
		if lengths[i+1] != 0 and lengths[i] != 0:
			lengths[i] = lengths[i+1]

	return lengths

def computeLongPeaks(array):

	lengths = computePeakLengths(array)

	lengthThreshold = 15
	lengths = [(0,i)[i > lengthThreshold] for i in lengths]

	localMaxima = np.zeros(len(array))
	extent = 10

	# if there is a gap between peaks then fill it in
	for i in range(extent,len(array)-extent):
		if lengths[i] == 0 and lengths[i-extent]>0 and lengths[i+extent]>0:
			lengths[i-extent:i+extent+1] = np.full([2*extent+1], max(lengths[i-extent:i+extent+1]))

	return lengths

playerPositions = np.load("positions.npy")
jumps, noJumps = jumpDetection(playerPositions)
playerPositions = recomputePositions(playerPositions, jumps)
markPositions(5,playerPositions)