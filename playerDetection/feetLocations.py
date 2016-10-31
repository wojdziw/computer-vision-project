import numpy as np
from jumpDetection import *
from annotationFunctions import *

def computeFeetLocations(centres, areas, initialDistance, imageHeight, constantExtrapolation=0):

	feetLocations = centres

	# this has to be much muuuuch smoother
	smoothingFilter = gaussianFilter(30,15)
	areas = smoothArray(areas, smoothingFilter)

	smoothingFilter = [0.2, 0.2, 0.2, 0.2, 0.2]
	areas = smoothArray(areas, smoothingFilter)

	areas = medianFilter(areas,30)
	areas = smoothArray(areas, smoothingFilter)

	initialArea = areas[0]+0.0

	fig1 = plt.figure(1)
	plt.plot(range(len(areas)), areas, 'r-')
	fig1.savefig('areas.png')

	for i, area in enumerate(areas):
	
		if constantExtrapolation>0:
			newDistance = constantExtrapolation
		else:
			newDistance = initialDistance*(area/initialArea)

		feetLocations[i,0] = min(centres[i,0]+newDistance, imageHeight-1)

	return feetLocations

def sparser(array, extent):
	for i in range(len(array)-len(array)%extent):
		array[i] = array[i-i%extent]
		
	return array

def medianFilter(array, extent):
	medians = array

	for i in range(extent/2, len(array)-extent/2):
		medians[i] = np.median(array[i-extent/2: i+extent/2+1])

	return medians


videoNumber = 2
playerNumber = 1

centres = np.load("../positionArrays/feet/positions" + str(videoNumber) + "_" + str(playerNumber) + "_centres.npy")
areas = np.load("../positionArrays/feet/areas" + str(videoNumber) + "_" + str(playerNumber) + ".npy")


imageHeight = 296
initialDistance = 130
feetLocations = computeFeetLocations(centres, areas, initialDistance, imageHeight)

videoNumber = 2
playerNumber = 1

markPositions(videoNumber,playerNumber,feetLocations)



