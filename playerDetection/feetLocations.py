import numpy as np
from jumpDetection import *
from annotationFunctions import *

def computeFeetLocations(centres, areas, initialDistance, imageHeight):

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
	
		newDistance = initialDistance*(area/initialArea)

		feetLocations[i,0] = min(centres[i,0]+50, imageHeight-1)

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

centres = np.load("../positionArrays/positions" + str(videoNumber) + "_" + str(playerNumber) + "_centres.npy")
bottoms = np.load("../positionArrays/positions" + str(videoNumber) + "_" + str(playerNumber) + "_bottoms.npy")
areas = np.load("../positionArrays/areas" + str(videoNumber) + "_" + str(playerNumber) + ".npy")

imageHeight = 296
initialDistance = 130
feetLocations = computeFeetLocations(centres, areas, initialDistance, imageHeight)

videoNumber = 2
playerNumber = 1

markPositions(videoNumber,playerNumber,feetLocations)



