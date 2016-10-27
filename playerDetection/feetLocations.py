import numpy as np
from jumpDetection import *

def computeFeetLocations(centres, bottoms):

	verticalDistances = bottoms[:,0]-centres[:,0]

	# this has to be much muuuuch smoother
	smoothingFilter = gaussianFilter(10,15)
	verticalDistances = smoothArray(verticalDistances, smoothingFilter)

	fig1 = plt.figure(1)
	plt.plot(range(len(verticalDistances)), verticalDistances, 'r-')
	fig1.savefig('verticalDistance.png')

	return verticalDistances


videoNumber = 2
playerNumber = 1

centres = np.load("../positionArrays/positions" + str(videoNumber) + "_" + str(playerNumber) + "_centres.npy")
bottoms = np.load("../positionArrays/positions" + str(videoNumber) + "_" + str(playerNumber) + "_bottoms.npy")

feetLocations = computeFeetLocations(centres, bottoms)

print feetLocations

