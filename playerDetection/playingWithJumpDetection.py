from jumpDetection import *

videoNumber = 2
playerNumber = 1

# very video-dependent
# 2_1 = 30
# 5_1 = 
jumpLengthThreshold = 30

playerPositions = np.load("../positionArrays/positions" + str(videoNumber) + "_" + str(playerNumber) + "_centres.npy")
jumps, noJumps = jumpDetection(playerPositions, jumpLengthThreshold)
playerPositions = recomputePositions(playerPositions, jumps)
markPositions(videoNumber,playerNumber,playerPositions,jumps)