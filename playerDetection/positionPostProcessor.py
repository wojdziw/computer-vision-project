from jumpDetection import *
from annotationFunctions import *

videoNumber = 2
playerName = "greenBack"
# playerName = "greenFront"
# playerName = "whiteLeft"
# playerName = "whiteRight"
# playerName = "greenDown"
# playerName = "greenUp"
# playerName = "redBack"
# playerName = "redFront"
# playerName = "redDown"
# playerName = "redUp"
# playerName = "redLeft"
# playerName = "redRight"
# playerName = "whiteDown"
# playerName = "whiteUp"

jumpLengthThreshold = 53

playerPositions = np.load("../positionArrays/positions/positions" + str(videoNumber) + "_" + playerName + ".npy")

# computing the jumps
# jumps, noJumps = jumpDetection(playerPositions, jumpLengthThreshold)
# np.save("../positionArrays/jumps/jumps" + str(videoNumber) + "_" + playerName + ".npy", jumps)

# just loading the jumps from a file
jumps = np.load("../positionArrays/jumps/jumps" + str(videoNumber) + "_" + playerName + ".npy")
areas = np.load("../positionArrays/areas/areas" + str(videoNumber) + "_" + playerName + ".npy")

extrapolation = 50
feetPositions = recomputePositions(playerPositions, jumps, extrapolation) #, areas) # uncomment if area scaling for extrapolation needed

# uncomment if want to overwrite feet positions
# np.save("../positionArrays/feetPositions/feet" + str(videoNumber) + "_" + playerName, feetPositions)

markPositions(videoNumber,playerName,feetPositions,jumps)

'''
PROBLEMS & PARAMS:

videoNo_player - jumps detected? - extrapolation done?

1_greenBack - YES - YES
	- threshold 20
	- manually tweaked it not to include weird jump at the end
	- constant extrapolation: 50
	- at the end maybe try areas?

1_greenFront - YES - YES
	- threshold 25
	- constant extrapolation: 25

1_whiteLeft - YES - YES
	- threshold 15
	- constant extrapolation: 15

1_whiteRight - YES - YES
	- weird jumps again
	- constant extrapolation: 15

2_greenBack - YES - YES
	- threshold 50
	- constant extrapolation: can't really
	- initial extrapolation - 50

2_greenFront - YES - YES
NOISY!
	- threshold 50
	- constant extrapolation: 60

2_whiteLeft - YES - YES
	- threshold 50
	- constant extrapolation: 30

2_whiteRight - YES - YES
	- threshold - 10
	- detects few first frames as jump because of noise
	- constant extrapolation: 30

3_greenDown - YES - YES
	- threshold - 55
	- constant extrapolation: 40

3_greenUp - YES - YES
	- threshold - 55
	- constant extrapolation: 40

3_whiteLeft - YES - YES
	- threshold - 50
	- constant extrapolation: 35

3_whiteRight - YES - YES
	- threshold - 5
	- misses the first landing
	- constant extrapolation: 35

4_greenDown - YES - YES
	- threshold - 20
	- constant extrapolation: 35
	- maybe apart from the last few frames when he lies down

4_greenUp - YES - YES
	- threshold - 60
	- constant extrapolation: 30

4_whiteLeft - YES - YES
	- threshold - 0
	- doesn't detect the tiny jump
	- constant extrapolation: 30

4_whiteRight - YES - YES
	- threshold - 0
	- doesn't detect the tiny jump
	- constant extrapolation: 30

5_redBack - YES - YES
	- threshold - 53
	- detecting 4 - 2nd and 4th are weird
	- manually tweaked it - 4th by precise threshold, second by changing array
	- constant extrapolation: can't do, do areas
	- initial-extrapolation: 45
	- working half decently, super noisy data

5_redFront - YES - YES
SUPER NOISY!
	- threshold - 20
	- detecting 4 - just the first one is weird
	- first one and last one weird due to changing frames
	- manually tweaked it
	- constant extrapolation: 30

6_redDown - YES - YES
	- threshold - 40
	- doesn't detect a tiny jump
	- constant extrapolation: 30 

6_redUp - YES - YES
	- threshold - 10
	- detects everything apart from the jump
	- constant extrapolation: 50

6_whiteLeft - YES - YES
	- threshold - 35
	- constant extrapolation: 50

6_whiteRight - YES - YES
	- threshold - 55
	- constant extrapolation: 55

7_redLeft - YES - YES
	- threshold - 25
	- constant extrapolation: 60

7_redRight - YES - YES
	- threshold
	- constant extrapolation: 30

7_whiteDown - YES - YES
	- threshold - 60
	- doesn't detect a small jump at the end
	- constant extrapolation: 50

7_whiteUp - YES - YES
	- threshold - 46
	- constant extrapolation: 50


'''