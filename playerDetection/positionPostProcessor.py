from jumpDetection import *
from annotationFunctions import *

videoNumber = 2
# playerNumber = 1
# playerNumber = 2
# playerNumber = 3
playerNumber = 4


jumpLengthThreshold = 53

playerPositions = np.load("../data/playerPos/positions" + str(videoNumber) + "_" + str(playerNumber) + ".npy")

# computing the jumps - uncomment if jumps need to be recomputed from scratch
# jumps, noJumps = jumpDetection(playerPositions, jumpLengthThreshold)
# np.save("../data/jumps/jumps" + str(videoNumber) + "_" + str(playerNumber) + ".npy", jumps)

# just loading the jumps from a file
jumps = np.load("../data/jumps/jumps" + str(videoNumber) + "_" + str(playerNumber) + ".npy")
areas = np.load("../data/areas/areas" + str(videoNumber) + "_" + str(playerNumber) + ".npy")

extrapolation = 50
feetPositions = recomputePositions(playerPositions, jumps, extrapolation) #, areas) # uncomment if area scaling for extrapolation needed

# uncomment if want to overwrite feet positions
# np.save("../data/rawFeetPositions/feet" + str(videoNumber) + "_" + str(playerNumber), feetPositions)

markPositions(videoNumber,str(playerNumber),feetPositions,jumps)

'''
PARAMS:

videoNo_player - jumps detected? - extrapolation done?

1_greenBack - YES - YES
1_1
	- threshold 20
	- manually tweaked it not to include weird jump at the end
	- constant extrapolation: 50

1_greenFront - YES - YES
1_2
	- threshold 25
	- constant extrapolation: 25

1_whiteLeft - YES - YES
1_3
	- threshold 15
	- constant extrapolation: 15

1_whiteRight - YES - YES
1_4
	- weird jumps again
	- constant extrapolation: 15

2_greenBack - YES - YES
2_1
	- threshold 50
	- constant extrapolation: can't really
	- initial extrapolation - 50

2_greenFront - YES - YES
2_2
NOISY!
	- threshold 50
	- constant extrapolation: 60

2_whiteLeft - YES - YES
2_3
	- threshold 50
	- constant extrapolation: 30

2_whiteRight - YES - YES
2_4
	- threshold - 10
	- detects few first frames as jump because of noise
	- constant extrapolation: 30

3_greenDown - YES - YES
3_1
	- threshold - 55
	- constant extrapolation: 40

3_greenUp - YES - YES
3_2
	- threshold - 55
	- constant extrapolation: 40

3_whiteLeft - YES - YES
3_3
	- threshold - 50
	- constant extrapolation: 35

3_whiteRight - YES - YES
3_4
	- threshold - 5
	- misses the first landing
	- constant extrapolation: 35

4_greenDown - YES - YES
4_1
	- threshold - 20
	- constant extrapolation: 35
	- maybe apart from the last few frames when he lies down

4_greenUp - YES - YES
4_2
	- threshold - 60
	- constant extrapolation: 30

4_whiteLeft - YES - YES
4_3
	- threshold - 0
	- doesn't detect the tiny jump
	- constant extrapolation: 30

4_whiteRight - YES - YES
4_4
	- threshold - 0
	- doesn't detect the tiny jump
	- constant extrapolation: 30

5_redBack - YES - YES
5_1
	- threshold - 53
	- detecting 4 - 2nd and 4th are weird
	- manually tweaked it - 4th by precise threshold, second by changing array
	- constant extrapolation: can't do, do areas
	- initial-extrapolation: 45
	- working half decently, super noisy data

5_redFront - YES - YES
5_2
SUPER NOISY!
	- threshold - 20
	- detecting 4 - just the first one is weird
	- first one and last one weird due to changing frames
	- manually tweaked it
	- constant extrapolation: 30

6_redDown - YES - YES
6_1
	- threshold - 40
	- doesn't detect a tiny jump
	- constant extrapolation: 30 

6_redUp - YES - YES
6_2
	- threshold - 10
	- detects everything apart from the jump
	- constant extrapolation: 50

6_whiteLeft - YES - YES
6_3
	- threshold - 35
	- constant extrapolation: 50

6_whiteRight - YES - YES
6_4
	- threshold - 55
	- constant extrapolation: 55

7_redLeft - YES - YES
7_1
	- threshold - 25
	- constant extrapolation: 60

7_redRight - YES - YES
7_2
	- threshold
	- constant extrapolation: 30

7_whiteDown - YES - YES
7_3
	- threshold - 60
	- doesn't detect a small jump at the end
	- constant extrapolation: 50

7_whiteUp - YES - YES
7_4
	- threshold - 46
	- constant extrapolation: 50


'''