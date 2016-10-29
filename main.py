import cv2
import sys
sys.path.append("playerDetection")
from playerDetection import *
from pickStartPoint import *
import numpy as np

##########################################################################

# Choose which video to process 
if (sys.argv < 1):
	print 'Usage: main.py <VideoIndex>'
else:
	try:
		videoNumber = str(sys.argv[1])
	except:
		print 'Usage: main.py <VideoIndex (int)>'

startFr = 36

# Read first frame players are seen for picking starting points
video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

for i in range(startFr):
	_,image = video.read()

#cv2.imwrite('vid'+str(videoNumber)+'.jpg', image)

# Let user pick tracking point for the 4 players 
x, y = indicateLocation(image)
print(y, x)
player1Loc = [y,x]
#player1Loc = [69, 102] #Vid1: WhiteLeft
#player1Loc = [60, 174] #Vid1, WhiteRight
#player1Loc = [62, 394] #Vid2: WhiteLeft, [66,390]=shorts [62,394]=shirt [73,384]=leg
#player1Loc = [75, 485] #Vid2, WhiteRight, shorts [76, 481] = leg


# Track players, returns foot position for every frame and total number of jumps
player1positions, areas = playerDetection(videoNumber, player1Loc, startFr)

np.save('positionArrays/adam/positions' + str(videoNumber) +'_.npy', player1positions)
np.save('positionArrays/adam/areas' + str(videoNumber) +'_.npy', areas)