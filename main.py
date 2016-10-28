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

# Read first frame for picking starting points
video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')
_,image = video.read()
#cv2.imwrite('vid'+str(videoNumber)+'.jpg', image)

# Let user pick tracking point for the 4 players 
x, y = indicateLocation(image)
print(y, x)
player1Loc = [y,x]
#player1Loc = [69, 102] #Vid1: WhiteLeft
#player1Loc = [60, 174] #Vid1, WhiteRight


# Track players, returns foot position for every frame and total number of jumps
player1positions = playerDetection(videoNumber, player1Loc)

np.save('positionArrays/adam/positions' + str(videoNumber) +'_greenBack.npy', player1positions)
