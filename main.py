import cv2
import sys
sys.path.append("playerDetection")
from playerDetection import *
from pickStartPoint import *


# Choose which video to process 
if (sys.argv < 1):
	print 'Usage: main.py <VideoIndex>'
else:
	try:
		videoNumber = str(sys.argv[1])
	except:
		print 'Usage: main.py <VideoIndex (int)>'

# Read first frame for picking starting points
video = cv2.VideoCapture('beach-volleyball-films/beachVolleyball'+str(videoNumber)+'.mov')
_,image = video.read()

# Let user pick tracking point for the 4 players 
x, y = indicateLocation(image)
print(y, x)
player1Loc = [y,x]
'''
x, y = indicateLocation(image)
player2Loc = [y,x]
x, y = indicateLocation(image)
player3Loc = [y,x]
x, y = indicateLocation(image)
player4Loc = [y,x]
'''

# Track players for the whole video
player1positions = playerDetection(videoNumber, player1Loc)
# player2positions = playerDetection(videoNumber, player2Loc)
# player3positions = playerDetection(videoNumber, player3Loc)
# player4positions = playerDetection(videoNumber, player4Loc)

