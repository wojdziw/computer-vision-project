import sys
sys.path.append("playerDetection")

from playerDetection import *
from pickStartPoint import *
import cv2
import numpy as np

videoNumber = 2
video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')
_,image = video.read()

a, b = indicateLocation(image)
print(a, b)

player1loc = [b,a]

player1positions = playerDetection(videoNumber, player1loc)
# player2positions = playerDetection(videoNumber, player12oc)
# player3positions = playerDetection(videoNumber, player12oc)
# player3positions = playerDetection(videoNumber, player14oc)
