#!/usr/bin/env python 

import numpy as np
import cv2
import math
import sys
from playerDetection import playerDetec

# Choose which video to process 
if (sys.argv < 2):
	print 'Usage: main.py <VideoIndex> <SearchRadius>'
else:
	try:
		vidNr = str(sys.argv[1])
		radius = int(sys.argv[2])
	except:
		print 'Usage: main.py <(int)VideoIndex> <(int)SearchRadius>'

# Where to start look
pl1 = np.zeros([4, 2], int)
pl1[0, 0] = 235 #uint8(sys,argv[3])
pl1[0, 1] = 496 #uint8(sys.argv[4])
pl1[1, 0] = 111
pl1[1, 1] = 206
pl1[2, 0] = 75
pl1[2, 1] = 96
pl1[3, 0] = 68
pl1[3, 1] = 168


playerPos = playerDetec(vidNr, pl1[0, :], radius)

#print playerPos
