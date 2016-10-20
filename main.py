#!/usr/bin/env python 

import numpy as np
import cv2
import math
import sys
from playerDetection import playerDetec

# Choose which video to process 
if (sys.argv < 1):
	print 'Usage: main.py <VideoIndex>'
else:
	try:
		vidNr = str(sys.argv[1])
	except:
		print 'Usage: main.py <(int)VideoIndex>'

# Where to start look
indLoc = np.zeros(2, int)
indLoc[0] = 235 #uint8(sys,argv[2])
indLoc[1] = 496 #uint8(sys.argv[3])


playerPos = playerDetec(vidNr, indLoc)

#print playerPos
