import cv2
from cv2 import cv
import numpy as np
import sys

def showCorners(video_name, corners):
	""" display corners on a video """
	cap = cv2.VideoCapture(video_name)
	frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
	print frameCount, fps, frameHeight, frameWidth
	timeBetweenFrame = 1000 / frameCount
	for fr in range(0, frameCount-1):
		img = cap.read()[1]
		print "FRAME: ", fr, frameWidth, frameHeight
		pad = np.zeros((3*frameHeight,3*frameWidth,3),dtype='uint8')
		pad[frameHeight:2*frameHeight, frameWidth:2*frameWidth] = img
		for i in range(0, corners.shape[1]):
			if corners[fr,i,0] != -9999:
				print corners[fr,i,:]
				cv2.circle(pad, (corners[fr,i,0]+frameWidth, corners[fr,i,1]+frameHeight), 4, (0,0,255), 2)
		cv2.imshow('pad', pad)
		cv2.waitKey(1)
	
if __name__ == '__main__':
	showCorners(sys.argv[1], np.load(sys.argv[2]))