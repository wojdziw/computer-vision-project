import cv2 as cv2
import cv2.cv as cv
import numpy as np
import math as mth
import visu as v
import sys

def main():

	if len(sys.argv)<2:
		print("<video number>")
		sys.exit(0)

	video = sys.argv[1]
	cap = cv2.VideoCapture("../data/videos/balltrackEnhanced"+video+".avi")
	p1 = np.load("../data/topViewPlayers/video"+video+"_player1.npy")
	p2 = np.load("../data/topViewPlayers/video"+video+"_player2.npy")
	p3 = np.load("../data/topViewPlayers/video"+video+"_player3.npy")
	p4 = np.load("../data/topViewPlayers/video"+video+"_player4.npy")


	jumpsList = list()
	p1j = np.load("../data/jumps/jumps"+video+"_1.npy")
	p2j = np.load("../data/jumps/jumps"+video+"_2.npy")
	p3j = np.load("../data/jumps/jumps"+video+"_3.npy")
	p4j = np.load("../data/jumps/jumps"+video+"_4.npy")

	jumpsList.append(p1j)
	jumpsList.append(p2j)
	jumpsList.append(p3j)
	jumpsList.append(p4j)

	jumpStatus = list()
	jumps = list()

	for i in range(4):
		jumpStatus.append(False)
		jumps.append(0)

	frCount = (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	fourcc = cv.CV_FOURCC('F', 'L', 'V', '1')
	video = cv2.VideoWriter('finalVisu'+video+'.avi',fourcc,30,(1200,1000))

	d = list();
	for i in range(4):
		d.append(0)

	for i in range(1, len(p1)):
		print(len(p1))
		print(frCount)
		players = list()


		if p1[i-1, 0]>0 and p1[i-1, 1]>0:
			delta = np.sqrt((p1[i-1, 0]/100 - p1[i, 0]/100)*(p1[i-1, 0]/100 - p1[i, 0]/100) + (p1[i-1, 1]/100 - p1[i, 1]/100)*(p1[i-1, 1]/100 - p1[i, 1]/100))
			d[0] = d[0] + delta

		if p2[i-1, 0]>0 and p2[i-1, 1]>0:
			delta = np.sqrt((p2[i-1, 0]/100 - p2[i, 0]/100)*(p2[i-1, 0]/100 - p2[i, 0]/100) + (p2[i-1, 1]/100 - p2[i, 1]/100)*(p2[i-1, 1]/100 - p2[i, 1]/100))
			d[1] = d[1] + delta

		if p3[i-1, 0]>0 and p3[i-1, 1]>0:
			delta = np.sqrt((p3[i-1, 0]/100 - p3[i, 0]/100)*(p3[i-1, 0]/100 - p3[i, 0]/100) + (p3[i-1, 1]/100 - p3[i, 1]/100)*(p3[i-1, 1] /100- p3[i, 1]/100))
			d[2] = d[2] + delta

		if p4[i-1, 0]>0 and p4[i-1, 1]>0:
			delta = np.sqrt((p4[i-1, 0]/100 - p4[i, 0]/100)*(p4[i-1, 0]/100 - p4[i, 0]/100) + (p4[i-1, 1]/100 - p4[i, 1]/100)*(p4[i-1, 1]/100 - p4[i, 1]/100))
			d[3] = d[3] + delta

		#jumps detection
		for p in range(4):
			if jumpsList[p][i] == True  and jumpStatus[p] == False:
				jumps[p] = jumps[p] + 1
				jumpStatus[p] = True
			if jumpsList[p][i] == False and jumpStatus[p] == True:
				jumpStatus[p] = False	

		players.append(v.Player(p1[i,0], p1[i,1], 1, (255,0,255),d[0], jumpStatus[0], jumps[0]))
		players.append(v.Player(p2[i,0], p2[i,1], 2, (255,0,255),d[1], jumpStatus[1], jumps[1]))
		players.append(v.Player(p3[i,0], p3[i,1], 3, (255,255,0),d[2], jumpStatus[2], jumps[2]))
		players.append(v.Player(p4[i,0], p4[i,1] ,4, (255,255,0),d[3], jumpStatus[3], jumps[3]))

		field  = v.Field(1600,800)

		ret, image = cap.read()

		visu = v.drawScreen(players, field, image, image)

		video.write(visu)

		cv2.imshow("visu", visu)
		cv2.waitKey(10)

	cap.release()
	video.release()

if __name__ == '__main__':
	main()