import numpy as np 
import cv2
import sys

top_left = [0,0]
bot_left = [0,800]
top_right = [1600,0]
bot_right = [1600,800]
top_stack = [800,-100]
bot_stack = [800,900]
pointsField=[top_left, bot_left, top_right, bot_right, top_stack, bot_stack]

def getHomography(points, frame):
	pD = list()
	pT = list()

	i = 0
	#print(points)
	for i in range(6):
		[x, y] = points[i]
		if x>0 or y >0 and i<6:
			cv2.circle(frame,((int)(x),(int)(y)),10,[255,255,255],-1)
			pD.append([x, y])
			pT.append(pointsField[i])

	H = np.identity(3)
	if len(pD)>=4:
	 	H, mask = cv2.findHomography(np.array(pT, np.float64), np.array(pD,  np.float64))


	H = np.linalg.inv(H)

	for i in range(len(pT)):
		p = [1, 1, 1]
		p[0:2] = pD[i]
		pTop = np.matrix(H)*np.array(np.transpose(np.matrix(p)))
		pTop = pTop / pTop[2]
		#print(pTop)

	H = np.matrix(H)
	return H



if len(sys.argv)<2:
	print("<video number>")
	sys.exit(0)

video = sys.argv[1]
pointsFile = np.load("../data/points/video"+video+"_points.npy")
cap = cv2.VideoCapture("../data/videos/beachVolleyball"+video+".mov")

player1Pos = np.load("../data/feetPositions/feet"+video+"_1.npy")
player2Pos = np.load("../data/feetPositions/feet"+video+"_2.npy")
player3Pos = np.load("../data/feetPositions/feet"+video+"_3.npy")
player4Pos = np.load("../data/feetPositions/feet"+video+"_4.npy")

p1 = list()
p2 = list()
p3 = list()
p4 = list()

p1D = [1,1,1]
p2D = [1,1,1]
p3D = [1,1,1]
p4D = [1,1,1]

for i in range(len(pointsFile)):

	ret, frame= cap.read()

	repres = np.zeros((2000, 2000), np.uint8)

	cv2.circle(repres,(0,0),10,[255,0,0],-1)
	cv2.circle(repres,(0,800),20,[255,0,0],-1)
	cv2.circle(repres,(1600,0),30,[255,0,0],-1)
	cv2.circle(repres,(1600,800),40,[255,0,0],-1)
	cv2.circle(repres,(800,0),3,[255,0,0],-1)
	cv2.circle(repres,(800,800),3,[255,0,0],-1)

	points =  pointsFile[i]
	#print(points)

	H = getHomography(points, frame)

	#print(H)
	p1D[0:2] = player1Pos[i]
	tmp = p1D[0]
	#p1D[0] = p1D[1]
	#p1D[1] = tmp

	p2D[0:2] = player2Pos[i]
	tmp = p2D[0]
	#p2D[0] = p2D[1]
	#p2D[1] = tmp

	p3D[0:2] = player3Pos[i]
	tmp = p3D[0]
	#p3D[0] = p3D[1]
	#p3D[1] = tmp

	p4D[0:2] = player4Pos[i]
	tmp = p4D[0]
	#p4D[0] = p4D[1]
	#p4D[1] = tmp

	#print(p1D)
	p1T = H * np.array(np.transpose(np.matrix(p1D)))
	p2T = H * np.array(np.transpose(np.matrix(p2D)))
	p3T = H * np.array( np.transpose(np.matrix(p3D)))
	p4T = H * np.array(np.transpose(np.matrix(p4D)))

	for i in range(len(pointsField)):
		p = [1, 1, 1]
		p[0:2] = pointsField[i]
		pTop = np.linalg.inv(H)*np.array(np.transpose(np.matrix(p)))
		pTop = pTop / pTop[2]            
		#print(pTop)
		if(abs(pTop[0])<10000 and abs(pTop[1])<10000):
			cv2.circle(frame,((int)(pTop[0]),(int)(pTop[1])),5,[255,255,0],-1)


	p1T = p1T  / p1T[2] 
	p2T = p2T  / p2T[2] 
	p3T = p3T  / p3T[2] 
	p4T = p4T  / p4T[2]

	[[x, y, z]] = np.array(np.transpose(p1T))
	p1T = [x, y, z]
	[[x, y, z]] = np.array(np.transpose(p2T))
	p2T = [x, y, z]
	[[x, y, z]] = np.array(np.transpose(p3T))
	p3T = [x, y, z]
	[[x, y, z]] = np.array(np.transpose(p4T))
	p4T = [x, y, z]


	cv2.circle(repres,((int)(p1T[0]),(int)(p1T[1])),20,[255,0,0],-1)
	cv2.circle(repres,((int)(p2T[0]),(int)(p2T[1])),20,[255,0,0],-1)
	cv2.circle(repres,((int)(p3T[0]),(int)(p3T[1])),20,[255,0,0],-1)
	cv2.circle(repres,((int)(p4T[0]),(int)(p4T[1])),20,[255,0,0],-1)

	cv2.circle(frame,((int)(p1D[0]),(int)(p1D[1])),20,[255,0,0],-1)
	cv2.circle(frame,((int)(p2D[0]),(int)(p2D[1])),20,[255,0,0],-1)
	cv2.circle(frame,((int)(p3D[0]),(int)(p3D[1])),20,[255,0,0],-1)
	cv2.circle(frame,((int)(p4D[0]),(int)(p4D[1])),20,[255,0,0],-1)


	cv2.imshow("currPos",repres)
	cv2.imshow("posFeet",frame)
	cv2.waitKey(10)
	p1.append(p1T)
	p2.append(p2T)
	p3.append(p3T)
	p4.append(p4T)
	#print(p1T)

np.save("../data/topViewPlayers/video"+video+"_player1.npy", p1)
np.save("../data/topViewPlayers/video"+video+"_player2.npy", p2)
np.save("../data/topViewPlayers/video"+video+"_player3.npy", p3)
np.save("../data/topViewPlayers/video"+video+"_player4.npy", p4)