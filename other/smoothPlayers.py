import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

videoNumber = 1
#startFr = 215

playerName = ["greenBack", "greenFront", "whiteRight", "whiteLeft"]
#playerName = "greenBack"
#playerName = "greenFront"
#playerName = "whiteLeft"
#playerName = "whiteRight"
#playerName = "greenDown"
#playerName = "greenUp"
#playerName = "redBack"
#playerName = "redFront"
#playerName = "redDown"
#playerName = "redUp"
#playerName = "redLeft"
#playerName = "redRight"
#playerName = "whiteDown"
#playerName = "whiteUp

for name in playerName:

	pts = np.load('../positionArrays/feetPositions/feet'+str(videoNumber)+'_'+str(name)+'.npy')
	outPts = np.zeros(pts.shape)

	#for fr in range(startFr):
	#	pts[fr, :] = pts[startFr+1, :]


	print outPts.shape
	outPts[:,1] = signal.savgol_filter(pts[:,0], 81, 3)
	outPts[:,0] = signal.savgol_filter(pts[:,1], 81, 3)

	fig1 = plt.figure(1)
	plt.plot(range(pts.shape[0]), pts[:,:], 'r-')
	fig1.savefig('pics/orig_'+str(name)+'.png')
	plt.close()

	fig2 = plt.figure(1)
	plt.plot(range(outPts.shape[0]), outPts[:,:], 'r-')
	fig2.savefig('pics/smooth_'+str(name)+'.png')
	plt.close()


	np.save('../positionArrays/smoothFeetPos/feet' + str(videoNumber) +'_'+str(name)+'.npy', outPts)
