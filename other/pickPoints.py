import cv2
import sys
sys.path.append("../playerDetection")
from playerDetection import *
from pickStartPoint import *
from smoothPos import *
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



ptsAll = np.load('../positionArrays/homographyPts/rawPts_vid' + str(videoNumber) + '_all.npy')
#ptsAll = np.load('../positionArrays/homographyPts/video' + str(videoNumber) + '_points_raw.npy')
#ptsAll = np.load('../positionArrays/homographyPts/video' + str(videoNumber) + '_points_smoothed.npy')

#pts0 = np.load('../positionArrays/homographyPts/video2_points_0.npy')
#pts1 = np.load('../positionArrays/homographyPts/video2_points_1.npy')
#pts2 = np.load('../positionArrays/homographyPts/video2_points_2.npy')
#pts3 = np.load('../positionArrays/homographyPts/video2_points_3.npy')
#pts4 = np.load('../positionArrays/homographyPts/video2_points_4.npy')
#pts5 = np.load('../positionArrays/homographyPts/video2_points_5.npy')
#pts6 = np.load('../positionArrays/homographyPts/pts_vid4_6.npy')
#pts7 = np.load('../positionArrays/homographyPts/pts_vid4_7.npy')



video = cv2.VideoCapture('../beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(video.get(cv2.CAP_PROP_FPS))
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#frameCount = frameCount-23

newWidth = frameWidth+200
newHeight = frameHeight+200

#newHeight = frameWidth+200
#newWidth = frameHeight+200

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outObj = cv2.VideoWriter('../output/HomoPts_' + str(videoNumber) + '.avi',fourcc, frameFPS, (newWidth, newHeight))

nrPts = ptsAll.shape[1] #10
noPt = 0

outPts = np.zeros([frameCount, nrPts, 2], int)
outImg = np.zeros([newHeight, newWidth, 3], np.uint8)

# Swap!
#print ptsAll[0,0,:]
#print ptsAll[0,2,:]
#outPts[:,0,0] = ptsAll[:,0,1]
#outPts[:,0,1] = ptsAll[:,0,0]
#outPts[:,1,0] = ptsAll[:,1,1]
#outPts[:,1,1] = ptsAll[:,1,0]
#outPts[:,4,0] = ptsAll[:,2,1]
#outPts[:,4,1] = ptsAll[:,2,0]
#outPts[:,5,0] = ptsAll[:,3,1]
#outPts[:,5,1] = ptsAll[:,3,0]
#print outPts[0,0,:]
#print outPts[0,4,:]

'''
#Smooth positions!
outPts = smoothPointsArray(ptsAll)

for pt in range(nrPts):
	fig1 = plt.figure(1)
	plt.plot(range(frameCount), ptsAll[:-23,pt,:], 'r-')
	fig1.savefig('pics/smoothed'+str(pt+nrPts)+'.png')
	plt.close()
	fig2 = plt.figure(1)
	plt.plot(range(outPts.shape[0]), outPts[:,pt,:], 'r-')
	fig2.savefig('pics/smoothed'+str(pt)+'.png')
	plt.close()
'''
# Mark points for check
for fr in range(frameCount):
	outPts[fr, :, :] = ptsAll[fr, :, :]
	#outPts[fr, 0, :] = pts0[fr, :, :]
	#outPts[fr, 1, :] = pts1[fr, :, :]
	#outPts[fr, 2, :] = pts2[fr, :, :]
	#outPts[fr, 3, :] = pts3[fr, :, :]
	#outPts[fr, 4, :] = pts4[fr, :, :]
	#outPts[fr, 5, :] = pts5[fr, :, :]
	#outPts[fr, 6, :] = pts6[fr, :, :]
	#outPts[fr, 7, :] = pts7[fr, :, :]

	_,image = video.read()
	outImg[100:image.shape[0]+100, 100:image.shape[1]+100] = image

	# Mark positions
	for pt in range(nrPts):

		# Smooth the shit outa that!
		if outPts[fr, pt, 1]+105 > newWidth:
			outPts[fr, pt, 1] = newWidth-106
		if outPts[fr, pt, 0]+105 > newHeight:
			outPts[fr, pt, 0] = newHeight-106

		if outPts[fr, pt, 0] > -94 : 
			outImg[outPts[fr, pt, 0]+95:outPts[fr, pt, 0]+105,outPts[fr, pt, 1]+100]=[0,0,255]
			outImg[outPts[fr, pt, 0]+100,outPts[fr, pt, 1]+95:outPts[fr, pt, 1]+105]=[0,0,255]
		
		else:
			noPt = noPt + 1

	if noPt > nrPts-4 : 
		print "Fr: " + str(fr)

	noPt = 0

	outObj.write(outImg)

video.release()
outObj.release()


'''
# Pick points
for pt in range(nrPts):

	for fr in range(frameCount):
		_,image = video.read()
		outImg[100:image.shape[0]+100, 100:image.shape[1]+100] = image

		x, y = indicateLocation(outImg)
		print str(fr) + " out of " + str(frameCount) + ", (" +str(y-100)+", " +str(x-100) +")"
		outPts[fr, pt, :] = [y-100,x-100]


	# Start over again
	video.release()
	video = cv2.VideoCapture('../beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')


video.release()
outObj.release()
'''


np.save('../positionArrays/homographyPts/video' + str(videoNumber) +'_points_.npy', outPts)
