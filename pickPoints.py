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



#vid2pts = np.load('./data/all_corners_video2.npy')
#vid2pts = np.load('./data/all_corners_video3.npy')
#vid2pts = np.load('./data/all_corners_video7.npy')

#vid2pts = np.load('./positionArrays/homographyPts/pts_vid4.npy')
ptsAll = np.load('./positionArrays/homographyPts/pts_vid' + str(videoNumber) + '_all.npy')
#pts0 = np.load('./positionArrays/homographyPts/pts_vid4_0.npy')
#pts1 = np.load('./positionArrays/homographyPts/pts_vid4_1.npy')
#pts2 = np.load('./positionArrays/homographyPts/pts_vid4_2.npy')
#pts3 = np.load('./positionArrays/homographyPts/pts_vid4_3.npy')
#pts4 = np.load('./positionArrays/homographyPts/pts_vid4_4.npy')
#pts5 = np.load('./positionArrays/homographyPts/pts_vid4_5.npy')
#pts6 = np.load('./positionArrays/homographyPts/pts_vid4_6.npy')
#pts7 = np.load('./positionArrays/homographyPts/pts_vid4_7.npy')
#print pts0.shape
#print pts1.shape
#print pts2.shape
#print pts3.shape
#print pts4.shape
#print pts5.shape
#print pts6.shape
#print pts7.shape

video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(video.get(cv2.CAP_PROP_FPS))
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


newWidth = frameWidth+200
newHeight = frameHeight+200

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outObj = cv2.VideoWriter('output/HomoPts_' + str(videoNumber) + '.avi',fourcc, frameFPS, (newWidth, newHeight))

nrPts = 1 #ptsAll.shape[1] #vid2pts.shape[1] #10
noPt = 0

outPts = np.zeros([frameCount, nrPts, 2], int)
outImg = np.zeros([newHeight, newWidth, 3], np.uint8)


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

		if outPts[fr, pt, 0] > -95 : # and outPts[fr, pt, 0] < image.shape[0] and outPts[fr, pt, 1] > -10 and outPts[fr, pt, 1] < image.shape[1]:
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
	video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')


video.release()
outObj.release()


np.save('positionArrays/homographyPts/video' + str(videoNumber) +'_points_.npy', outPts)
'''