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


video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')

frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(video.get(cv2.CAP_PROP_FPS))
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#print frameCount
#print [frameHeight, frameWidth]
newWidth = frameWidth+200
newHeight = frameHeight+200

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outObj = cv2.VideoWriter('output/HomoPts_' + str(videoNumber) + '.avi',fourcc, frameFPS, (newWidth, newHeight))

nrPts = 10 #vid2pts.shape[1] #10
noPt = 0

outPts = np.zeros([frameCount, nrPts, 2], int)
outImg = np.zeros([newHeight, newWidth, 3], np.uint8)


'''
# Mark points for check
for fr in range(frameCount):

	_,image = video.read()
	outImg[100:image.shape[0]+100, 100:image.shape[1]+100] = image

	# Mark positions
	for pt in range(nrPts):

		if vid2pts[fr, pt, 0] > -100 : # and vid2pts[fr, pt, 0] < image.shape[0] and vid2pts[fr, pt, 1] > -10 and vid2pts[fr, pt, 1] < image.shape[1]:
			outImg[vid2pts[fr, pt, 0]+95:vid2pts[fr, pt, 0]+105,vid2pts[fr, pt, 1]+100]=[0,0,255]
			outImg[vid2pts[fr, pt, 0]+100,vid2pts[fr, pt, 1]+95:vid2pts[fr, pt, 1]+105]=[0,0,255]
		
		else:
			noPt = noPt + 1

	if noPt > 6 : 
		print "Fr: " + str(fr)

	noPt = 0

	outObj.write(outImg)

video.release()
outObj.release()
'''


# Pick points
#for pt in range(nrPts):
pt = 0

for fr in range(frameCount):
	_,image = video.read()
	outImg[100:image.shape[0]+100, 100:image.shape[1]+100] = image

	x, y = indicateLocation(outImg)
	print(y-100, x-100)
	print(str(fr) + "/" + str(frameCount))
	outPts[fr, pt, :] = [y-100,x-100]


# Start over again
video.release()
video = cv2.VideoCapture('beachVolleyballFilms/beachVolleyball'+str(videoNumber)+'.mov')


video.release()
outObj.release()


np.save('positionArrays/homographyPts/pts_vid' + str(videoNumber) +'_.npy', outPts)
