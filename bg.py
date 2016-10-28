import cv2 as cv2
import cv2.cv as cv
import numpy as np
import pick as pick

def reverseFilter(ff):
	ffH = ff.shape[0]
	ffW = ff.shape[1]
	l = 0
	r = ffW - 1
	while r>l :
		for j in range(0, ffH):
			tmp = ff[j][l]
			ff[j][l] = ff[j][r]
			ff[j][r] = tmp
		r = r-1
		l = l+1
	l = 0
	r = ffH - 1
	while r>l :
		for j in range(0, ffW):
			tmp = ff[l][j]
			ff[l][j] = ff[r][j]
			ff[r][j] = tmp
		r = r-1
		l = l+1
	return ff
			

def MyConvolve(img, ff):
	result = np.zeros(img.shape)
	ff = reverseFilter(ff)
	#print(ff)
	ffH = ff.shape[0]
	ffW = ff.shape[1]
	imgH = img.shape[0]
	imgW = img.shape[1]



	#print(str(imgH)+" "+str(imgW))	
	#print(str(ffH)+" "+str(ffW))
	maxV = 0
	minV = 1000	
	for i in range((ffW-1)/2, imgH - (ffW-1)/2): #we don't know the size of our filter in advance
		for j in range((ffH-1)/2, imgW - (ffH-1)/2):
			count = 0
			for u in range(-(ffW-1)/2, (ffW-1)/2+1):
				for v in range(-(ffH-1)/2, (ffH-1)/2+1):
					count = count + (img[i+u][j+v]*ff[u+(ffW-1)/2][v+(ffH-1)/2])
			result[i][j] = count
	return result

def gauss_kernels(size,sigma=1.0):
## returns a 2d gaussian kernel
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	kernel_sum = kernel.sum()
	#print(kernel.sum())
	if not sum==0:
		kernel = kernel/kernel_sum
	return kernel

def computeHarrisCorners(image, imageRGB, incr):
	filterH =np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	filterV =np.array([[-1, -2, -1],[0, 0, 0],[1, 2,1]])
	gx = MyConvolve(image, filterH)
	gy = MyConvolve(image, filterV)
	
	I_xx = gx*gx
	I_xy = gx*gy
	I_yy = gy*gy

	gaussFilter = gauss_kernels(3, 1.0)
	#print(gaussFilter)
	W_xx = MyConvolve(I_xx, gaussFilter)
	W_xy = MyConvolve(I_xy, gaussFilter)
	W_yy = MyConvolve(I_yy, gaussFilter)
	
	response = np.zeros(image.shape)
	height = image.shape[0]
	width  = image.shape[1]
	maxR = -1000
	i=incr + height/2;
	while i<height:
		j = incr
		while j<width:

			detW = W_xx[i][j]*W_yy[i][j] - W_xy[i][j]*W_xy[i][j]
			traceW = W_xx[i][j]+W_yy[i][j]
			response[i][j] = detW - 0.06*traceW*traceW
			if response[i][j] > maxR:
				maxR = response[i][j]
			j = j+incr
		i = i + incr

	#print(str(maxR)+"max resp")
	i=incr + height/2
	while i<height:
		j = incr
		while j<width:
			if response[i][j] >= 0.05*maxR:
				for k  in range(-4,5):
					if(i+4<height and j+4<width):
						imageRGB[i-k][j-4] = [0,0 ,255]
						imageRGB[i-k][j+4] = [0,0 ,255]
						imageRGB[i-4][j-k] = [0,0 ,255]
						imageRGB[i+4][j-k] = [0,0 ,255]
			j = j + incr
				
		i = i +incr
	return image



def getHomographyMatrix(points1, points2):
	M = np.matrix(np.zeros((8,9)))

	print(points1)
	print(points2)
	i = 0
	for i in range(0,len(points1)):
		uc, vc = points1[i]
		up, vp = points2[i]
		zp = 1
		M[2*i,:] = [up, vp, zp, 0, 0, 0, -uc*up, -uc*vp, -zp*uc]
		M[2*i+1,:] = [0, 0, 0, up, vp, zp, -vc*up, -vc*vp, -zp*vc]

	#print(M)

	U, s, V = np.linalg.svd(M, full_matrices=True)
	V = np.transpose(V)
	S = np.diag(s)

	print(S)
	#print(V)
	indexes = list()
	for i in range(0, S.shape[0]):
		array = S[i,:]
		nz = np.nonzero(array > 0.01)
		if not len(nz[0]):
			indexes.append(i)

	resVector = np.zeros(9)	
	
	for index in indexes:
		resVector = resVector + np.transpose(V[:, i])

	homography = np.reshape(resVector, (3, 3))

	return homography


def main():
	cap = cv2.VideoCapture("beachVolleyball1.mov")

	points1 = list()

	fps = (int)(cap.get(cv2.cv.CV_CAP_PROP_FPS))
	frameCount = (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	width = (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	height = (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	a , fstImg = cap.read()

	for i in range(0, 4):
		x, y =  0,0#pick.indicateLocation(fstImg)
		points1.append([(float)(x),(float)(y)])

	avgImg = np.float32(fstImg)
	normImg = np.uint8(fstImg)
	for fr in range(1, frameCount):
		_,img = cap.read()
 

		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		cornerMap = cv.CreateMat(height, width, cv.CV_32FC1)
		# OpenCV corner detection
		mt = cv.fromarray(gray)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		out = cv.fromarray(img)
		for y in range(0, height):
		 for x in range(0, width):
		  harris = cv.Get2D(cornerMap, y, x) # get the x,y value
		  # check the corner detector response
		 img[dst>0.01*dst.max()]=[0,0,255]


		cv2.imshow("corners", np.array(out))
		cv2.waitKey(30)

		if(fr == frameCount - 1)	:
				points2 = list()
				proj = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

				for i in range(0, 4):
					x, y = 0,0 #pick.indicateLocation(img)
					points2.append([(float)(x),(float)(y)])


		    # Calculate Homograph

				h, status = cv2.findHomography(np.array(points2), np.array(points1))

				#print(homography)
				print(h)
				print(points1)
				print(points2)


				for i in range(img.shape[0]):
					for j in range(img.shape[1]):
						p = np.zeros((3,1))
						p[0,0] = i
						p[1,0] = j
						p[2,0] = 1
						pos = np.dot(h, p)
						#print(pos)

						px = (int)(pos[0,0]/pos[2,0])
						py = (int)(pos[1,0]/pos[2,0])

						if(px>=0 and py>=0 and px<img.shape[0] and py<img.shape[1]):
							proj[i][j] = fstImg[px][py]


				

				cv2.accumulateWeighted(proj, avgImg, 1./fr)
				cv2.convertScaleAbs(avgImg, normImg) 
				cv2.imwrite('p.jpg', proj)
				cv2.waitKey(30)

		print "fr = ", fr
		#cv2.imwrite('img',img)

	cv2.convertScaleAbs(avgImg, normImg) 
	cv2.imshow('normImg', normImg) # normImg is avgImg converted into uint8
	#print "fr = ", fr, " alpha = ", alpha
	cv2.imwrite('background.jpg', normImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	main()
