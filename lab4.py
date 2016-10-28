import cv2 as cv2
import cv2.cv as cv
import numpy as np
import math as mth

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
	i=incr
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
	i=incr
	while i<height:
		j = incr
		while j<width:
			if response[i][j] >= 0.1*maxR:
				for k  in range(-4,5):
					if(i+4<height and j+4<width):
						imageRGB[i-k][j-4] = [0,0 ,255]
						imageRGB[i-k][j+4] = [0,0 ,255]
						imageRGB[i-4][j-k] = [0,0 ,255]
						imageRGB[i+4][j-k] = [0,0 ,255]
			j = j + incr
				
		i = i +incr
	return image
	
def main():

	names = ["checker", "flower", "test1", "test2", "test3"]
	for name in names:
		print("starting "+name)
		img = cv2.imread(name+".jpg",0)
		imgRGB = cv2.imread(name+".jpg")
		image = computeHarrisCorners(img, imgRGB,10)
		cv2.imwrite(name+"_corner_10.jpg", imgRGB)
		image = computeHarrisCorners(img, imgRGB,1)
		cv2.imwrite(name+"_corner_1.jpg", imgRGB)
		print("finishing "+name)

if __name__ == '__main__':
	main()
