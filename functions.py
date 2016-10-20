import numpy as np
import cv2
import math



def gauss_kernels(size, sigma=1.0):
	## returns a 2d gaussian kernel
	if size<3:
		size = 3
	m = size/2

	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	kernel_sum = kernel.sum()

	if not sum==0:
		kernel = kernel/kernel_sum

	return kernel


def MyConvolve (img, kernel):
	result = np.zeros(img.shape, np.uint8)
	rows = img.shape[0]
	cols = img.shape[1]

	margin = kernel.shape[0]/2

	# zero padding
	padImg = np.zeros((rows+2*margin, cols+2*margin))
	padImg[margin:rows+margin, margin:cols+margin] = img

	# Flip kernel!
	kernel = np.fliplr(np.flipud(kernel))
	
	# Skip the zero padding so we don't detect the edges later on
	for i in range(margin, rows+margin):
		for j in range(margin, cols+margin):
			
			#value = (padImg[i-1,j-1]*kernel[0,0])+(padImg[i-1,j]*kernel[0,1])+(padImg[i-1,j+1]*kernel[0,2])+ \
			#		(padImg[i,j-1]*kernel[1,0])+(padImg[i,j]*kernel[1,1])+(padImg[i,j+1]*kernel[1,2])+ \
			#		(padImg[i+1,j-1]*kernel[2,0])+(padImg[i+1,j]*kernel[2,1])+(padImg[i+1,j+1]*kernel[2,2])
			
			Mvalue = np.multiply(padImg[i-margin:i+margin+1,j-margin:j+margin+1],kernel)
			value = np.sum(Mvalue)

			result[i-margin, j-margin] = abs(value)

	return result