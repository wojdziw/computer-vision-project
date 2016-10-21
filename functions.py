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


# Calculate the average RGB values
def avgColor(area):

	color = np.zeros(3, np.uint8)
	rows = area.shape[0]
	cols = area.shape[1]

	color[2] = area[:,:,0].mean(axis=(0,1))
	color[1] = area[:,:,1].mean(axis=(0,1))
	color[0] = area[:,:,2].mean(axis=(0,1))

	return color

# Convert rgb to Lab using RGB -> XYZ -> Lab
def rgb2lab(rgb):

	#print rgb
	## Convert RGB to XYZ
	num = 0
	RGB = [0, 0, 0]

	for value in rgb :
	   value = float(value) / 255

	   if value > 0.04045 :
	       value = np.power(((value+0.055)/1.055), 2.4)
	   else :
	       value = value / 12.92

	   RGB[num] = value * 100
	   num = num + 1

	XYZ = [0, 0, 0,]

	X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
	Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
	Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505

	XYZ[0] = round(X, 4)
	XYZ[1] = round(Y, 4)
	XYZ[2] = round(Z, 4)

	#print XYZ
	## Convert XYZ to Lab

	XYZ[0] = float(XYZ[0]) / 95.047		# ref_X =  95.047   Observer= 2 deg, Illuminant= D65
	XYZ[1] = float(XYZ[1]) / 100.0		# ref_Y = 100.000
	XYZ[2] = float(XYZ[2]) / 108.883	# ref_Z = 108.883

	num = 0
	for value in XYZ :

	   if value > 0.008856 :
	       value = np.power(value, float(1)/3)
	   else :
	       value = (7.787*value) + (16/116)

	   XYZ[num] = value
	   num = num + 1

	Lab = [0, 0, 0]

	L = (116 * XYZ[1]) - 16
	a = 500 * (XYZ[0] - XYZ[1])
	b = 200 * (XYZ[1] - XYZ[2])

	Lab [ 0 ] = round( L, 4 )
	Lab [ 1 ] = round( a, 4 )
	Lab [ 2 ] = round( b, 4 )

	#print Lab

	return Lab


# Calculate the difference between two Lab colors (range 0-100, accept colors below 20?)
# Use DeltaE CIE1976, problem with saturation but best performance
def deltaE(lab1, lab2):

	L1, a1, b1 = lab1[0], lab1[1], lab1[2]
	L2, a2, b2 = lab2[0], lab2[1], lab2[2]

	diff = np.sqrt(np.power(L1-L2, 2), np.power(a1-a2, 2), np.power(b1-b2, 2))

	return diff


# Traverse down to the bottom-most part with similar 
# How to make sure we stay on the same player?
def traveseDown(start, color):

	footPt = np.zeros(2, int)
	newColor = color


	return footPt, newColor


# Find closest pixel with similar color
def findLeg(startPt, color):

	legPt = np.zeros(2, int)
	newColor = color


	return legPt, newColor