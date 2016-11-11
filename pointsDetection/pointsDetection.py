import os
import numpy as np
import cv2
import sys
import math
import cv2.cv as cv

HORIZONTAL=1
VERTICAL=0


class _Getch:
	"""Gets a single character from standard input.  Does not echo to the
screen."""
	def __init__(self):
		try:
			self.impl = _GetchWindows()
		except ImportError:
			self.impl = _GetchUnix()

	def __call__(self): return self.impl()


class _GetchUnix:
	def __init__(self):
		import tty, sys

	def __call__(self):
		import sys, tty, termios
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch

class _GetchWindows:
	def __init__(self):
		import msvcrt

	def __call__(self):	
		import msvcrt
		return msvcrt.getch()


class kernel(object):
	SOBEL_3X3_H = np.array([[-1, 0, 1], \
							[-2, 0, 2], \
							[-1, 0, 1]])
	SOBEL_3X3_V = np.array([[-1, -2, -1], \
							[0, 0, 0], \
							[1, 2, 1]])
	SOBEL_5X5_H = np.array([[-2, -1, 0 ,1 ,2], \
							[-3, -2, 0, 2, 3], \
							[-4, -3, 0, 3, 4], \
							[-3, -2, 0, 2, 3], \
							[-2, -1, 0 ,1 ,2]])
	SOBEL_5X5_V = SOBEL_5X5_H.T
	SOBEL_7X7_H = np.array([[-3, -2, -1, 0 ,1 ,2, 3], \
							[-4, -3, -2, 0, 2, 3, 4], \
							[-5, -4, -3, 0, 3, 4, 5], \
							[-6, -5, -4, 0, 4, 5, 6], \
							[-5, -4, -3, 0, 3, 4, 5], \
							[-4, -3, -2, 0, 2, 3, 4], \
							[-3, -2, -1, 0 ,1 ,2, 3]])
	SOBEL_7X7_V = SOBEL_7X7_H.T
	SOBEL_11X11_V = np.array([[-4, -3, -2, -1, 0, 1, 2, 3, 4], \
					  [-5, -4, -3, -2, 0, 2, 3, 4, 5], \
					  [-6, -5, -4, -3, 0, 3, 4, 5, 6], \
					  [-7, -6, -5, -4, 0, 4, 5, 6, 7], \
					  [-8, -6, -5, -4, 0, 4, 5, 6, 7], \
					  [-7, -6, -5, -4, 0, 4, 5, 6, 7], \
					  [-6, -5, -4, -3, 0, 3, 4, 5, 6], \
					  [-5, -4, -3, -2, 0, 2, 3, 4, 5], \
					  [-4, -3, -2, -1, 0, 1, 2, 3, 4]])
	SOBEL_11X11_H = SOBEL_11X11_V.T

def checkLine(mask, x1, y1, x2, y2, minLineLength=0, maxLineGap = float('inf')):
	""" check if line respect the condition of have at least a certain length
		with an gap allowance threshold """
	vector = np.array([x2-x1,y2-y1],dtype='float32')
	img = np.empty((mask.shape[0], mask.shape[1],3))
	for i in range(0, 3):
		img[:,:,i] = mask
	nb_iterations = np.sum(vector)
	vector /= nb_iterations
	_pt = np.array([x1, y1], dtype='float32')
	onLine = False
	maxLineLength = 0
	curLineLength = 0
	curGapLength = 0
	for i in range(0, nb_iterations):
		value = mask[int(_pt[1]), int(_pt[0])]
		if onLine:
			if value == 0:
				if curGapLength < maxLineGap:
					curGapLength += 1
				else:
					if curLineLength > maxLineLength:
						maxLineLength = curLineLength
					onLine = False
					curGapLength = 0
					curLineLength = 0
			else:
				if curGapLength != 0:
					curLineLength += curGapLength 
					curGapLength = 0
				curLineLength += 1
				if curLineLength >= minLineLength:
					return True
		else:
			if value != 0:
				onLine = True
				curLineLength += 1
		_pt += vector
	return False

def MyConvolveWithIm2Col(img, ff): # Using im2col method
	filter_shape = ff.shape
	flat_filt = ff[::-1, ::-1].flatten().astype('float32')
	flat_filt = flat_filt.astype('float32') #/ float(np.sum(flat_filt))
	padded_img = np.zeros((img.shape[0] + filter_shape[0] - 1, img.shape[1] +  filter_shape[1] - 1))
	padding_size = (np.array([filter_shape[0], filter_shape[1]]) - 1) / 2
	padded_img[padding_size[0]:padding_size[0] + img.shape[0], padding_size[1]:padding_size[1] + img.shape[1]] = img.astype('float32')
	im2col=np.empty((flat_filt.size,img.shape[0] * img.shape[1]))
	for i in range(0, filter_shape[0]):
		for j in range(0, filter_shape[1]):
			for k in range(0, img.shape[0]):
				im2col[i * filter_shape[1] + j, k * img.shape[1]:(k + 1) * img.shape[1]] = padded_img[k+i,j:j+img.shape[1]]
	return np.reshape(np.dot(flat_filt, im2col), img.shape)

def getSandMask(img, mask_border=False, mask_logo=False, for_stack=False):
	""" Get the mask for the sand, you can mask the logo, the border (to avoid border gradient),
		if it's not for the stack detection it's reducing the mask for avoiding things on the border """

	kernel3x3 = np.ones((3,3))
	getch = _Getch()
	hue = cv2.cvtColor(img,cv.CV_BGR2HSV)[:,:,0]
	hue = cv2.blur(hue, (50,50))
	ret, _temp = cv2.threshold(hue,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	high_thresh = cv2.morphologyEx(_temp, cv2.MORPH_OPEN,kernel3x3, iterations = 20)
	cv2.morphologyEx(high_thresh, cv2.MORPH_CLOSE,kernel3x3, dst=high_thresh, iterations = 100)
	if for_stack:
		cv2.erode(high_thresh, kernel3x3, dst=high_thresh, iterations=20)
	else:
		cv2.erode(high_thresh, kernel3x3, dst=high_thresh, iterations=50)
	if mask_border:
		high_thresh[0,:] = 0
		high_thresh[-1,:] = 0
		high_thresh[:,0] = 0
		high_thresh[:,-1] = 0
	if mask_logo:
		LOGO_X1 = img.shape[1]-56
		LOGO_X2 = img.shape[1]-12
		LOGO_Y1 = img.shape[0]-93
		LOGO_Y2 = img.shape[0]-21
		high_thresh[LOGO_Y1:LOGO_Y2,LOGO_X1:LOGO_X2] = 0
	return cv2.erode(high_thresh, kernel3x3, iterations=4).astype('uint8') / 255.

def houghToCoords(rho, theta):
	""" Change from hough coordinate to cartesian coordinates"""
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	return np.array([x1,y1,x2,y2], dtype='int')

def getField(img, video_name):
	""" Try to return the visible lines """
	parameters = { 'beachVolleyball1.mov': [5, 6, 50, 0, 24, 190, 196, 92, 224], \
				   'beachVolleyball2.mov': [5, 6, 150, 0, 40, 134, 252, 90, 228], \
				   'beachVolleyball3.mov': [5, 3, 65, 0, 36, 138, 20, 68, 224],  \
				   'beachVolleyball4.mov': [5, 3, 65, 0, 36, 138, 20, 68, 224], \
				   'beachVolleyball5.mov': [5, 6, 50, 0, 36, 138, 264, 62, 216], \
				   'beachVolleyball6.mov': [5, 6, 100, 0, 36, 138, 26, 62, 220], \
				   'beachVolleyball7.mov': [5, 6, 65, 0, 36, 138, 264, 62, 216]}
	
	thresh1 = parameters[video_name][0]
	thresh2 = parameters[video_name][1]
	accu = parameters[video_name][2]
	H1 = parameters[video_name][3]
	S1 = parameters[video_name][4]
	V1 = parameters[video_name][5]
	H2 = parameters[video_name][6]
	S2 = parameters[video_name][7]
	V2 = parameters[video_name][8]
	mask = getSandMask(img, True, True)
	red = np.copy(img[:,:,1])
	blur = cv2.blur(red,(3,3))
	blur3 = cv2.blur(img, (5,5))
	HSV = cv2.cvtColor(blur3,cv.CV_BGR2HSV)
	tests = None
	while (True):
		img_copy = np.copy(img)
		img_copy2 = np.copy(img)
		_test1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh1, thresh2)
		hsv1 = np.array([H1,S1,V1])
		hsv2 = np.array([H2,S2,V2])
		_test3 = cv2.inRange(HSV, hsv1, hsv2)
		_test = np.multiply(_test3, _test1 / 255.)
		sand = np.multiply(_test, mask).astype('uint8')
		lines = cv2.HoughLines(sand, 1, np.pi / 500., accu)
		if lines is not None:
			# print lines.shape
			# for rho, theta in lines[0]:
			# 	x1,y1,x2,y2 = houghToCoords(rho, theta)
			# 	cv2.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
			# output = []
			# for rho, theta in lines[0]:
			# 	x1,y1,x2,y2 = extendVector(img, houghToCoords(rho, theta))
			# 	if checkLine(_test, x1,y1,x2,y2, line, 0):
			# 		output.append([rho,theta])
			# 		cv2.line(img_copy,(x1,y1),(x2,y2),(255,0,0),2)
			# output = np.array(output)
			# output = output.reshape(output.shape[0],1,2)
			tests = PruneNonProbaLines(lines[0,:,:], img)
			# if tests is not None:
			# 	for rho, theta in tests:
			# 		print rho, theta
			# 		x1,y1,x2,y2 = houghToCoords(rho, theta)
			# 		cv2.line(img_copy2,(x1,y1),(x2,y2),(0,0,255),2)
		if True:
			break
		else:
			getch = _Getch()
			cv2.imshow('img_copy', img_copy)
			cv2.imshow('lines', img_copy2)
			cv2.imshow('adaptative', _test1)
			cv2.imshow('HSV', _test3)
			cv2.imshow('merge', sand)
			print H1, S1, V1, H2, S2, V2
			print thresh1, thresh2, accu 
			cv2.waitKey(100)
			ch = getch()
			if ch=='w':
				H1 += 2
			if ch=='s':
				H1 -= 2
			if ch=='e':
				S1 += 2
			if ch=='d':
				S1 -= 2
			if ch=='r':
				V1 += 2
			if ch=='f':
				V1 -= 2
			if ch=='t':
				H2 += 2
			if ch=='g':
				H2 -= 2
			if ch=='y':
				S2 += 2
			if ch=='h':
				S2 -= 2
			if ch=='u':
				V2 += 2
			if ch=='j':
				V2 -= 2
			if ch=='q':
				break
			if ch=='p':
				ch = getch()
				if ch=='w':
					thresh1 += 2
				if ch=='s':
					thresh1 -= 2
				if ch=='e':
					thresh2 += 1
				if ch=='d':
					thresh2 -= 1
				if ch=='r':
					thresh1 += 1
				if ch=='f':
					thresh1 -= 1
				if ch=='t':
					thresh2 += 1
				if ch=='g':
					thresh2 -= 1
				if ch=='u':
					accu += 1
				if ch=='j':
					accu -= 1
				if ch=='q':
					break
	return tests

def extendVector(img, coords):
	""" Extend any vector to the borders of the image """
	xy = np.array(coords, dtype='float32')
	vector = xy[2:] - xy[:2]
	coeff = np.array([-xy[0] / vector[0], (img.shape[1] - xy[0]) / vector[0], \
					-xy[1] / vector[1], (img.shape[0] - xy[1]) / vector[1]])
	output = []
	for i in range(0, 4):
		_temp = (xy[:2] + vector * coeff[i]).astype('int')
		if (_temp[0] >= 0 and _temp[0] <= img.shape[1]) and \
			(_temp[1] >= 0 and _temp[1] <= img.shape[0]):
			if (_temp[0] == 0 or _temp[0] > img.shape[1] - 2 or \
				_temp[1] == 0 or _temp[1] > img.shape[0] - 2):
				output.append(_temp[0])
				output.append(_temp[1])
	if output[0] == img.shape[1]:
		output[0] -= 1
	if output[2] == img.shape[1]:
		output[2] -= 1
	if output[1] == img.shape[0]:
		output[1] -= 1
	if output[3] == img.shape[0]:
		output[3] -= 1
	return np.array([int(output[0]), int(output[1]), int(output[2]), int(output[3])])


def getStacks(img,video_name=None):
	""" Get the stacks as a binary mask """
	parameters = {'beachVolleyball1.mov': [5, 3, 14, 0, 0, 180, 115, 255, 255], \
				  'beachVolleyball2.mov': [5, 4, 14, 0, 0, 190, 115, 255, 255], \
				  'beachVolleyball3.mov': [5, 4, 14, 0, 0, 190, 115, 255, 255], \
				  'beachVolleyball4.mov': [5, 4, 14, 0, 0, 190, 115, 255, 255], \
				  'beachVolleyball5.mov': [5, 4, 14, 0, 0, 205, 115, 255, 255], \
				  'beachVolleyball6.mov': [5, 4, 14, 0, 0, 190, 115, 255, 255], \
				  'beachVolleyball7.mov': [1, 2, 14, 0, 0, 200, 115, 255, 255]}
	getch = _Getch()
	if video_name is None:
		iter1 = 1
		iter2 = 2
		iter3 = 14
		B1 = 0
		G1 = 0
		R1 = 200
		B2 = 115
		G2 = 255
		R2 = 255
	else:
		iter1 = parameters[video_name][0]
		iter2 = parameters[video_name][1]
		iter3 = parameters[video_name][2]
		B1 = parameters[video_name][3]
		G1 = parameters[video_name][4]
		R1 = parameters[video_name][5]
		B2 = parameters[video_name][6]
		G2 = parameters[video_name][7]
		R2 = parameters[video_name][8]
	while(True):
		kernel3x3 = np.ones((3,3))
		mask = getSandMask(img,for_stack=True)
		color1 = np.array([B1, G1, R1])
		color2 = np.array([B2, G2, R2])
		pouet = cv2.inRange(img, color1, color2)
		pouet = np.multiply(pouet, mask)
		# cv2.imshow('mask', mask)
		cv2.morphologyEx(pouet, cv2.MORPH_CLOSE, kernel3x3, dst=pouet, iterations=iter1)
		# cv2.imshow('iter1', pouet)
		cv2.morphologyEx(pouet, cv2.MORPH_OPEN, kernel3x3, dst=pouet, iterations=iter2)
		# cv2.imshow('iter2', pouet)
		cv2.morphologyEx(pouet, cv2.MORPH_CLOSE, kernel3x3, dst=pouet, iterations=iter3)
		# cv2.imshow('iter3', pouet)
		if True:
			break
		else:
			cv2.waitKey(100)
			print iter1, iter2, iter3, B1, G1, R1, B2, G2, R2
			ch=getch()
			# if ch=='w':
			# 	iter1 += 1
			# if ch=='s':
			# 	iter1 -= 1
			# if ch=='e':
			# 	iter2 += 1
			# if ch=='d':
			# 	iter2 -= 1
			# if ch=='r':
			# 	iter3 += 1
			# if ch=='f':
			# 	iter3 -= 1
			if ch=='w':
				B1 += 5
			if ch=='s':
				B1 -= 5
			if ch=='e':
				G1 += 5
			if ch=='d':
				G1 -= 5
			if ch=='r':
				R1 += 5
			if ch=='f':
				R1 -= 5
			if ch=='t':
				B2 += 5
			if ch=='g':
				B2 -= 5
			if ch=='y':
				G2 += 5
			if ch=='h':
				G2 -= 5
			if ch=='u':
				R2 += 5
			if ch=='j':
				R2 -= 5
			if ch=='q':
				break
			# break
	return pouet.astype('uint8') / 255.


def PassThroughStacks(mask, x1, y1, x2, y2):
	""" Check if the vector go through the stack (in case the net is detected as a line)"""
	vector = np.array([x2-x1,y2-y1], dtype='float32')
	nb_iterations = np.linalg.norm(vector)
	# print nb_iterations
	vector /= nb_iterations
	_pt = np.array([x1, y1], dtype='float32')
	firststack = False
	betweenstacks = False
	secondstack = False
	for i in range(0, nb_iterations):
		value = mask[int(_pt[1]), int(_pt[0])]
		if not firststack:
			if value != 0:
				firststack = True
		elif not betweenstacks:
			if value == 0:
				betweenstacks = True
		elif not secondstack:
			if value != 0:
				secondstack = True
				break
		_pt += vector
	return secondstack

def PruneNonProbaLines(lines, img, nb_max=3):
	""" Keep the most probables lines which are non the same.
		different angle and position """
	stacks_mask = getStacks(img)
	DIRECTION_THRESHOLD = math.pi * 20. / 180.
	OFFSET_THRESHOLD = 100
	final_lines = np.zeros((3,2), dtype='float32')
	u = 0
	j = 0
	while (j < nb_max) and (u < lines.shape[0]):
		img_copy = np.copy(img)
		rho = lines[u,0]
		theta = lines[u,1]
		x1,y1,x2,y2 = extendVector(img, houghToCoords(rho, theta))
		if not PassThroughStacks(stacks_mask, x1, y1, x2, y2):
			sel = True
			for i in range(0, j):
				if ((abs(theta - final_lines[i,1]) < DIRECTION_THRESHOLD) and \
						(np.abs(rho - final_lines[i,0]) < OFFSET_THRESHOLD)) or \
						((abs(abs(theta - final_lines[i,1]) - math.pi) < DIRECTION_THRESHOLD) and \
						(abs(rho + final_lines[i,0]) < OFFSET_THRESHOLD)):
					sel = False
					break
			if sel:
				final_lines[j,:] = lines[u,:]
				j += 1
		u += 1
	return final_lines[:j]

def intersections(coords1, coords2, mask, out_of_screen=False):
	""" Check if two vector intersect """
	secondpt = np.empty(2, dtype='float32')
	p = coords1[:2]
	q = coords2[:2]
	r = coords1[2:] - p
	s = coords2[2:] - q
	coeff1 = np.cross(r,s)
	coeff2 = np.cross(q - p, r)
	if np.cross(r, s) != 0:
		u = float(coeff2) / float(coeff1)
		# firstpt = p + t * r
		secondpt[:] = q + u * s
		if (secondpt[0] >= 0 and secondpt[0] < mask.shape[1] and \
			secondpt[1] >= 0 and secondpt[1] < mask.shape[0]): 
			if mask[secondpt[1], secondpt[0]] != 0:
				secondpt[:] = q + u * s
			else:
				secondpt = None 
		elif (secondpt[1] >= 0) and ((secondpt[0] >= -300) \
			and (secondpt[0] <= mask.shape[1] + 300)) and out_of_screen:
			secondpt[:] = q + u * s
		else:	
			secondpt = None
	else:
		secondpt = None
	return secondpt


def subIntersection(img, coords, line1, line2, offset, video_name):
	""" Check in the intersection detected is not a false positive with a closer look"""
	parameters = {'beachVolleyball1.mov': [5, 6, 20, 0, 24, 190, 196, 92, 224], \
				  'beachVolleyball2.mov': [5, 6, 30, 0, 40, 134, 252, 135, 228], \
				  'beachVolleyball3.mov': [5, 3, 20, 0, 20, 145, 20, 64, 220], \
				  'beachVolleyball4.mov': [5, 3, 20, 0, 20, 145, 20, 64, 220], \
				  'beachVolleyball5.mov': [5, 6, 20, 0, 20, 145, 260, 64, 195], \
				  'beachVolleyball6.mov': [5, 4, 40, 0, 35, 155, 208, 62, 195], \
				  'beachVolleyball7.mov': [5, 6, 30, 0, 24, 140, 165, 60, 210]}
	H1 = parameters[video_name][3]
	S1 = parameters[video_name][4]
	V1 = parameters[video_name][5]
	H2 = parameters[video_name][6]
	S2 = parameters[video_name][7]
	V2 = parameters[video_name][8]
	mask1 = getSandMask(img, False, True);
	mask2 = getSandMask(img, False, False);
	hsv1 = np.array([H1,S1,V1], dtype='float32')
	hsv2 = np.array([H2,S2,V2], dtype='float32')
	DIRECTION_EPSILON = math.pi * 4. / 180.
	DIRECTION_THRESHOLD = math.pi * 15. / 180.
	thresh1 = parameters[video_name][0]
	thresh2 = parameters[video_name][1]
	accu = parameters[video_name][2]
	if coords[0] >= 0 and coords[0] < img.shape[1] and \
		coords[1] >= 0 and coords[1] < img.shape[0]:
		hx1 = max(0, coords[0] - offset)
		hx2 = min(img.shape[1], coords[0] + offset)
		hy1 = max(0, coords[1] - offset)
		hy2 = min(img.shape[0], coords[1] + offset)
		hsv1 = np.array([H1,S1,V1])
		hsv2 = np.array([H2,S2,V2])
		subImg = np.copy(img[hy1:hy2,hx1:hx2,:])
		subMask1 = mask1[hy1:hy2,hx1:hx2]
		subMask2 = mask2[hy1:hy2,hx1:hx2]
		getch=_Getch()
		while(True):
			hsv1 = np.array([H1,S1,V1])
			hsv2 = np.array([H2,S2,V2])
			HSV = cv2.cvtColor(subImg, cv.CV_BGR2HSV)
			_inrange = cv2.inRange(HSV, hsv1, hsv2)
			_inrange = np.multiply(_inrange, subMask1)
			_adaptative = cv2.adaptiveThreshold(subImg[:,:,1], 255,cv2.ADAPTIVE_THRESH_MEAN_C, \
												cv2.THRESH_BINARY_INV, thresh1, thresh2)
			_thresh = np.multiply(_inrange, _adaptative / 255.).astype('uint8')
			lines = cv2.HoughLines(_thresh, 1, np.pi / 360., accu)
			if True:
				break
			else:
				cv2.imshow('_thresh', _thresh)
				cv2.imshow('_adaptative', _adaptative)
				cv2.imshow('_adaptative', _adaptative)
				cv2.imshow('_inrange', _inrange)
				copy_lines = cv2.cvtColor(_thresh, cv.CV_GRAY2BGR)
				if lines is not None:
					for rho, theta in lines[0]:
						prout = houghToCoords(rho,theta)
						cv2.line(copy_lines, (prout[0], prout[1]), (prout[2], prout[3]), (0,255,0), 2)
				cv2.imshow('copy_lines', copy_lines)

				cv2.waitKey(100)
				print H1,S1,V1,H2,S2,V2,thresh1, thresh2
				ch=getch()
				if ch=='w':
					H1 += 5
				if ch=='s':
					H1 -= 5
				if ch=='e':
					S1 += 5
				if ch=='d':
					S1 -= 5
				if ch=='r':
					V1 += 5
				if ch=='f':
					V1 -= 5
				if ch=='t':
					H2 += 5
				if ch=='g':
					H2 -= 5
				if ch=='y':
					S2 += 5
				if ch=='h':
					S2 -= 5
				if ch=='u':
					V2 += 5
				if ch=='j':
					V2 -= 5
				if ch=='i':
					thresh1 += 2
				if ch=='k':
					thresh1 -= 2
				if ch=='o':
					thresh2 += 1
				if ch=='l':
					thresh2 -= 1
				if ch=='q':
					break
				# break
		if lines is not None:
			sel_lines = np.empty((2,2), dtype='float32')
			nb_sel = 0
			for rho, theta in lines[0]:
				if (abs(theta - line1[1]) < DIRECTION_EPSILON) or (abs(theta - line2[1]) < DIRECTION_EPSILON) or \
					(abs(abs(theta - line1[1]) - math.pi) < DIRECTION_THRESHOLD) or \
					(abs(abs(theta - line2[1]) - math.pi) < DIRECTION_THRESHOLD):
					if nb_sel == 1:
						if (abs(theta - sel_lines[0][1]) > DIRECTION_THRESHOLD) or \
							(abs(abs(theta - sel_lines[0][1]) - math.pi) < DIRECTION_THRESHOLD):
							sel_lines[1,:] = [rho, theta]
							nb_sel = 2
							break
					else:
						sel_lines[0,:] = [rho, theta]
						nb_sel = 1
			if nb_sel == 2:
				output_line1 = houghToCoords(sel_lines[0,0], sel_lines[0,1])
				output_line2 = houghToCoords(sel_lines[1,0], sel_lines[1,1])
				# cv2.line(_thresh, (output_line1[0], output_line1[1]), (output_line1[2], output_line1[3]), 4, (0, 255, 0), 2)
				output = intersections(np.array(output_line1, dtype='float32'), \
								np.array(output_line2, dtype='float32'), subMask2, False)
				if output == None:
					return None, None
				output[0] += hx1
				output[1] += hy1
				return output, sel_lines
	return None, None

def getBadStacksCoords(img, threshold, video_name):
	""" Try to detect the stacks coords given the stack mask """
	stacks_mask = getStacks(img, video_name)
	pts = [None, None]
	first_stack = False
	x1 = -1
	x2 = -1
	for y in range(stacks_mask.shape[0] - 1, -1, -1):
		line = stacks_mask[y,:]
		for x in range(0, stacks_mask.shape[1]):
			if line[x] != 0:
				if x1 < 0:
					x1 = x
					y1 = 0
			if (line[x] == 0 or x == stacks_mask.shape[1] - 1) and x1 > 0 :
				x2 = x
				new_pt = np.array([int((x2 + x1) / 2),y], dtype='int')
				if first_stack:
					if np.linalg.norm(pts[0] - new_pt) > threshold:
						pts[1] = new_pt
						return pts
				else:
					pts[0] = new_pt
					first_stack = True
				x1 = -1
				x2 = -1
	return pts

def trackStack(mask):
	""" Track the stack on the stack mask """
	x1 = -1
	for y in range(mask.shape[0] - 1, -1, -1):
		line = mask[y,:]
		for x in range(0, mask.shape[1]):
			if line[x] != 0:
				if x1 < 0:
					x1 = x
					y1 = 0
			if (line[x] == 0 or x == mask.shape[1] - 1) and x1 > 0:
				return np.array([int((x + x1) / 2), y], dtype='int')
	return None

def trackCorners(img, video_name, previous_corners, previous_lines):
	""" Tracks the corners """
	resized_img = cv2.resize(img, (img.shape[1], img.shape[0] * 2), interpolation=cv.CV_INTER_LINEAR)
	stacks_mask = getStacks(resized_img, video_name)
	new_corners = [None] * len(previous_corners)
	new_lines = [None] * len(previous_corners)
	for i in range(0, len(previous_corners) - 4):
		if previous_corners[i] is not None:
			new_corners[i], new_lines[i] = subIntersection(resized_img, previous_corners[i], previous_lines[i][0], previous_lines[i][1], 60, video_name)
			if new_corners[i] is None:
				new_corners[i], new_lines[i] = subIntersection(resized_img, previous_corners[i], previous_lines[i][0], previous_lines[i][1], 80, video_name)
		else:
			pass
	for i in range(len(previous_corners) - 2, len(previous_corners)):
		if previous_corners[i] is not None:
			thx1 = max(0, previous_corners[i][0] - 30)
			thx2 = min(resized_img.shape[1], previous_corners[i][0] + 30)
			thy1 = max(0, previous_corners[i][1] - 30)
			thy2 = min(resized_img.shape[0], previous_corners[i][1] + 30)
			new_corners[i] = trackStack(stacks_mask[thy1:thy2,thx1:thx2])
			if new_corners[i] != None:
				new_corners[i][0] += thx1
				new_corners[i][1] += thy1
	return new_corners, new_lines



def getCorners(img, video_name):
	""" Get the corners """
	resized_img = cv2.resize(img, (img.shape[1], img.shape[0] * 2), interpolation=cv.CV_INTER_LINEAR)
	mask = getSandMask(resized_img, False, False)
	field_corner = []
	field_lines = []
	output_coords = [None] * 6
	output_lines = [None] * 6
	lines = getField(resized_img, video_name)
	stack_coords = getBadStacksCoords(resized_img, 200, video_name)
	img_copy = np.copy(resized_img)
	if lines is not None:
		if lines.shape[0] == 3:
			combinaisons = [[0,1],[0,2],[1,2]]
			for n1, n2 in combinaisons:
				coords1 = houghToCoords(lines[n1,0], lines[n1,1])	
				coords2 = houghToCoords(lines[n2,0], lines[n2,1])					# print flines[coords1], flines[coords2]
				inter_coords = intersections(np.array(coords1, dtype='float32'), np.array(coords2, dtype='float32'), mask, True)
				if inter_coords is not None:
					if inter_coords[0] < 0 or inter_coords[1] < 0 or inter_coords[0] > img.shape[1] - 1 or inter_coords[1] > img.shape[0] * 2 - 1:
						field_corner.append(inter_coords)
						field_lines.append(lines[[n1,n2]])
					else:
						inter_coords2 = subIntersection(resized_img, inter_coords, lines[n1], lines[n2], 60, video_name)[0]
						if inter_coords2 is None:
							inter_coords2 = subIntersection(resized_img, inter_coords, lines[n1], lines[n2], 120, video_name)[0]
						if inter_coords2 is not None:
							field_corner.append(inter_coords2)
							field_lines.append(lines[[n1,n2]])
		elif lines.shape[0] == 2:
			coords1 = houghToCoords(lines[0,0], lines[0,1])
			coords2 = houghToCoords(lines[1,0], lines[1,1])
			inter_coords = intersections(np.array(coords1, dtype='float32'), np.array(coords2, dtype='float32'), mask, True)
			if inter_coords is not None:
				if inter_coords[0] < 0 or inter_coords[1] < 0 or inter_coords[0] > img.shape[1] - 1 or inter_coords[1] > img.shape[0] * 2 - 1:
					field_corner.append(inter_coords)
					field_lines.append(lines[[0,1]])
				else:
					inter_coords2 = subIntersection(resized_img, inter_coords, lines[0], lines[1], 60, video_name)[0]
					if inter_coords2 is None:
						inter_coords2 = subIntersection(resized_img, inter_coords, lines[0], lines[1], 120, video_name)[0]
					if inter_coords2 is not None:
						field_corner.append(inter_coords2)
						field_lines.append(lines)
	return field_corner, field_lines, stack_coords, lines

def mergeCorners(img, video_name, previous_corners, previous_lines):
	""" Use all the functions above to return a corner list """
	parameters = { 'beachVolleyball1.mov': VERTICAL, \
				   'beachVolleyball2.mov': VERTICAL, \
				   'beachVolleyball3.mov': HORIZONTAL,  \
				   'beachVolleyball4.mov': HORIZONTAL, \
				   'beachVolleyball5.mov': VERTICAL, \
				   'beachVolleyball6.mov': HORIZONTAL, \
				   'beachVolleyball7.mov': HORIZONTAL}
	output_corners = [None] * len(previous_corners)
	output_lines = [None] * len(previous_corners)
	tracked_corners, tracked_lines = trackCorners(img, video_name, previous_corners, previous_lines)
	field_corners, field_lines, stack_coords, lines = getCorners(img, video_name)
	for j in range(len(previous_corners) - 2, len(previous_corners)):
		if tracked_corners[j] is not None:
			for i in range(0, len(stack_coords)):
				if stack_coords[i] is not None:
					if np.linalg.norm(tracked_corners[j] - stack_coords[i]) < 100:
						stack_coords.pop(i)
						break
	# check the number of untracked stacks + numbers of kept previous corners amongs  
	nb_untracked_stack = 0
	nb_prev_stacks = 0
	for j in range(len(previous_corners) - 2, len(previous_corners)):
		if tracked_corners[j] is None:
			nb_untracked_stack += 1
			if previous_corners[j] is not None:
				nb_prev_stacks += 1
	#add stacks to output if detected or tracked
	stack_index = 0
	for j in range(len(previous_corners) - 2, len(previous_corners)):
		if tracked_corners[j] is None:
			output_corners[j] = stack_coords[stack_index]
			stack_index += 1
		else:
			output_corners[j] = tracked_corners[j]
	unknown_indexes = []
	for i in range(0, len(tracked_corners) - 4):
		if previous_corners[i] is not None:
			for j in range(0, len(field_corners)):
				if np.linalg.norm(field_corners[j] - previous_corners[i]) < 30:
					output_corners[i] = field_corners.pop(j)
					output_lines[i] = field_lines.pop(j)
					break
		else:
			unknown_indexes.append(i)
	while len(field_corners) > 0 and len(unknown_indexes) > 0:
		index = unknown_indexes.pop(0)
		output_corners[index] = field_corners.pop(0)
		output_lines[index] = field_lines.pop(0)
	intersec = []
	_mask = np.ones((img.shape[0] * 2, img.shape[1])) * 255
	_temp = np.ones((img.shape[0] * 2, img.shape[1])) * 255
	if output_corners[-1] is not None and output_corners[-2] is not None: 
		net_line = np.array([output_corners[-1][0], output_corners[-1][1], output_corners[-2][0], output_corners[-2][1]])
		if lines is not None:
			for i in range(0, len(lines)):
				coords = houghToCoords(lines[i][0], lines[i][1])
				# print coords, net_line
				_pt = intersections(coords, net_line, _mask, parameters[video_name]==HORIZONTAL)
				# print "pt intersection: ", _pt
				if _pt is not None:
					intersec.append(_pt)
			for it in intersec:
				if np.linalg.norm(it - output_corners[-2]) < np.linalg.norm(it - output_corners[-1]):
					output_corners[-4] = it
				else:
					output_corners[-3] = it
	return output_corners, output_lines, intersec


def FindVideoCorners(video_name):
	""" Main function """
	NB_POINTS = 10
	print '../data/video/'+video_name
	cap = cv2.VideoCapture('../data/video/'+video_name)
	frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
	noneCount = np.zeros(NB_POINTS)
	values = np.ones((frameCount, 8, 2), dtype='int') * (-9999)
	output_coords = [None] * NB_POINTS
	output_lines = [None] * NB_POINTS
	for fr in range(0, frameCount-1):
		img = cap.read()[1]
		if fr % 50 == 0:
			print "FRAME: ", fr
		# if fr < 740:
		# 	continue
		output_coords, output_lines, intersec = mergeCorners(img, video_name, output_coords, output_lines)
		for i in range(0, len(output_coords) - 2):
			if output_coords[i] is None:
				noneCount[i] += 1
			else:
				x = int(output_coords[i][0])
				y = int(output_coords[i][1] / 2.)
				# print x,y
				cv2.circle(img, (x, y), 4, (0, 255, 0), 2)
				values[fr,i,0] = x
				values[fr,i,1] = y
		for i in range(len(output_coords) - 2, len(output_coords)):
			if output_coords[i] is None:
				noneCount[i] += 1
		# print "count: ", noneCount
		# cv2.imshow('final', img)
		# cv2.waitKey(1)
	cap.release
	return values

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "Not enough arguments"
		print "Usage: pointsDetection.py [video index]"
		sys.exit(1)
	np.save('../data/points/corners' + str(sys.argv[1]) + '.npy', FindVideoCorners('beachVolleyball' + str(sys.argv[1]) + '.mov'))

