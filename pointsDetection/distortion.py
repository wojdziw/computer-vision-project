import numpy as np
import cv2
from cv2 import cv
import math

def houghToCoords(rho, theta):
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	return np.array([x1,y1,x2,y2], dtype='int')

def extendVector(img, coords):
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


def PruneNonProbaLines(lines, img, nb_max=3):
	DIRECTION_THRESHOLD = math.pi * 20. / 180.
	OFFSET_THRESHOLD = 50
	final_lines = np.zeros((nb_max,3), dtype='float32')
	u = 0
	j = 0
	while (j < nb_max) and (u < lines.shape[0]):
		rho = lines[u,0]
		theta = lines[u,1]
		x1,y1,x2,y2 = extendVector(img, houghToCoords(rho, theta))
		sel = True
		for i in range(0, j):
			if ((abs(theta - final_lines[i,1]) < DIRECTION_THRESHOLD) and \
					(np.abs(rho - final_lines[i,0]) < OFFSET_THRESHOLD)) or \
					((abs(abs(theta - final_lines[i,1]) - math.pi) < DIRECTION_THRESHOLD) and \
					(abs(rho + final_lines[i,0]) < OFFSET_THRESHOLD)):
				sel = False
				break
		if sel:
			final_lines[j,:] = np.append(lines[u,:],u)
			j += 1
		u += 1
	return final_lines[:j]

def hough_line(img):
	# Rho and Theta ranges
	thetas = np.deg2rad(np.arange(-90.0, 90.0))
	width, height = img.shape
	diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

	# Cache some resuable values
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	num_thetas = len(thetas)

	# Hough accumulator array of theta vs rho
	accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
	y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

	# Vote in the hough accumulator
	t_idx = np.arange(num_thetas)
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
		# Calculate rho. diag_len is added for a positive index
		rho = np.round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
		accumulator[rho.astype('int'), t_idx.astype('int')] += 1
	return accumulator, thetas, rhos

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

def correctDistor(img, k1=0, k2=0, k3=0):
	halfWidth = img.shape[1] / 2
	halfHeight = img.shape[0] / 2
	output = np.zeros(img.shape, dtype='uint8')
	coeff = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
	for y in range(0, img.shape[0]):
		for x in range(0, img.shape[1]):
			newX = x - halfWidth
			newY = y - halfHeight
			r = np.sqrt(newX ** 2 + newY ** 2) / coeff
			theta = 1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)
			sourceX = int(halfWidth + theta * newX)
			sourceY = int(halfHeight + theta * newY)
			if sourceX >= 0 and sourceX < img.shape[1] \
					and sourceY >= 0 and sourceY < img.shape[0]:
				output[y,x] = img[sourceY,sourceX]
	return output
VIDEO_NAME = 'beachVolleyball3.mov'
cap = cv2.VideoCapture(VIDEO_NAME)

frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
print frameCount, fps, frameHeight, frameWidth
thresh1 = 50
thresh2 = 100
# k1 = 0.1
# k2 = 0.4
# k3 = -0.1
k1 = 0
k2 = 0
k3 = 0
for i in range(0, 50):
	img = cap.read()[1]
grayscale = cv2.cvtColor(img, cv.CV_BGR2GRAY)
# lines = cv2.HoughLines(thresh, 1, np.pi / 100., accu)
strength = 0
cv2.imshow('img', img)
getch = _Getch()

while (True):
	img_copy = np.copy(img)
	image = correctDistor(grayscale, k1, k2, k3)
	img_copy = correctDistor(img_copy, k1, k2, k3)
	# thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh1, thresh2)
	thresh = cv2.Canny(image, thresh1, thresh2)
	accumulator, thetas, rhos = hough_line(thresh)
	idx = np.argsort(accumulator,axis=None)[::-1][:40]
	# print idx
	x = idx / accumulator.shape[1]
	y = idx % accumulator.shape[1]
	lines = np.empty((40,2))
	lines[:,0] = rhos[x]
	lines[:,1] = thetas[y]
	output_line = PruneNonProbaLines(lines, img, 5)
	print output_line
	X = x[output_line[:,2].astype('int')]
	Y = y[output_line[:,2].astype('int')]
	print accumulator[X,Y]
	print k1, k2, k3
	for i in range(0, output_line.shape[0]):
		x1, y1, x2, y2 = extendVector(image, houghToCoords(output_line[i,0], output_line[i,1]))
		cv2.line(img_copy, (x1, y1), (x2, y2), (0,0,255), 2)
	cv2.imshow('lines', img_copy)
	cv2.imshow('output', image)
	cv2.imshow('thresh', thresh)
	cv2.waitKey(100)
	ch=getch()
	if ch == 'r':
		k1 += 0.05
	if ch == 'f':
		k1 -= 0.05
	if ch == 't':
		k2 += 0.1
	if ch == 'g':
		k2 -= 0.1
	if ch == 'y':
		k3 += 0.5
	if ch == 'h':
		k3 -= 0.5
	# if ch == 'u':
	# 	thresh2 += 2
	# if ch == 'j':
	# 	thresh2 -= 2