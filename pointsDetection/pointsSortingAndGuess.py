import numpy as np
import sys
import cv2
import hm_sparse
from cv2 import cv as cv


HORIZONTAL = 1
VERTICAL = 0

def getTransMatrix(points1, points2):
	""" compute translation given set of points1 and points2 """
	transMat = np.identity(3, dtype='float32')
	tx = 0.
	ty = 0.
	for i in range(0, points1.shape[0]):
		uc, vc = points1[i]
		up, vp = points2[i]
		tx += uc - up
		ty += vc - vp
	tx /= points1.shape[0]
	ty /= points1.shape[0]
	transMat[0,2] = tx
	transMat[1,2] = ty
	return transMat
	
def getTransScaleMatrix(points1, points2):
	""" compute scale + translation given set of points1 and points2 """
	if points1.shape[0] == 2:
		mix = [[0,1]]
	else:
		mix = [[0,1],[1,2],[0,2]]
	scaling = 0
	for a,b in mix:
		scaling += np.linalg.norm(points1[a] - points1[b]) / np.linalg.norm(points2[a] - points2[b])
	scaling /= len(mix)
	tx = 0.
	ty = 0.
	for i in range(0, points1.shape[0]):
		uc = points1[i][0]
		vc = points1[i][1]
		up = points2[i][0] * scaling
		vp = points2[i][1] * scaling
		tx += uc - up
		ty += vc - vp
	tx /= points1.shape[0]
	ty /= points1.shape[0]
	affine_mat = np.zeros((3,3), dtype='float32')
	affine_mat[0,0] = scaling
	affine_mat[0,2] = tx
	affine_mat[1,1] = scaling
	affine_mat[1,2] = ty
	affine_mat[2,2] = 1
	return affine_mat

def getTransform(points1, points2):
	""" Compute the transformation given the number of pts displayed """
	if points1.shape[0] > 3:
		return cv2.findHomography(points1, points2)
	elif points1.shape[0] == 1:
		return getTransMatrix(points1, points2)
	elif points1.shape[0] == 0:
		return np.identity(3)
	else:
		return getTransScaleMatrix(points1, points2)

def getRef(ref_array, orientation):
	""" sort the corner on the reference frame """
	output_array = np.ones((6,2),dtype='int') * -9999
	if orientation == HORIZONTAL:
		highest_stack = np.argmin(ref_array[-2:,1])
		output_array[-2] = ref_array[highest_stack-2]
		output_array[-1] = ref_array[-(highest_stack - 1)-2]
		for i in range(0, len(ref_array) - 2):
			if np.any(ref_array[i] != -9999):
				index = 0
				index_guess = 0
				if np.abs(ref_array[i,1] - output_array[-2,1]) > np.abs(ref_array[i,1] - output_array[-1,1]):
					index = 1
					index_guess = 1
				if ref_array[i,0] > output_array[-2,0]:
					index += 2
				else:
					index_guess += 2
				output_array[index] = ref_array[i]
				output_array[index_guess] = 2 * output_array[(index % 2) - 2] - ref_array[i]
	else:
		left_stack = np.argmin(ref_array[-2:,0])
		output_array[-2] = ref_array[left_stack-2]
		output_array[-1] = ref_array[-(left_stack - 1)-2]
		for i in range(0, len(ref_array) - 2):
			if np.any(ref_array[i] != -9999):
				if np.linalg.norm(ref_array[i] - output_array[-2]) > np.linalg.norm(ref_array[i] - output_array[-1]):
					output_array[1] = ref_array[i]
					output_array[3] = 2 * output_array[-1] - ref_array[i]
				else:
					output_array[0] = ref_array[i]
					output_array[2] = 2 * output_array[-2] - ref_array[i]
	return output_array

def changementPtsRecursion(values, begin, threshold):
	""" function which create unseen corners if we don't have enough corners on img
		 it compute it using the closest img with 4 corners """
	j = 1
	u = 0
	p = np.ones(3)
	while(np.count_nonzero(values[begin + j,:,0] != -9999) < threshold):
		j += 1
		if begin + j == values.shape[0]:
			j -= 1
			break
	while u < j:
		if np.count_nonzero(values[begin + u,:,0] != -9999) < threshold - 1:
			u += changementPtsRecursion(values, begin + u, threshold - 1)
		else:
			mask_same1 = np.logical_and(values[begin-1,:,0] != -9999, values[begin+u,:,0] != -9999)
			same1 = np.where(mask_same1)[0]
			mask_diff1 = np.logical_and(values[begin-1,:,0] != -9999, values[begin+u,:,0] == -9999)
			diff1 = np.where(mask_diff1)[0]
			prev_homography = getTransform(values[begin+u,same1,:].reshape(len(same1),2), values[begin-1,same1,:].reshape(len(same1),2))
			for i in diff1:
				p[0] = values[begin-1,i,0]
				p[1] = values[begin-1,i,1]
				pos = np.dot(prev_homography, p)
				pos /= pos[2]
				values[begin+u,i,0] = int(pos[0])
				values[begin+u,i,1] = int(pos[1])
		u += 1
	return j - 1

def changementPtsRecursionBack(values, begin, threshold):
	""" function which create unseen corners if we don't have enough corners on img
		 it compute it using the closest img with 4 corners (in backtrack direction)"""
	j = -1
	u = 0
	p = np.ones(3)
	while(np.count_nonzero(values[begin + j,:,0] != -9999) < threshold):
		j -= 1
		if begin + j < 0:
			j +=1
			break
	while u > j:
		if np.count_nonzero(values[begin + u,:,0] != -9999) < threshold - 1:
			u += changementPtsRecursion(values, begin + u, threshold - 1)
		else:
			mask_same1 = np.logical_and(values[begin,:,0] != -9999, values[begin+u-1,:,0] != -9999)
			same1 = np.where(mask_same1)[0]
			mask_diff1 = np.logical_and(values[begin,:,0] != -9999, values[begin+u-1,:,0] == -9999)
			diff1 = np.where(mask_diff1)[0]
			prev_homography = getTransform(values[begin+u-1,same1,:].reshape(len(same1),2), values[begin,same1,:].reshape(len(same1),2))
			for i in diff1:
				p[0] = values[begin,i,0]
				p[1] = values[begin,i,1]
				pos = np.dot(prev_homography, p)
				pos /= pos[2]
				values[begin+u-1,i,0] = int(pos[0])
				values[begin+u-1,i,1] = int(pos[1])
		u -= 1
	return j - 1
		
def cumulFrame2Frame(mats):
	""" cumulate frame 2 frame to have homography to the first frame """
	output_map = np.empty((mats.shape[0] + 1, mats.shape[1], mats.shape[2]))
	output_map[0] = np.eye(3)
	for fr in range(0, mats.shape[0]):
		output_map[fr+1] = np.dot(mats[fr], output_map[fr])
	return output_map

def sortCorners(video_name, corners_array):
	""" Main function: sort the corners in the array 
		[top-left, bot-left, top-right, bot-right, top stack, bot stack] """
	parameter = { 	'beachVolleyball1.mov': VERTICAL,
					'beachVolleyball2.mov': VERTICAL,
					'beachVolleyball3.mov': HORIZONTAL,
					'beachVolleyball4.mov': HORIZONTAL,
					'beachVolleyball5.mov': VERTICAL,
					'beachVolleyball6.mov': HORIZONTAL,
					'beachVolleyball7.mov': HORIZONTAL}

	cap = cv2.VideoCapture('../data/video/'+video_name)
	frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
	timeBetweenFrame = 1000 / frameCount
	noneCount = np.zeros(8)
	values = np.ones((frameCount, 6, 2), dtype='int') * (-9999)
	output_coords = [None] * 8
	output_lines = [None] * 8

	mats = hm_sparse.ptsArrToHomArr(corners_array)

	output_map = cumulFrame2Frame(mats)

	proj=np.zeros((frameHeight, frameWidth * 2, 3), dtype='uint8')

	indexes=np.where(np.ones((frameHeight, frameWidth)) == 1)
	nb_corners = corners_array.shape[1]


	### Find the first frame with the 2 stack and 2 corners
	for i in range(0, corners_array.shape[0]):
		img=cap.read()[1]
		_arr = corners_array[i]
		if (np.count_nonzero(_arr[:-2,:] != -9999) >= 2) and np.all(_arr[-2:,:] != -9999):
			index_ref_frame = i
			ref_corners = _arr
			cv2.imshow('ref', img)
			break
	cap.release()
	cap = cv2.VideoCapture('../data/video/'+video_name)

	# print index_ref_frame
	ref_corners_sorted = getRef(ref_corners, parameter[video_name])
	if video_name == 'beachVolleyball1.mov':
		for i in range(0, len(ref_corners)):
			for j in range(0, len(ref_corners_sorted)):
				if np.all(ref_corners[i] == ref_corners_sorted[j]):
					for k in range(0, values.shape[0]):
						values[k][j] = corners_array[k][i]
		return values
	ref_homography = np.copy(output_map[index_ref_frame])
	p = np.ones(3, dtype='float32')

	#### rebuild homography to reference
	for fr in range(0, corners_array.shape[0]):
		output_map[fr] = np.dot(np.linalg.inv(ref_homography), output_map[fr])

	### Sort points by proximity to reference frame (using the homography to this frame)
	for fr in range(0, corners_array.shape[0]):
		img=cap.read()[1]
		for rand_ind in range(0,nb_corners-2):
			if np.any(corners_array[fr,rand_ind] != -9999):
				p[0] = corners_array[fr,rand_ind,0]
				p[1] = corners_array[fr,rand_ind,1]
				pos = np.dot(output_map[fr], p)
				pos /= pos[2]
				for ind in range(0, 4):
					if abs(ref_corners_sorted[ind,0] - pos[0]) < 150 and abs(ref_corners_sorted[ind,1] - pos[1]) < 70:
						values[fr,ind,:] = p[:2]
						break
		for rand_ind in range(nb_corners-2, nb_corners):
			if np.any(corners_array[fr,rand_ind] != -9999):
				p[0] = corners_array[fr,rand_ind,0]
				p[1] = corners_array[fr,rand_ind,1]
				pos = np.dot(output_map[fr], p)
				pos /= pos[2]
				for ind in range(4, 6):
					if abs(ref_corners_sorted[ind,0] - pos[0]) < 150 and abs(ref_corners_sorted[ind,1] - pos[1]) < 70:
						values[fr,ind,:] = p[:2]
						break

		if np.count_nonzero(values[fr] != -9999) != np.count_nonzero(corners_array[fr] != -9999):
			for i in range(0, nb_corners):
				if np.any(corners_array[fr,i] != -9999):
					cv2.circle(img, (corners_array[fr,i,0], corners_array[fr,i,1]), 4, (0,255,0), 2)
	cap.release()
	cap = cv2.VideoCapture('../data/video/'+video_name)
	# Guessing points regarding to previous 4 corners frame
	changementPtsRecursionBack(values, index_ref_frame, 4)
	fr = index_ref_frame
	while fr < frameCount - 1:
		img = cap.read()[1]
		j = 1
		if np.count_nonzero(values[fr,:,0] != -9999) < 4:
			while(np.count_nonzero(values[fr+j,:,0] != -9999) < 4):
				j += 1
				if fr + j == values.shape[0]:
					j -= 1
					break
			u = 0
			while u < j:
				if np.count_nonzero(values[fr+u,:,0] != -9999) < 3:
					u += changementPtsRecursion(values, fr+u, 3)
				else:
					# Using last frame with 4 corners and next frame with 4 corners
					# We're gonna create new pts by computing the homography w.r.t to those frames
					mask_same1 = np.logical_and(values[fr-1,:,0] != -9999, values[fr+u,:,0] != -9999)
					mask_same2 = np.logical_and(values[fr+j,:,0] != -9999, values[fr+u,:,0] != -9999)
					same1 = np.where(mask_same1)[0]
					same2 = np.where(mask_same2)[0]
					mask_diff1 = np.logical_and(values[fr-1,:,0] != -9999, values[fr+u,:,0] == -9999)
					mask_diff2 = np.logical_and(values[fr+j,:,0] != -9999, values[fr+u,:,0] == -9999)
					mask_double_same = np.logical_and(mask_same1, mask_same2)
					mask_double_diff = np.logical_and(mask_diff1, mask_diff2)
					mask_diff1_only = np.logical_and(mask_diff1, ~mask_diff2)
					mask_diff2_only = np.logical_and(~mask_diff1, mask_diff2)
					diff1 = np.where(mask_diff1_only)[0]
					diff2 = np.where(mask_diff2_only)[0]
					double_diff = np.where(mask_double_diff)[0]
					prev_homography = getTransform(values[fr+u,same1,:].reshape(len(same1),2), values[fr-1,same1,:].reshape(len(same1),2))
					next_homography = getTransform(values[fr+u,same2,:].reshape(len(same2),2), values[fr+j,same2,:].reshape(len(same2),2))
					for i in double_diff:
						p[0] = values[fr-1,i,0]
						p[1] = values[fr-1,i,1]
						pos = np.dot(prev_homography, p)
						pos /= pos[2]
						_temp = pos
						p[0] = values[fr+j,i,0]
						p[1] = values[fr+j,i,1]
						pos = np.dot(next_homography, p)
						pos /= pos[2]
						_temp += pos
						_temp /= 2
						values[fr+u,i,0] = int(_temp[0])
						values[fr+u,i,1] = int(_temp[1])
					for i in diff1:
						p[0] = values[fr-1,i,0]
						p[1] = values[fr-1,i,1]
						pos = np.dot(prev_homography, p)
						pos /= pos[2]
						values[fr+u,i,0] = int(pos[0])
						values[fr+u,i,1] = int(pos[1])
					for i in diff2:
						p[0] = values[fr+j,i,0]
						p[1] = values[fr+j,i,1]
						pos = np.dot(next_homography, p)
						pos /= pos[2]
						values[fr+u,i,0] = int(pos[0])
						values[fr+u,i,1] = int(pos[1])
				u += 1
		fr += j
	return values

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "not enough arguments"
		print "Usage: pointsSortingAndGuess.py [video index]"
		sys.exit(1)
	np.save('../data/points/video' + str(sys.argv[1]) + '_points.npy', sortCorners('beachVolleyball' + str(sys.argv[1]) + '.mov', np.load('../data/points/corners' + str(sys.argv[1]) + '.npy')))