import numpy as np
import cv2

def getTransMatrix(points1, points2):
	transMat = np.identity(3, dtype='float32')
	tx = 0.
	ty = 0.
	for i in range(0, points1.shape[0]):
		uc, vc = points1[i]
		up, vp = points2[i]
		tx += uc - up
		ty + vc - vp
	tx /= points1.shape[0]
	ty /= points1.shape[0]
	transMat[0,2] = tx
	transMat[1,2] = ty
	return transMat

def getAffineMatrix(points1, points2):
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

MISSING = [-9999, -9999]
# MISSING = [-10099, -10099]

def ptsArrToHomArr(pts):
	N, d, _ = pts.shape
	hom_arr = np.empty([N - 1, 3, 3]) # Holds the homographies between every two frames
	refPoints = pts[0]
	for i in range(1, N):
		curr = pts[i]
		prev = pts[i - 1]
		currActive = []
		prevActive = []
		matchCurr = []
		matchRef = []
		# Collect all the points that are found in both frames
		# Note: This assumes that if there is an entry in a column in two
		# 		consecutive frames, it will refer to the position of the same point
		for j in range(d):
			if curr[j].tolist() != MISSING and prev[j].tolist() != MISSING:
				currActive.append(curr[j].tolist())
				prevActive.append(prev[j].tolist())
			if curr[j].tolist() != MISSING and refPoints[j].tolist() != MISSING:
				matchCurr.append(curr[j].tolist())
				matchRef.append(refPoints[j].tolist())
		if len(currActive) == 0:
			hom_arr[i - 1] = np.eye(3) # If we have no points, we do nothing
		elif len(currActive) == 1:
			# hom_arr[i - 1] = np.eye(3) # If we have no points, we do nothing
			hom_arr[i - 1] = np.matrix(getTransMatrix(np.array(prevActive), np.array(currActive)))
		elif len(currActive) < 4:
			# hom_arr[i - 1] = np.eye(3) # If we have no points, we do nothing
			hom_arr[i - 1] = np.matrix(getAffineMatrix(np.array(prevActive), np.array(currActive)))
		elif len(currActive) > 3:
			hom_arr[i - 1] = np.matrix(cv2.findHomography(np.array(currActive, dtype='float'), np.array(prevActive, dtype='float'))[0])
			refPoints = curr
		# elif len(matchCurr) > 3:
		# 	hom_arr[i - 1] = np.matrix(cv2.findHomography(np.array(matchCurr, dtype='float'), np.array(matchRef, dtype='float'))[0])
		# 	refPoints = curr
		else:
			print 'wtf'
			hom_arr[i - 1] = np.matrix(np.eye(3))

	return hom_arr
