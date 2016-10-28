import cv2 as cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt


def rotatePointOfAngle(p, angle, axis):
	angleRad = angleDegToRad(angle)
	pq = getQuaternionFromPoint(p)
	q = quaternionFromAngle(angleRad, axis)
	qstr = getOppositeOfQuaternion(q)
	
	newP = np.zeros(4)
	newP = quatmult(q, pq)
	newP = quatmult(newP, qstr)
	
	return newP[1:]
	
def angleDegToRad(angle):
	return angle * np.pi / 180.0

def getQuaternionFromPoint(point):
	out = np.zeros(4)
	out[1:] = point
	return out

def quaternionFromAngle(angle, axis):
	out = np.zeros(4);
	out[0] = np.cos([angle/2])
	sin = np.sin(np.array([angle/2]))
	out[1:] =  sin[0] * axis
	return out

def quaternionFromDegAngle(angle, axis):
	return quaternionFromAngle(angleDegToRad(angle),axis)

def getOppositeOfQuaternion(q):
	out = np.zeros(4)
	out[0] = q[0]
	out[1:] = -1 * q[1:]
	return out

def quatmult(q1, q2):
	out = np.zeros(4)
	out[0] = q1[0]*q2[0] - np.dot(q1[1:], q2[1:])
	out[1:] = np.cross(q1[1:], q2[1:]) + q2[0] * q1[1:] + q1[0] * q2[1:]

	return out

def  quat2rot(q):
	out = np.mat('0. 0. 0.; 0. 0. 0.; 0. 0. 0.')

	out[0, 0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
	out[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
	out[0,2] = 2*(q[1]*q[3] + q[0]*q[2])
	
	out[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
	out[1,1] = q[0]*q[0] + q[2]*q[2] - q[1]*q[1] - q[3]*q[3]
	out[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
	
	out[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
	out[2,1] = 2*(q[2]*q[3] + q[0]*q[1])
	out[2,2] = q[0]*q[0] + q[3]*q[3] - q[1]*q[1] - q[2]*q[2]

	return out

def getPerspectiveProjectionPoint(sp, t, o, f, u0, v0, bu, bv):
	diff = np.matrix(sp-t)
	u = f * np.dot(diff,np.transpose(o[0,0:])) * bu / np.dot(diff, np.transpose(o[2,0:])) + u0
	v = f * np.dot(diff , np.transpose(o[1,0:])) * bu /  np.dot(diff , np.transpose(o[2,0:])) + v0

	return u, v

def getOrthographicProjectionPoint(sp, t, o, f, u0, v0, bu, bv):
	diff = np.matrix(sp-t)
	u = f * np.dot(diff,np.transpose(o[0,0:])) * bu  + u0
	v = f * np.dot(diff , np.transpose(o[1,0:])) * bu + v0

	return u, v

def main():
	pts = np.zeros([11, 3])
	pts[0, :] = [-1, -1, -1]
	pts[1, :] = [1, -1, -1]
	pts[2, :] = [1, 1, -1]
	pts[3, :] = [-1, 1, -1]
	pts[4, :] = [-1, -1, 1]
	pts[5, :] = [1, -1, 1]
	pts[6, :] = [1, 1, 1]
	pts[7, :] = [-1, 1, 1]
	pts[8, :] = [-0.5, -0.5, -1]
	pts[9, :] = [0.5, -0.5, -1]
	pts[10, :] = [0, 0.5, -1]

	#out = quatmult([1, 2, 3 , 4], [1, 2 ,3 ,4])
	axis = np.zeros(3)
	axis[0:] = [0, 1, 0]
	p = np.zeros(3)
	p[0:] = [0, 0, -5]

	pt1 = p
	pt2 = rotatePointOfAngle(p, -30, axis)
	pt3 = rotatePointOfAngle(p, -60, axis)
	pt4 = rotatePointOfAngle(p, -90, axis)

	quatmat_1 = quat2rot(quaternionFromDegAngle(0, axis)) * np.matrix(np.identity(3))
	quatmat_2 = quat2rot(quaternionFromDegAngle(30, axis)) * np.matrix(quatmat_1)
	quatmat_3 = quat2rot(quaternionFromDegAngle(60, axis)) * np.matrix(quatmat_1)
	quatmat_4 = quat2rot(quaternionFromDegAngle(90, axis)) * np.matrix(quatmat_1)
	
	#shape points
	plt.subplot(2,2,1)
	for i in range(0,11):
		u1, v1 = getPerspectiveProjectionPoint(pts[i,:], pt1, quatmat_1, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,2)
	for i in range(0,11):
		u1, v1 = getPerspectiveProjectionPoint(pts[i,:], pt2, quatmat_2, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,3)
	for i in range(0,11):
		u1, v1 = getPerspectiveProjectionPoint(pts[i,:], pt3, quatmat_3, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,4)
	for i in range(0,11):
		u1, v1 = getPerspectiveProjectionPoint(pts[i,:], pt4, quatmat_4, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.figure()
	plt.subplot(2,2,1)
	for i in range(0,11):
		u1, v1 = getOrthographicProjectionPoint(pts[i,:], pt1, quatmat_1, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,2)
	for i in range(0,11):
		u1, v1 = getOrthographicProjectionPoint(pts[i,:], pt2, quatmat_2, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,3)
	for i in range(0,11):
		u1, v1 = getOrthographicProjectionPoint(pts[i,:], pt3, quatmat_3, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.subplot(2,2,4)
	for i in range(0,11):
		u1, v1 = getOrthographicProjectionPoint(pts[i,:], pt4, quatmat_4, 1, 0, 0, 1, 1)
		plt.scatter(u1, v1)

	plt.show()
	#homography
	homoPoints = [pts[0,:],pts[1,:], pts[2,:], pts[3,:], pts[8,:]]	
	camPoints = list()
	for point in homoPoints:
		camPoints.append(getPerspectiveProjectionPoint(point, pt3, quatmat_3, 1, 0, 0, 1, 1))

	M = np.matrix(np.zeros((10,9)))

	i = 0
	for i in range(0,len(camPoints)):
		uc, vc = camPoints[i]
		up = homoPoints[i][0]
		vp = homoPoints[i][1]
		zp = homoPoints[i][2]
		M[2*i,:] = [up, vp, zp, 0, 0, 0, -uc*up, -uc*vp, -zp*uc]
		M[2*i+1,:] = [0, 0, 0, up, vp, zp, -vc*up, -vc*vp, -zp*vc]

	U, s, V = np.linalg.svd(M, full_matrices=True)
	V = np.transpose(V)
	S = np.diag(s)

	indexes = list()
	for i in range(0, S.shape[0]):
		array = S[i,:]
		nz = np.nonzero(array > 0.00001)
		if not len(nz[0]):
			indexes.append(i)

	resVector = np.zeros(9)	
	
	for index in indexes:
		resVector = resVector + np.transpose(V[:, i])

	homography = np.reshape(resVector, (3, 3))
	homography = homography / homography[2,2]
	print(homography)
	uc, vc = camPoints[0]

if __name__ == '__main__':
	main()
