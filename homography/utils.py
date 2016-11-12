import numpy as np
import numpy
import cv2

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def computeNewPos(p, H):
    '''
        Compute the image of p under transformation with H
    '''
    old_hc = np.matrix([p[0], p[1], 1]).T
    new_hc = np.matrix(H) * old_hc
    u = round(new_hc[0,0] / float(new_hc[2, 0]))
    v = round(new_hc[1,0] / float(new_hc[2, 0]))
    return np.array([u, v])

def myWarpPerspective(img, H):
    '''
        img -> numpy array
        H -> 3x3 homography matrix
        returns:
            new -> The image if img under H
    '''
    new = np.zeros(img.shape)
    H_inv = np.linalg.inv(H)
    for v in range(img.shape[0]):
        for u in range(img.shape[1]):
            u_pre, v_pre = computeNewPos([u, v], H_inv)
            if 0 <= u_pre < img.shape[1] and 0 <= v_pre < img.shape[0]:
                new[v, u] = img[v_pre, u_pre]
    return new


def smoothPointsArray(points, MISSING=-9999, ks=19):
    '''
        points -> F x P x 2 array, where F is the number of frames, P the number of points
                and points[i, j] hold the x,y coords of point j in frame i
        MISSING -> The value which replaces coordinates for unknown points in the array
        ks -> window size for the smoothing
        returns - an array with the same length, with all continuos patches smoothened
    '''
    F, P, d = points.shape
    result = np.array(points)

    currentSeqStarts = - np.ones(P)

    for i, pos in enumerate(points[0, :, :]):
        if not (pos[0] == pos[1] == MISSING):
            currentSeqStarts[i] = 0

    # Add a dummy MISSING row in the end so as not to worry about edge cases
    dummy = MISSING * np.ones([1, P, d])
    points = np.concatenate((points, dummy), axis=0)

    for i in range(F + 1):
        row = points[i, :, :]
        for j in range(P):
            pos = row[j, :]
            if pos[0] == pos[1] == MISSING: # We don't see this point in this frame
                if currentSeqStarts[j] >= 0: # We just lost this point, we want to smooth the patch
                    seqStart = currentSeqStarts[j]
                    seqEnd = i
                    if ks <= seqEnd - seqStart: # Make sure the patch is long enough to convolve with our kernel
                        for k in range(d):
                            smoothened = smooth(points[int(seqStart): int(seqEnd), j, k], window_len=ks)
                            result[int(seqStart):int(seqEnd), j, k] = smoothened[int(ks/2) : int(-ks/2 + 1)]
                    currentSeqStarts[j] = -1 #We're not seeing this point anymores
            else: # We see this point in this frame
                if points[i - 1, j, 0] == points[i - 1, j, 1] == MISSING: # We did not have it in the last frame
                    currentSeqStarts[j] = i
    # Remove dummy row, return result
    points = points[:-1, :, :]
    return result


def getCorrespondingPointsTuples(row, refPoints, refIndices, MISSING=-9999):
    # Collect tuples points in the current frame that are also in the reference frame
    correspondingTuples = []
    for index, point in zip(refIndices, refPoints):
        if not (row[index, 0] == row[index, 1] == MISSING):
            correspondingTuples.append((point, row[index].tolist()))
    return correspondingTuples

def findActivePoints(row, MISSING):
    # Which poits in this row can we 'see'?
    activeIndices = []
    activePoints = []
    for i, p in enumerate(row):
        if not (p[0] == p[1] == MISSING):
            activeIndices.append(i)
            activePoints.append(p.tolist())
    return activeIndices, activePoints

def getTransMatrix(points1, points2):
    '''
        Returns a translation between points1 and points 2
    '''
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
    '''
        Returns the affine transform from points1 to points2
    '''
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
    	uc = points1[i][0] * scaling
    	vc = points1[i][1] * scaling
    	up = points2[i][0] * scaling
    	vp = points2[i][1] * scaling
    	tx += uc - up
    	ty += vc - vp
    tx /= points1.shape[0]
    ty /= points1.shape[0]
    affine_mat = np.zeros((3,3), dtype='float32')
    affine_mat[0,0] = scaling
    affine_mat[0,2] = tx * scaling
    affine_mat[1,1] = scaling
    affine_mat[1,2] = ty * scaling
    affine_mat[2,2] = 1
    return affine_mat


def getFrameToFrameHomographies(pts, MISSING = [-9999, -9999]):
    '''
        points -> F x P x 2 array, where F is the number of frames, P the number of points
                and points[i, j] hold the x,y coords of point j in frame i
        MISSING -> The value which replaces coordinates for unknown points in the array
        returns:
            hom_arr -> F-1 x 3 x 3 array, where hom_arr[9] is the homoraphy induced
                betweem frame i and frame i + 1
        NOTE:
            In cases where we have less than 4 points from frame to frame,
            we approximate the homography by an affine transformation
    '''
    N, d, _ = pts.shape
    hom_arr = np.empty([N - 1, 3, 3]) # Holds the homographies between every two frames
    for i in range(1, N):
    	curr = pts[i]
    	prev = pts[i - 1]
    	currActive = []
    	prevActive = []
    	# Collect all the points that are found in both frames
    	# Note: This assumes that if there is an entry in a column in two
    	# 		consecutive frames, it will refer to the position of the same point
    	for j in range(d):
    		if curr[j].tolist() != MISSING and prev[j].tolist() != MISSING:
    			currActive.append(curr[j].tolist())
    			prevActive.append(prev[j].tolist())
    	if len(currActive) == 0:
    		hom_arr[i - 1] = np.eye(3) # If we have no points, we do nothing
    	elif len(currActive) == 1:
    		hom_arr[i - 1] = np.matrix(getTransMatrix(np.array(prevActive), np.array(currActive)))
    	elif len(currActive) < 4:
    		hom_arr[i - 1] = np.matrix(getAffineMatrix(np.array(prevActive), np.array(currActive)))
    	else: # len(currActive) > 3:
    		hom_arr[i - 1] = np.matrix(cv2.findHomography(np.array(currActive, dtype='float'), np.array(prevActive, dtype='float'))[0])

    return hom_arr

def generateHomographiesFromPoints(points, refIndex, frameByFrame, MISSING=-9999):
    '''
        points -> F x P x 2 array, where F is the number of frames, P the number of points
                and points[i, j] hold the x,y coords of point j in frame i
        refIndex -> scalar between 0 and F-1 - reference frame index
        MISSING -> The value which replaces coordinates for unknown points in the array
        returns homographies - an F x 3 x 3 array, where result[i] is the homography
                induced between frame i and the reference frame
    '''
    F, P, d = points.shape
    homographies = np.zeros([F, 3, 3])

    # Which points are present in our reference frame?
    refIndices = []
    refPoints = []
    for i, p in enumerate(points[refIndex]):
        if not(p[0] == p[1] == MISSING):
            refIndices.append(i)
            refPoints.append(p.tolist())
    # Look at the following frames
    for i in range(F):
        row = points[i]
        if i == refIndex: # This is our reference frame
            homographies[i] = np.eye(3)
        else:
            currentHomTuples = getCorrespondingPointsTuples(row, refPoints, refIndices, MISSING)
            if len(currentHomTuples) > 3: # We have enough points to compute a homography. Yay!
                foundRefPoints = np.array([r for (r, c) in currentHomTuples], dtype='float')
                currentPoints = np.array([c for (r, c) in currentHomTuples], dtype='float')
                homographies[i] = cv2.findHomography(currentPoints, foundRefPoints)[0]
            else: # We don't have enough points in the current frame. Bugger!
                # Find the points that are seen in this frame
                currentActiveIndices, currentActivePoints = findActivePoints(row, MISSING)

                # Traverse the array until we find a frame that has 4 points that
                # Are both in the reference frame and in this one
                foundAGoodFrame = False
                for k in range(1, F - i):
                    if (i - k) >= 0 and not foundAGoodFrame:
                        rowToCheck = points[i - k]
                        correspondingToCurrent = getCorrespondingPointsTuples(rowToCheck, currentActivePoints, currentActiveIndices, MISSING)
                        correspondingToRef = getCorrespondingPointsTuples(rowToCheck, refPoints, refIndices, MISSING)
                        if len(correspondingToCurrent) > 3 and len(correspondingToRef) > 3:
                            foundAGoodFrame = True
                            currentToCheckPointsInCheck = np.array([ch for (curr, ch) in correspondingToCurrent], dtype='float')
                            currentToCheckPointsInCurrent = np.array([curr for (curr, ch) in correspondingToCurrent], dtype='float')
                            checkToRefPointsInCheck = np.array([ch for (ref, ch) in correspondingToRef], dtype='float')
                            checkToRefPointsInRef = np.array([ref for (ref, ch) in correspondingToRef], dtype='float')
                            currentToCheckHomography = np.matrix(cv2.findHomography(currentToCheckPointsInCurrent, currentToCheckPointsInCheck)[0])
                            checkToRefHomography = np.matrix(cv2.findHomography(checkToRefPointsInCheck, checkToRefPointsInRef)[0])
                            homographies[i] = np.array(checkToRefHomography * currentToCheckHomography)
                    if (i + k) < F and not foundAGoodFrame:
                        rowToCheck = points[i + k]
                        correspondingToCurrent = getCorrespondingPointsTuples(rowToCheck, currentActivePoints, currentActiveIndices, MISSING)
                        correspondingToRef = getCorrespondingPointsTuples(rowToCheck, refPoints, refIndices, MISSING)
                        if len(correspondingToCurrent) > 3 and len(correspondingToRef) > 3:
                            foundAGoodFrame = True
                            currentToCheckPointsInCheck = np.array([ch for (curr, ch) in correspondingToCurrent], dtype='float')
                            currentToCheckPointsInCurrent = np.array([curr for (curr, ch) in correspondingToCurrent], dtype='float')
                            checkToRefPointsInCheck = np.array([ch for (ref, ch) in correspondingToRef], dtype='float')
                            checkToRefPointsInRef = np.array([ref for (ref, ch) in correspondingToRef], dtype='float')
                            currentToCheckHomography = np.matrix(cv2.findHomography(currentToCheckPointsInCurrent, currentToCheckPointsInCheck)[0])
                            checkToRefHomography = np.matrix(cv2.findHomography(checkToRefPointsInCheck, checkToRefPointsInRef)[0])
                            homographies[i] = np.array(checkToRefHomography * currentToCheckHomography)
                if not foundAGoodFrame:
                    # We'll find the closest frame that has at least 4 points
                    # of those in our original frame and compute the homography
                    # by multiplying the frame-by-frame homorgaphies
                    fbfF = np.matrix(np.eye(3))
                    fbfB = np.matrix(np.eye(3))
                    prevRow = row
                    for k in range(1, max(F - i, i - refIndex)):
                        if not foundAGoodFrame and i - k >= 0:
                            rtc = points[i - k]
                            corrt = getCorrespondingPointsTuples(rtc, refPoints, refIndices, MISSING)
                            if len(corrt) > 3:
                                corrRef = np.array([r for (r, c) in corrt], dtype='float')
                                corrC = np.array([c for (r, c) in corrt], dtype='float')
                                H = np.matrix(cv2.findHomography(corrC, corrRef)[0])
                                homographies[i] = H * fbfB
                                foundAGoodFrame = True
                            else:
                                fbfB = np.matrix(frameByFrame[i - k]) * fbfB
                            prevRow = rtc
                        if not foundAGoodFrame and i + k < F - 1:
                            rtc = points[i + k]
                            corrt = getCorrespondingPointsTuples(rtc, refPoints, refIndices, MISSING)
                            if len(corrt) > 3:
                                corrRef = np.array([r for (r, c) in corrt], dtype='float')
                                corrC = np.array([c for (r, c) in corrt], dtype='float')
                                H = np.matrix(cv2.findHomography(corrC, corrRef)[0])
                                homographies[i] = H * fbfF
                                foundAGoodFrame = True
                            else:
                                fbfF = np.matrix(np.linalg.inv(frameByFrame[i + k])) * fbfF
                            prevRow = rtc


    return homographies

def getMinMaxWidthHeight(frame_width, frame_height, homographies):
    '''
        Get the minimum and maximum coordinates in the x and y direction
        that these homographies yield
    '''
    # Create a matrix with the coordinates of the four corners
    posArray = np.array([[0, 0, 1], [frame_width, 0, 1], [0, frame_height, 1], [frame_width, frame_height, 1]]).T
    uMinG = 0
    uMaxG = frame_width
    vMinG = 0
    vMaxG = frame_height
    for i in range(homographies.shape[0]):
        H = homographies[i]
        # Get the coordinates of the four corners in the warped frame
        newPoints = np.array(np.matrix(H) * np.matrix(posArray))
        newPoints[0, :] /= newPoints[2, :]
        newPoints[1, :] /= newPoints[2, :]
        uMinG = int(round(min(uMinG, np.min(newPoints[0, :]))))
        uMaxG = int(round(max(uMaxG, np.max(newPoints[0, :]))))
        vMinG = int(round(min(vMinG, np.min(newPoints[1, :]))))
        vMaxG = int(round(max(vMaxG, np.max(newPoints[1, :]))))

    uMinG = max(-frame_width, uMinG)
    uMaxG = min(2 * frame_width, uMaxG)
    vMinG = max(-frame_height, vMinG)
    vMaxG = min(2 * frame_height, vMaxG)
    return uMinG, uMaxG, vMinG, vMaxG

def getFilenamesForIndex(i):
    points_file = '../data/points/video' + str(i) + '_points.npy'
    source_video_filename = '../data/videos/beachVolleyball' + str(i) + '.mov'
    background_filename = '../data/background' + str(i) + '_background.jpg'
    stitched_video_filename = '../data/stitched/stitching_' + str(i) + '.avi'
    background_substracted_video_filename = '../data/video' + str(i) + '_background_substracted.avi'
    return points_file, source_video_filename, background_filename, stitched_video_filename, background_substracted_video_filename
