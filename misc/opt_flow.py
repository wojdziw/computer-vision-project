import numpy as np
import cv2
import cv2.cv as cv


def getSandMask(img, mask_border=False, mask_logo=False):
	kernel3x3 = np.ones((3,3))
	hue = cv2.cvtColor(img,cv.CV_BGR2HSV)[:,:,0]
	ret, _temp = cv2.threshold(hue,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	high_thresh = cv2.morphologyEx(_temp, cv2.MORPH_OPEN,kernel3x3, iterations = 20)
	cv2.morphologyEx(high_thresh, cv2.MORPH_CLOSE,kernel3x3, dst=high_thresh, iterations = 20)
	if mask_border:
		high_thresh[0,:] = 0
		high_thresh[-1,:] = 0
		high_thresh[:,0] = 0
		high_thresh[:,-1] = 0
	if mask_logo:
		LOGO_X1 = 582
		LOGO_X2 = 626
		LOGO_Y1 = 253
		LOGO_Y2 = 288
		high_thresh[LOGO_Y1:LOGO_Y2,LOGO_X1:LOGO_X2] = 0
	return cv2.erode(high_thresh, kernel3x3,iterations=3).astype('uint8') / 255.


cap = cv2.VideoCapture('../videos/beachVolleyball7.mov')
 
 # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                        qualityLevel = 0.01,
                        minDistance = 3,
                        blockSize = 7 )
 
 # Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                   maxLevel = 2,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 

while(1):
    ret, frame = cap.read()
    cv2.imshow("hey", getSandMask(frame))
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
   
 
cv2.destroyAllWindows()
cap.release()
