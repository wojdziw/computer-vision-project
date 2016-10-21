from pickStartPoint import *
import cv2

image = cv2.imread("frame1.jpg")
a, b = indicatePosition(image)
print(a, b)