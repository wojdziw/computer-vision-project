import cv2 as cv2
import cv2.cv as cv
import numpy as np
import math as mth
import visu as v

def main():

	players = list()
	players.append(v.Player(50, 50, 1, (255,0,255)))
	players.append(v.Player(100, 100, 2, (255,255,0)))
	players.append(v.Player(400, 100, 3, (255,255,0)))
	players.append(v.Player(500, 300 ,54, (255,255,0)))

	field  = v.Field(500,300)

	image = cv2.imread("volley.jpg")

	v.drawScreen(players, field, image, image)

if __name__ == '__main__':
	main()
