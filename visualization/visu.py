import numpy as np
import cv2 as cv2
import cv2.cv as cv

class Rect:

	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height

class Player:

	def __init__(self, x, y, number, color, distance, jumping, jumps):
		self.x = x
		self.y = y
		self.number = number
		self.color = color
		self.distance = distance
		self.jumping = jumping
		self.jumps = jumps

class Field:

	def __init__(self, width, height):
		self.width = width
		self.height = height

class View:
	
	def __init__(self, children):
		self.children = children

	def draw(self, screen, rect):
		for child in children:
			child.draw(rect)

class HorizontalLayout(View):

	def __init__(self, children, spacing):
		self.children = children
		self.spacing = spacing

	def draw(self, screen, rect):
		childSize =  (rect.width )/ len(self.children)
		print(str(childSize) + " horizontal size")
		i = 0
		for child in self.children:
			r =  Rect(rect.x + i*childSize , rect.y, childSize - self.spacing, rect.height)
			child.draw(screen, r)
			i = i+1

class VerticalLayout(View):

	def __init__(self, children, spacing):
		self.children = children
		self.spacing = spacing

	def draw(self, screen, rect):
		childSize =  (rect.height )/ len(self.children)
		print(str(childSize) + " vertical size")
		i = 0
		for child in self.children:
			r =  Rect(rect.x  , rect.y + i*childSize, rect.width , childSize - self.spacing)
			child.draw(screen, r)
			i = i+1

class PlayerView(View):

		def __init__(self, player):
			self.player = player

		def draw(self, screen, rect):
			rad = min(rect.width, rect.height) / 2
			x = rect.x + rad
			y = rect.y + rad
			cv2.circle(screen,(x,y), rad, self.player.color, -1) 
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(screen,str(self.player.number),(rect.x+rad,rect.y+rad), font, 1, (255,255,255),2, cv2.CV_AA)

class FieldView(View):

	def __init__(self, field, players):
			self.field = field
			self.players = list()
			for player in players:
				self.players.append(PlayerView(player))


	def draw(self, screen, rect):

		fieldRatio = (float)(self.field.height) / (float)(self.field.width)

		if fieldRatio * rect.width <= rect.height:
			fieldDrawnWidth= rect.width
			fieldDrawnHeight= (int)(fieldRatio * rect.width)
		else:	
			fieldDrawnWidth= (int)(rect.width * 1.0/fieldRatio)
			fieldDrawnHeight= rect.height

		print(rect.width)
		print(rect.height)
		print(fieldDrawnWidth)
		print(fieldDrawnHeight)
		print(fieldRatio)	

		xFieldRatio = (float)(self.field.width) / fieldDrawnWidth
		yFieldRatio = (float)(self.field.height) / fieldDrawnHeight

		d = list()

		for player in self.players:
			px = (int)( rect.x + (float)(player.player.x) / self.field.width * fieldDrawnWidth)
			py = (int)( rect.y + (float)(player.player.y) / self.field.height * fieldDrawnHeight)
			player.draw(screen, Rect(px -10, py - 10, 20, 20))
			d.append(player.player.distance)		

		fieldRect = Rect(rect.x, rect.y, fieldDrawnWidth, fieldDrawnHeight)
		drawFieldBorders(screen, fieldRect, self.players)
			
class ImageView(View):
	
	def __init__(self, image):
		self.image = image

	def draw(self, screen, rect):
		res = cv2.resize(self.image, (rect.width, rect.height), interpolation = cv2.INTER_CUBIC)
		screen[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width] = res

class ViewBorder(View):

	def __init__(self, view):
		self.view = view

	def draw(self, screen, rect):

		cv2.rectangle(screen,(rect.x, rect.y),(rect.x+rect.width, rect.y+rect.height),(255,255,255),3)
		self.view.draw(screen, rect)

class ViewPadding(View):

	def __init__(self, view, padding):
		self.view = view
		self.padding = padding

	def draw(self, screen, rect) :

		r = Rect(rect.x+self.padding, rect.y+self.padding, rect.width - 2*self.padding, rect.height - 2*self.padding)
		self.view.draw(screen, r)

def drawFieldBorders(screen, rect, players):
	#draw borders
	cv2.rectangle(screen,(rect.x, rect.y),(rect.x+rect.width, rect.y+rect.height),(255,255,255),3)

	#draw net
	cv2.line(screen,(rect.x+rect.width/2, rect.y),(rect.x+rect.width/2, rect.y+rect.height),(255,0,0),5)
	font = cv2.FONT_HERSHEY_SIMPLEX
	colEsp = rect.width / 3 / 2
	rowEsp = rect.height / 2 / 4

	cv2.putText(screen,"d",(rect.x +colEsp, rect.y+rect.height+rowEsp), font, 1,(255,255,255),2, cv2.CV_AA)
	cv2.putText(screen,"jumping",(rect.x+colEsp*2, rect.y+rect.height+rowEsp), font, 1,(255,255,255),2, cv2.CV_AA)
	cv2.putText(screen,"jumps",(rect.x+colEsp*4, rect.y+rect.height+rowEsp), font, 1,(255,255,255),2, cv2.CV_AA)

	i = 0
	for player in players:
		jumping = "False"

		if player.player.jumping == True:
			jumping = "True"

		cv2.putText(screen,"p"+str(i+1),(rect.x, rect.y+rect.height+rowEsp*(2+i)), font, 1,(255,255,255),2, cv2.CV_AA)
		#draw distance
		cv2.putText(screen,str((int)(player.player.distance)),(rect.x +colEsp, rect.y+rect.height+rowEsp*(2+i)), font, 1,(255,255,255),2, cv2.CV_AA)
		#draw jumping
		cv2.putText(screen,jumping,(rect.x +colEsp*2, rect.y+rect.height+rowEsp*(2+i)), font, 1,(255,255,255),2, cv2.CV_AA)
		#draw number of jumps
		cv2.putText(screen,str(player.player.jumps),(rect.x +colEsp*4, rect.y+rect.height+rowEsp*(2+i)), font, 1,(255,255,255),2, cv2.CV_AA)
		i = i + 1

	return screen

def drawPlayer(screen, number, color, x, y):

	cv2.circle(screen,(x,y), 20, color, -1) 
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(screen,str(number),(x-10,y+10), font, 1,(255,255,255),2, cv2.CV_AA)


def drawScreen(players, field, img1, img2):
	img = np.zeros((1000,1200,3), np.uint8)

	views = list()

	views.append(ViewBorder(ImageView(img1)))

	views.append(FieldView(field, players))
	horizontaLayout =  ViewPadding(HorizontalLayout(views, 20), 10)

	vert = list()
	vert.append(horizontaLayout)
	vert.append(ViewBorder(ImageView(img2)))

	verticalLayout = ViewBorder(ViewPadding(VerticalLayout(vert, 20), 20))

	verticalLayout.draw(img, Rect(0, 0, 1200, 1000))

	return img

def visu():
	 # Create a black image
	img = np.zeros((100,100,3), np.uint8)
	 
	 # Draw a diagonal blue line with thickness of 5 px
	cv2.line(img,(0,0),(511,511),(255,0,0),5)

	cv2.rectangle(img,(384,0),(510,128),(0,255,0),3) 
	cv2.circle(img,(447,63), 63, (0,0,255), -1) 
	cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
	pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
	pts = pts.reshape((-1,1,2))
	cv2.polylines(img,[pts],True,(0,255,255))
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2, cv2.CV_AA)

	flowers = cv2.imread("volley.jpg")
	width = flowers.shape[1]
	height = flowers.shape[0]
	img[0:height, 0:width] = flowers

	cv2.imshow("draw tuto", img)
	cv2.waitKey(0)