** CS4243 Computer Vision Project**

### Best parameters to track with (to date)

* = extra difficult to define start pos 
X = disappears out of frame
+ = gets occluded

		track	(pSize, gradDiv, th, ray, st_th, nc_th, tr_th)		
#### Vid 1:
GreenFront	Shirt	2, 10, 7, 30, 4, 1.5, 2
GreenBack	Shirt	2, 10, 7, 30, 4, 1.5, 2
WhiteLeft	+ Shorts ([69, 102]) 1, 30, 8, 10, 5, 1.5, 2		Pos saved
WhiteRight	Shorts	1, 20, 6, 8, 5, 1.5, 4				Pos saved

#### Vid 2: 	
GreenFront	Shirt	2, 10, 7, 30, 4, 1.5, 2				Pos saved
GreenBack	Shirt	2, 10, 7, 30, 4, 1.5, 2				Pos saved
WhiteLeft	+
WhiteRight	

#### Vid 3: 	
GreenDown	*
GreenUp		*+
WhiteLeft	Shorts	2, 30, 6, 30, 4, 1.5, 2
WhiteRight	Shorts	2, 10, 7, 30, 4, 1.5, 2

#### Vid 4: 	
GreenDown	*
GreenUp		*
WhiteLeft	X Shorts	3, 10, 7, 30, 6, 1.5, 2
WhiteRight	X+ Shorts	1, 10, 7, 15, 4, 1.5, 2

#### Vid 5: 	
RedFront	+ Shirt
RedBack		Shirt	3, 5, 7, 30, 4, 2.5, 2
WhiteLeft	*
WhiteRight	*+

#### Vid 6: 	
RedUp		*+ Shirt
RedDown		* Shirt
WhiteLeft	X Shoulder 2, 7, 7, 25, 5, 1.5, 2
WhiteRight	Shoulder 2, 7, 7, 25, 5, 1.5, 2

#### Vid 7: 	
RedLeft		Shirt
RedRight	X Shirt(Aim for the red!) 3, 10, 7, 25, 4, 1.5, 2 
WhiteUp		*
WhiteDown	*











