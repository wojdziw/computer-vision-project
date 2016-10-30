# CS4243 Computer Vision Project

### Best parameters to track with (to date)

###### Explanations:
* (*) = extra difficult to define start pos 
* (X) = disappears out of frame
* (+) = gets occluded

		track	(pSize, gradDiv, th, ray, st_th, nc_th, tr_th)		
#### Vid 1:
| Player | What to track | pSize | gradDiv | th | ray | st-th | nc-th | tr-th | Status |

|------------|:-------:|:---:|:----:|:---:|:---:|:---:|:---:|:---:|:------------:|

| GreenFront | Shirt | 2 | 10 | 7 | 25 | 4 | 1.8 | 5 | Pos, area saved |

| GreenBack | Shirt | 2 | 15 | 7 | 25 | 4 | 1.5 | 8 | Pos, area saved |

| WhiteLeft | +Shorts([69, 102]) | 1 | 30 | 8 | 10 | 5 | 1.5 | 2 | Pos, area saved |

| WhiteRight | Shorts([60, 174]) | 1 | 10 | 5 | 8 | 5 | 1.5 | 2 | Pos, area saved |

#### Vid 2: 	
GreenFront	Shirt	2, 10, 7, 30, 4, 1.5, 12		Pos, area saved

GreenBack	Shirt	2, 10, 7, 30, 4, 1.5, 12		Pos, area saved

WhiteLeft	+Shirt ([62,394]) 1, 10, 4, 7, 5, 1.0, 5	Pos, area saved

WhiteRight	Shorts ([75,485]) 1, 10, 5, 8, 5, 1.5, 2	Pos, area saved

#### Vid 3: 	
GreenDown	*Shirt(Fr36,[225,8])   2, 4, 5, 10, 4, 1.6, 7	Pos, area saved
GreenUp		*+Shirt(Fr36,[189,26]) 2, 4, 6, 20, 5, 1.6, 7	Pos, area saved
WhiteLeft	Shorts	2, 30, 6, 30, 4, 1.5, 4			Pos, area saved
WhiteRight	Shorts	2, 10, 7, 30, 4, 1.5, 3			Pos, area saved

#### Vid 4: 	
GreenDown	*Shirt(Fr72)  1, 10, 5 20, 6, 1.6, 7		Pos, area saved
GreenUp		*Shirt(Fr135) 1, 10, 5 10, 4, 1.5, 7		Pos, area save
WhiteLeft	X Shorts  3, 10, 7, 10, 6, 1.5, 2		Pos, area saved
WhiteRight	X+ Shorts 1, 10, 7, 15, 4, 1.5, 2		Pos, area saved

#### Vid 5: 	
RedFront	+Shirt(Fr215) 2, 5, 8, 15, 5, 2.0, 10		Pos, area saved
RedBack		Shirt	      2, 5, 8, 15, 5, 2.0, 10		Pos, area saved
WhiteLeft	*
WhiteRight	*+

#### Vid 6: 	
RedUp		*+Shirt(Fr65) 1, 10, 8, 10, 5, 1.5, 8		Pos, area saved
RedDown		*Shirt(Fr65)  1, 10, 8, 10, 5, 1.5, 8		Pos, area saved(used bottemost instead of centre)
WhiteLeft	XShoulder 2, 7, 7, 25, 5, 1.5, 2		Pos, area saved
WhiteRight	Shoulder  2, 7, 7, 25, 5, 1.5, 2		Pos, area saved

#### Vid 7: 	
RedLeft		Shirt 1, 8, 8, 15, 4, 1.5, 4			Pos, area saved
RedRight	XShirt 1, 8, 8, 15, 4, 1.5, 4 			Pos, area saved(used bottemost instead of centre)
WhiteUp		*Shoulder(Fr106) 1, 8, 6, 15, 5, 1.5, 10	Pos, area saved
WhiteDown	*Shoulder(Fr106) 1, 4, 6, 8, 5, 1.5, 10		Pos, area saved(used bottemost instead of centre)











