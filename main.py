import cv2
import numpy as np


#for image read
img=cv2.imread('images/02.jpg',cv2.IMREAD_COLOR)

#details of Imge
print("Shape: ",img.shape)
print("\nSize: ",img.size)
print("\nDType: ",img.dtype)


#==============Manual edge ditect=====================
def segment_img(_img,limit):
	for i in range(0,_img.shape[0]-1):
		for j in range(0,_img.shape[1]-1): 
			if int(_img[i,j+1])-int(_img[i,j])>=limit:
				_img[i,j]=0
			elif(int(_img[i,j-1])-int(_img[i,j])>=limit):
				_img[i,j]=0
	
	return _img
#======================================================

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#for i in range(0,gray.shape[0]):
#	for j in range(0,gray.shape[1]): 
#		if (int(gray[i,j]))<=100:
#			gray[i,j]=100

#gray=segment_img(gray,15)
#cv2.imshow("GrayEdited",gray)
median = cv2.medianBlur(gray,5)
bool,threshold_img=cv2.threshold(median,130,255,cv2.THRESH_BINARY)
#blur=cv2.GaussianBlur(threshold_img,(7,7),0)
cv2.imshow("threshold",threshold_img)

initial=[]
final=[]
line=[]
for i in range(0,gray.shape[0]):
	tmp_initial=[]
	tmp_final=[]
	for j in range(0,gray.shape[1]-1):
		if threshold_img[i,j]==0 and (threshold_img[i,j+1])==255:
			tmp_initial.append((i,j))
			#img[i,j]=[255,0,0]
		if threshold_img[i,j]==255 and (threshold_img[i,j+1])==0:
			tmp_final.append((i,j))
			#img[i,j]=[255,0,0]
	
	x= [each for each in zip(tmp_initial,tmp_final)]
	x.sort(key= lambda each: each[1][1]-each[0][1])
	line.append(x[len(x)-1])


err= 25
danger_points=[]

for i in range(1,len(line)-1):
	try:
		prev_= line[i-1]
		next_= line[i+1]

		dist_prev= prev_[1][1]-prev_[0][1]
		dist_next= next_[1][1]-next_[0][1]
		if abs(dist_next-dist_prev)>err:
			print("Dist: {}".format(abs(dist_next-dist_prev)))
			print(line[i])
			danger_points.append(line[i])
	except:
		pass

	#print(each)
	start,end= line[i]
	#raise ZeroDivisionError
	mid=int((start[0]+end[0])/2),int((start[1]+end[1])/2)
	img[mid[0],mid[1]]=[0,0,255]

try:
	start_rect=danger_points[0][0][::-1]
	start_rect=(start_rect[0]-40, start_rect[1]-30)
	end_rect= danger_points[len(danger_points)-2][1][::-1]
	end_rect= (end_rect[0]+40, end_rect[1])
	cv2.rectangle(img,start_rect,end_rect,(0,255,0),2)
except:
	print("No points found")

#blur= cv2.GaussianBlur(img,(5,5),0)

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2)= plt.subplots(2,1)

x= np.arange(1,gray.shape[0]-1)
y= dist_list

#print(len(x),len(y))


ax1.plot(x,y)
img= np.rot90(img)
ax2.imshow(img)

plt.show()



#wait for key pressing
cv2.waitKey(0)

#Distroy all the cv windows
cv2.destroyAllWindows()
