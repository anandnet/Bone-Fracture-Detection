import cv2
import numpy as np

from pre_process import _reshape_img, get_model

img_name="new"
"""
Currently, `img_name` will be used to get resized image from `images/resized` folder
and original image from `images/Fractured Bone` so it expects the same named image file
to be available in both the folders.

All names starting with F{n} are available in both the folders. 1<= n <=100
"""

model_name= "ridge_model"



img_file= 'images/resized/{}'.format(img_name)
orig_img= 'images/Fractured Bone/{}'.format(img_name)

#for image read
try:
	img_t=cv2.imread(img_file+".jpg",cv2.IMREAD_COLOR)
	img=cv2.imread(orig_img+".jpg",cv2.IMREAD_COLOR)
	shape= img.shape
except (AttributeError,FileNotFoundError):
	try:
		img_t=cv2.imread(img_file+".JPG",cv2.IMREAD_COLOR)
		img=cv2.imread(orig_img+".JPG",cv2.IMREAD_COLOR)
		shape=img.shape
	except (AttributeError,FileNotFoundError):
		img_t=cv2.imread(img_file+".png",cv2.IMREAD_COLOR)
		img=cv2.imread(orig_img+".png",cv2.IMREAD_COLOR)
		shape=img.shape

	#else: raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))
#else:
#	raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))


#details of Imge
print("\nShape: ",shape)
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
cv2.imshow("GrayEdited",gray)
median = cv2.medianBlur(gray,5)

model= get_model(model_name)
pred_thresh= model.predict([_reshape_img(img_t)])
bool,threshold_img=cv2.threshold(median,pred_thresh,255,cv2.THRESH_BINARY)
#blur=cv2.GaussianBlur(threshold_img,(7,7),0)
cv2.imshow("threshold",threshold_img)


initial=[]
final=[]
line=[]
#count=[]
#for i in range(0,256):
#	count.append(0)

for i in range(0,gray.shape[0]):
	tmp_initial=[]
	tmp_final=[]
	for j in range(0,gray.shape[1]-1):
		#count[gray[i,j]]+=1
		if threshold_img[i,j]==0 and (threshold_img[i,j+1])==255:
			tmp_initial.append((i,j))
			#img[i,j]=[255,0,0]
		if threshold_img[i,j]==255 and (threshold_img[i,j+1])==0:
			tmp_final.append((i,j))
			#img[i,j]=[255,0,0]
	
	x= [each for each in zip(tmp_initial,tmp_final)]
	x.sort(key= lambda each: each[1][1]-each[0][1])
	try:
		line.append(x[len(x)-1])
	except IndexError: pass

#print(count)


err= 15
danger_points=[]

#store distances
dist_list=[]

for i in range(1,len(line)-1):
	dist_list.append(line[i][1][1]-line[i][0][1])
	try:
		prev_= line[i-3]
		next_= line[i+3]

		dist_prev= prev_[1][1]-prev_[0][1]
		dist_next= next_[1][1]-next_[0][1]
		diff= abs(dist_next-dist_prev)
		if diff>err:
			#print("Dist: {}".format(abs(dist_next-dist_prev)))
			#print(line[i])
			data=(diff, line[i])
			#print(data)
			if len(danger_points):
				prev_data=danger_points[len(danger_points)-1]
				#print(prev_data)
				#print("here1....")
				if abs(prev_data[0]-data[0])>2 or data[1][0]-prev_data[1][0]!=1:
					#print("here2....")
					print(data)
					danger_points.append(data)
			else:
				print(data)
				danger_points.append(data)
	except Exception as e:
		print(e)
		pass

	#print(each)
	start,end= line[i]
	#raise ZeroDivisionError
	mid=int((start[0]+end[0])/2),int((start[1]+end[1])/2)
	#img[mid[0],mid[1]]=[0,0,255]

for i in range(0,len(danger_points)-1,2):
	try:
		start_rect=danger_points[i][1][0][::-1]
		start_rect=(start_rect[0]-40, start_rect[1]-40)
  
		end_rect= danger_points[i+1][1][1][::-1]
		end_rect= (end_rect[0]+40, end_rect[1]+40)
  
		cv2.rectangle(img,start_rect,end_rect,(0,255,0),2)
	except:
		print("Pair not found")

#blur= cv2.GaussianBlur(img,(5,5),0)

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2)= plt.subplots(2,1)

fig2, ax3= plt.subplots(1,1)

x= np.arange(1,gray.shape[0]-1)
y= dist_list

#print(len(x),len(y))

cv2.calcHist(gray,[0],None,[256],[0,256])

try:
	ax1.plot(x,y)
except:
	print("Could not plot")
img= np.rot90(img)
ax2.imshow(img)

#count= range(256)
#ax3.hist(count, 255, weights=count, range=[0,256])
ax3.hist(gray.ravel(),256,[0,256])

plt.show()



#wait for key pressing
cv2.waitKey(0)

#Distroy all the cv windows
cv2.destroyAllWindows()
