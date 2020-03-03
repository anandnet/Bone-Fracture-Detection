from PIL import Image
import cv2
import numpy as np
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle

WIDTH = 310/2
HEIGHT = 568/2

size = (int(WIDTH), int(HEIGHT))

def resize_and_save(img_name):
    #try:
    main_image = Image.open("images/Fractured Bone/{}".format(img_name))
    #except IOError:
    #sys.stderr.write("ERROR: Could not open file {}\n".format(img_name))
    #return
    #exit(1)
        
    #main_image.thumbnail(size, Image.ANTIALIAS)
    x= main_image.resize(size, Image.NEAREST)
    x.save("images/resized/{}".format(img_name))

def _reshape_img(arr):
    #reshape a numpy image and returns a 1-D array
    flat_arr=[]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                flat_arr.append(arr[i][j][k])

    return flat_arr

def _create_data(train_img_list, label_list):
    """
    param train_img_list: A list containg images to train model upon
    param label_list: A list containing labels corresponding to images
                      in the train_img_list
    
    returns inp_arr: a ready-to-feed numpy array for training
    returns np.array(label_list): a numpy array of corresponding label_list
    """
    inp_arr=[]

    for img in train_img_list:
        img= cv2.imread(img)
        inp_arr.append(_reshape_img(img))
    
    inp_arr= np.array(inp_arr)

    return inp_arr,np.array(label_list)


def train_and_save(train_img_list, label_list, model_name):
    try:
        with open(model_name,"rb") as file:
            model= pickle.load(file)
    except FileNotFoundError:
        in_arr, out_arr= _create_data(train_img_list, label_list)
        
        #model= LogisticRegression(random_state=43,tol=1e-5,max_iter=5000,verbose=1).fit(in_arr,out_arr)
        #model = RandomForestClassifier(n_estimators=10, random_state=42).fit(in_arr, out_arr)
        model= Ridge(alpha=0.01,tol=0.000001,max_iter=5000,random_state=43).fit(in_arr,out_arr)

        with open(model_name,"wb") as file:
            pickle.dump(model,file)
    
    return model

def get_model(model_name):
    try:
        with open(model_name,"rb") as file:
            model= pickle.load(file)
            return model
    except FileNotFoundError:
        print("{} doesn't exist. Train and save a model first".format(model_name))
        sys.exit(0)

if __name__=="__main__":
    for each in range(1,101):
        try:
            resize_and_save("F{}.JPG".format(each))
        except IOError:
            try:
                resize_and_save("F{}.jpg".format(each))
            except IOError:
                resize_and_save("F{}.jpeg".format(each))


    from train_label import train_label, test_label

    train_img_list=[]
    train_label_list=[]
    
    for key in train_label.keys():
        train_img_list.append("images/resized/"+key+".jpg")
        train_label_list.append(train_label[key])
    
    test_img_list=[]
    test_label_list=[]
    for key in test_label.keys():
        test_img_list.append("images/resized/"+key+".jpg")
        test_label_list.append(test_label[key])
    
    #in_arr, out_arr= _create_data(train_img_list,label_list)
    #print(in_arr.shape)
    print("Training started...")
    svm_model=train_and_save(train_img_list,train_label_list, "ridge_model")
    print("Training finished...")

    train_in_arr, train_out_arr= _create_data(train_img_list,train_label_list)
    test_in_arr, test_out_arr= _create_data(test_img_list,test_label_list)

    print("Training set score: {:.2f}".format(svm_model.score(train_in_arr, train_out_arr)))
    print("Test set score: {:.2f}".format(svm_model.score(test_in_arr, test_out_arr)))