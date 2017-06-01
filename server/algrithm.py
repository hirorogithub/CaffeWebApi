# -*- coding: utf-8 -*-
import cascade_cnn_detect
import cv2
import age_estimation
import gender_estimation
import random
import time
import os.path
import json


def all_japi_path(path):

    data,faces = cascade_api_path(path,True)
    tmp_age = []
    tmp_gender = []
    for face in faces :
        tmp_age.append(age_api_face(face))
        tmp_gender.append(gender_api_face(face))
    result = {
    "points":data,
    "ages":tmp_age,
    "gender":tmp_gender
    }
    return json.dumps(result,indent=4)
     

#api version:
#api that return json
def cascade_japi_face(face,path,for_age = False):
    data,faces = cascade_api_face(face,path,for_age)
    tmp = {
    "points":data
    }
    print(json.dumps(tmp))
    return json.dumps(tmp)

def cascade_japi_path(path,for_age = False):
    return cascade_japi_face(cv2.imread(path),path,for_age)

#api that return dict and faces
def cascade_api_face(face,path,for_age = False):
    #return:(x1,y1),(x2,y2),confidence

    rectangles = cascade_cnn_detect.crop_faces(face)
    if len(rectangles) == 0:
        print("no people")
        return "",""

    save_crop_img(rectangles,face,path)   #just for server output ,can be commented

    height = face.shape[0]
    width = face.shape[1]
    faces_for_age = []
    if for_age:                      #for age estimation ,it should return point(x,y)X2
        for rectangle in rectangles: #TODO：merge with the following for loop
            x1,y1,x2,y2 = shape_rectangle(rectangle,height,width,0.2)
            faces_for_age.append(face[y1:y2, x1:x2])

    data = []
    for rectangle in rectangles:
        x1,y1,x2,y2 = shape_rectangle(rectangle,height,width,0.2)
        tmp = {
         "x1":x1,
         "y1":y1,
         "x2":x2,
         "y2":y2,
         "confidence":str(rectangle[4])     
        }
        data.append(tmp)
    return data,faces_for_age


def cascade_api_path(path,for_age = False):
    return cascade_api_face(cv2.imread(path),path,for_age)


#age api
#api that return json
def age_japi_face(face):
    return json.dumps(age_api_face(face),indent=4)

def age_japi_path(path):
    return json.dumps(age_api_path(path),indent=4)

#apt that return dict
def age_api_face(face):
    return age_estimation.judge_ages_by_face(face)

def age_api_path(path):
    return age_estimation.judge_ages_by_path(path)


#gender api
#api that return json
def gender_japi_face(face):
    return json.dumps(gender_api_face(face),indent=4)

def gender_japi_path(path):
    return json.dumps(gender_api_path(path),indent=4)

#apt that return dict
def gender_api_face(face):
    return gender_estimation.judge_gender_by_face(face)

def gender_api_path(path):
    return gender_estimation.judge_gender_by_path(path)





#some tools function:
def get_img_size(path):
    img = cv2.imread(path)
    width = img.shape[1]
    height = img.shape[0]
    return width,height



def save_crop_img(rectangles,face,path):
    output = []
    shape = face.shape
    height = shape[0]
    width = shape[1]
    i = 1       #because 0 for point`s json ,so img star from 1
    for rectangle in rectangles: 
        x1,y1,x2,y2 = shape_rectangle(rectangle,height,width,0.2)
        cv2.rectangle(face, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 0, 0), 5)
        cv2.putText(face, str(rectangle[4]), (rectangle[0], rectangle[1]), 0, 0.5, (0, 255, 0), 2)
        output_tmp = face[y1:y2, x1:x2]
        output.append(output_tmp)
        cv2.imwrite("../imgout0/"+str(i)+"--"+ os.path.basename(path), output_tmp)
        i+=1





def shape_rectangle(rectangle,height,width,pad):
    #enlarge the rectangle by pad(float)persent
    #return x1,y1,x2,y2

    x_offset = int((rectangle[2]-rectangle[0])*pad/2)
    y_offset = int((rectangle[3]-rectangle[1])*pad/2)


    if rectangle[1] - y_offset < 0:
        y1 = 0
    else:
        y1 = rectangle[1] - y_offset

    if rectangle[0] - x_offset < 0:
        x1 = 0
    else:
        x1 = rectangle[0] - x_offset

    if rectangle[3] + y_offset > height:
        y2 = height
    else:
        y2 = rectangle[3] + y_offset

    if rectangle[2] + x_offset > width:
        x2 = width
    else:
        x2 = rectangle[2] + x_offset    

    return x1,y1,x2,y2


