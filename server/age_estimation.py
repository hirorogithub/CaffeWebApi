# -*- coding: utf-8 -*-


from import_caffe import caffe

import numpy as np
import os
import cv2
import json

age_list=["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]


#load model globaly

#TODO: model and other file should be loaded once {
model =  "../model/cnn_age_gender_models_and_data.0.0.2/age_net.caffemodel"
net_def = "../model/cnn_age_gender_models_and_data.0.0.2/deploy_age.prototxt" 
caffe.set_mode_gpu()
net = caffe.Net(net_def, model, caffe.TEST) # TODO: make net singleton
#}TODO:write a singleton class to save these configurations (model paths, cpu/gpu mode, label)




def take_max(a):
    tempmax = 0
    for i in range(len(a)):
        if a[i] > a[tempmax]:
            tempmax = i
    return tempmax

#original version(ruozou`s) to use caffe model 
def judge_ages_by_face(face):


    #load mean
    #mean_npy=np.load('/home/curi/hiro/gitlab/flaskTest/model/cnn_age_gender_models_and_data.0.0.2/mean.npy')
    #mean  = mean_npy

    data = []
    img = face
    X = np.empty((1, 3, 227, 227))
    img = cv2.resize(img, (227, 227))
    image = np.array(img, dtype=np.float32)
    X[0, 0, :, :] = image[:, :, 0]
    X[0, 1, :, :] = image[:, :, 1]
    X[0, 2, :, :] = image[:, :, 2]
    #X[0,:,:,:] -= mean
    net.blobs['data'].data[...] = X
    net.forward()

    prob = net.blobs['prob'].data[0]
    index = take_max(prob)
    print("age is ", age_list[index], "rate is ", prob[index])
    data = {
        "index":index,
        "age":age_list[index],
        "confidence":str(prob[index])
    }
    return data

#age_and_gender classification demo version to use caffe model 
def judge_age_by_face_bak(face):

    model =  "../model/cnn_age_gender_models_and_data.0.0.2/age_net.caffemodel"
    net_def = "../model/cnn_age_gender_models_and_data.0.0.2/deploy_age.prototxt" 
    caffe.set_mode_gpu()

    #load mean
    #mean_filename='../model/cnn_age_gender_models_and_data.0.0.2/mean.binaryproto'
    #proto_data = open(mean_filename, "rb").read()
    #a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    #mean  = caffe.io.blobproto_to_array(a)[0]


    age_net_pretrained=model
    age_net_model_file=net_def
    age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,\
                       #mean=mean,\#mean`s dims is not equal 227*227
                       channel_swap=(2,1,0),\
                       raw_scale=255,\
                       image_dims=(227, 227))
    

    prediction = age_net.predict([face]) 

    print 'predicted age:', age_list[prediction[0].argmax()]
    print ("confidence is :",prediction[0][prediction[0].argmax()])
    index = prediction[0].argmax()
    data = {
        "index":index,
        "age":age_list[index],
        "confidence":str(prediction[0][index])
    }
    return data
   

def judge_ages_by_path(path):
    return judge_ages_by_face(cv2.imread(path))



