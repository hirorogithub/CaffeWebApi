# -*- coding: utf-8 -*-


from import_caffe import caffe

import numpy as np
import os
import cv2
import json

gender_list=["male","female"]

#load model globaly

model =  "../model/cnn_age_gender_models_and_data.0.0.2/gender_net.caffemodel"
net_def = "../model/cnn_age_gender_models_and_data.0.0.2/deploy_gender.prototxt" 
caffe.set_mode_gpu()

    #load mean
    #mean_filename='../model/cnn_age_gender_models_and_data.0.0.2/mean.binaryproto'
    #proto_data = open(mean_filename, "rb").read()
    #a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    #mean  = caffe.io.blobproto_to_array(a)[0]


gender_net_pretrained=model
gender_net_model_file=net_def
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,\
                       #mean=mean,\#mean`s dims is not equal 227*227
                       channel_swap=(2,1,0),\
                       raw_scale=255,\
                       image_dims=(227, 227))



#original version(ruozou`s) to use caffe model 
def judge_gender_by_face_bak(face):

    model =  "../model/cnn_age_gender_models_and_data.0.0.2/gender_net.caffemodel"
    net_def = "../model/cnn_age_gender_models_and_data.0.0.2/deploy_gender.prototxt" 
    caffe.set_mode_gpu()
    net = caffe.Net(net_def, model, caffe.TEST)
    data = []
    img = face
    X = np.empty((1, 3, 227, 227))
    img = cv2.resize(img, (227, 227))
    image = np.array(img, dtype=np.float32)
    X[0, 0, :, :] = image[:, :, 0]
    X[0, 1, :, :] = image[:, :, 1]
    X[0, 2, :, :] = image[:, :, 2]
    net.blobs['data'].data[...] = X
    net.forward()

    prob = net.blobs['prob'].data[0]
    index = 1 if prob[1]>prob[0] else 0
    print("gender is ", gender_list[index], "confidence is ", prob[index])
    print("gender is not", gender_list[~index], "confidence is ", prob[~index])
    data = {
        "index":index,
        "gender":gender_list[index],
        "confidence":str(prob[index])
    }
    return data

#age_and_gender classification demo version to use caffe model 
def judge_gender_by_face(face):

  
     

    prediction = gender_net.predict([face]) 

    print 'predicted gender:', gender_list[prediction[0].argmax()]
    print ("confidence is :",prediction[0][prediction[0].argmax()])
    index = prediction[0].argmax()
    data = {
        "index":index,
        "gender":gender_list[index],
        "confidence":str(prediction[0][index])
    }
    return data
   

    


def judge_gender_by_path(path):
    return judge_gender_by_face(cv2.imread(path))



