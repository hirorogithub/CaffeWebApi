# -*- coding: utf-8 -*-


from import_caffe import caffe
import numpy as np
import os
import cv2
import json

gender_list=["male","female"]

def judge_gender_by_face(face):

    model =  "../model/cnn_age_gender_models_and_data.0.0.2/gender_net.caffemodel"
    net_def = "../model/cnn_age_gender_models_and_data.0.0.2/deploy_gender.prototxt" 
    caffe.set_mode_gpu()


    mean_filename='../model/cnn_age_gender_models_and_data.0.0.2/mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]

    gender_net_pretrained=model
    gender_net_model_file=net_def
    gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,\
                       #mean=mean,\
                       channel_swap=(2,1,0),\
                       raw_scale=255,\
                       image_dims=(227, 227))

     

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


judge_gender_by_path("/home/curi/Desktop/4e1aaf42ada17.jpg")

