# -*- coding: utf-8 -*-

from import_caffe import caffe

import numpy as np

MEAN_PROTO_PATH = '/home/curi/hiro/gitlab/flaskTest/model/cnn_age_gender_models_and_data.0.0.2/mean.binaryproto'               # 待转换的pb格式图像均值文件路径
MEAN_NPY_PATH = '/home/curi/hiro/gitlab/flaskTest/model/cnn_age_gender_models_and_data.0.0.2/mean.npy'                         # 转换后的numpy格式图像均值文件路径

blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
blob.ParseFromString(data)                         # 解析文件内容到blob

array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）

mean_npy = array[0]    
mean_npy = np.resize(mean_npy,(3,227,227))                            # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
np.save(MEAN_NPY_PATH ,mean_npy)
print("mean_py is ",mean_npy.shape)
