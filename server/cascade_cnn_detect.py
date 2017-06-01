import numpy as np
import cv2
import time
import random
from operator import itemgetter
from load_model_functions import *
from face_detection_functions import *


from import_caffe import caffe
# ==================  load models  ======================================
net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal = \
load_face_models()

nets = (net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal)


def crop_faces(img):
    min_face_size = 48
    stride = 5
    img_forward = np.array(img, dtype=np.float32)
    img_forward -= np.array((104, 117, 123))
    rectangles = detect_faces_net(nets, img_forward, min_face_size, stride, True, 1.414, 0.82)
    return rectangles

