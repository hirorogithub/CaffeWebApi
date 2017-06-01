import numpy as np
import sys

def get_model_dir():
    return "../model/CacadeCNNModels/"

def load_face_models():
    '''
    Loads face detection models
    :return: all 6 models as a tuple
    '''


    #TODO:load any model in a function
    # ==================  caffe  ======================================
    caffe_root = "/home/curi/hiro/caffe/"  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + "python")
    import caffe


    # ==================  load face12c_full_conv  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"12/face12c_full_conv.prototxt"
    PRETRAINED = get_model_dir()+"12/face12c_full_conv.caffemodel"
    caffe.set_mode_gpu()
    net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)


    # ==================  load face_12_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"12cal/deploy.prototxt"
    PRETRAINED = get_model_dir()+"12cal/face_12_cal_train_iter_400000.caffemodel"
    caffe.set_mode_gpu()
    net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
   


    # ==================  load face_24c  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"24/deploy.prototxt"
    PRETRAINED = get_model_dir()+"24/face_24c_train_iter_400000.caffemodel"
    caffe.set_mode_gpu()
    net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)


    
    # ==================  load face_24_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"24cal/deploy.prototxt"
    PRETRAINED = get_model_dir()+"24cal/face_24_cal_train_iter_400000.caffemodel"
    caffe.set_mode_gpu()
    net_24_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    


    # ==================  load face_48c  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"48/deploy.prototxt"
    PRETRAINED = get_model_dir()+"48/face_48c_train_iter_200000.caffemodel"
    caffe.set_mode_gpu()
    net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)


   
    # ==================  load face_48_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = get_model_dir()+"48cal/deploy.prototxt"
    PRETRAINED = get_model_dir()+"48cal/face_48_cal_train_iter_300000.caffemodel"
    caffe.set_mode_gpu()
    net_48_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    

    return net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal

