'''
利用dlib检测人脸并定位五官的坐标
'''
import dlib
import glob
import csv
from skimage import io
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
# we can get this file  from dlib official webpage
predictor = dlib.shape_predictor('E:\\DeepLearning\\data\\shape_predictor_68_face_landmarks.dat')
num_landmarks = 68

'''
68个点对应的五官：
IdxRange jaw; // [0 , 16]
IdxRange rightBrow; // [17, 21]
IdxRange leftBrow; // [22, 26]
IdxRange nose; //[27, 35]
IdxRange rightEye; // [36, 41]
IdxRange leftEye; // [42, 47]
IdxRange mouth;// [48, 59]
IdxRange mouth2; // [60, 67] }
'''


for f in glob.glob('./data/*.jpg'):
    img = io.imread(f) #type:np.ndarray
    img2 = copy.copy(img)#type:np.ndarray
    dets = detector(img, 1)  # face detection
    print("detect %d faces in %s"%(len(dets), f))

    for i in range(len(dets)):

        d = dets[i]
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d) #type:dlib.full_object_detection
        print("get %d landmarks"%(shape.num_parts))

        for i in range(num_landmarks):
            part_i_x = shape.part(i).x
            part_i_y = shape.part(i).y
            if part_i_x < 0 or part_i_x >= img2.shape[1] or part_i_y < 0 or part_i_y >= img2.shape[0]:
                print("invalid part(%d,%d)"%(part_i_x, part_i_y))
                continue
            img2[part_i_y, part_i_x, 0] = 255
            img2[part_i_y, part_i_x, 1] = 255
            img2[part_i_y, part_i_x, 2] = 255
    io.imshow(img2)
    plt.show()

