import cv2
import os
import random
import numpy as np
from GenerateList import  make_list,read_list
from GenerateAugImage import pro_ImageWithoutResize
import multiprocessing
from anti.mtcnn_detector import MtcnnDetector
import mxnet as mx
import random
import sys

# loading detector module
def patchSelect(img,boxes,points,label,ImgName,save_dir2):
    draw = img.copy()
    for b in boxes:
        pass
    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
    crop = draw[int(b[1]):int(b[3]), int(b[0]):int(b[2])].copy()
    height = crop.shape[0]
    width = crop.shape[1]
    height_part = int(height/96)
    width_part = int(width/96)
    ##get face key point
    if label == 1:
        prefix_lable = "spoof"
        for p in points:
            for i in range(5):
                x_start = int(p[i]-48)
                y_start = int(p[i+5]-48)
                x_end = int(p[i]+48)
                y_end = int(p[i+5]+48)
                cv2.rectangle(draw, (x_start, y_start), (x_end, y_end), (255, 255, 255))
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
                cropKeypoint = img[y_start:y_end, x_start:x_end]
                cv2.imwrite(save_dir2 +"/"+prefix_lable+"/"+  "keypoint_" +str(int(random.random()*100))+"_"+ ImgName, cropKeypoint)
        # for b in boxes:
        #     x_start = int(b[0])
        #     y_start = int(b[1])
        #     x_end = int(b[2])
        #     y_end = int(b[3])
        #     cv2.rectangle(draw, (x_start, y_start), (x_start+96, y_start+96), (0, 255, 255))
        #     cv2.rectangle(draw, (x_end-96, y_start), (x_end, y_start+96), (0, 255, 255))
        #     cv2.rectangle(draw, (x_start, y_end-96), (x_start+96, y_end), (0, 255, 255))
        #     cv2.rectangle(draw, (x_end-96, y_end-96), (x_end, y_end), (0, 255, 255))
        #     crop1 = img[y_start:y_start+96, x_start:x_start+96]
        #     crop2 = img[y_start:y_start+96, x_end-96:x_end]
        #     crop3 = img[y_end-96:y_end, x_start:x_start+96]
        #     crop4 = img[y_end-96:y_end,x_end-96:x_end]
        #     # print save_dir2 +"/"+prefix_lable+"/"+ str(i) + "_step_" +str(int(random.random()*100))
        #     cv2.imwrite(save_dir2 +"/"+prefix_lable+"/"+ str(i) + "_step_" +str(int(random.random()*100)) + ImgName, crop1)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(i) + "_step_" + str(int(random.random() * 100)) + ImgName,crop2)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(i) + "_step_" + str(int(random.random() * 100)) + ImgName,crop3)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(i) + "_step_" + str(int(random.random() * 100)) + ImgName,crop4)
    if label==0:
        prefix_lable = "alive"
        for p in points:
            for i in range(5):
                x_start = int(p[i]-48)
                y_start = int(p[i+5]-48)
                x_end = int(p[i]+48)
                y_end = int(p[i+5]+48)
                cv2.rectangle(draw, (x_start, y_start), (x_end, y_end), (255, 255, 255))
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
                cropKeypoint = img[y_start:y_end, x_start:x_end]
                cv2.imwrite(save_dir2 +"/"+prefix_lable+"/"+  "keypoint_" +str(int(random.random()*100)) +"_"+ ImgName, cropKeypoint)
        ## get edge of rect
        # count = 0
        # for b in boxes:
        #     x_start = int(b[0])
        #     y_start = int(b[1])
        #     x_end = int(b[2])
        #     y_end = int(b[3])
        #     cv2.rectangle(draw, (x_start, y_start), (x_start+96, y_start+96), (0, 255, 255))
        #     cv2.rectangle(draw, (x_end-96, y_start), (x_end, y_start+96), (0, 255, 255))
        #     cv2.rectangle(draw, (x_start, y_end-96), (x_start+96, y_end), (0, 255, 255))
        #     cv2.rectangle(draw, (x_end-96, y_end-96), (x_end, y_end), (0, 255, 255))
        #     crop1 = img[y_start:y_start+96, x_start:x_start+96]
        #     crop2 = img[y_start:y_start+96, x_end-96:x_end]
        #     crop3 = img[y_end-96:y_end, x_start:x_start+96]
        #     crop4 = img[y_end-96:y_end,x_end-96:x_end]
        #     # print save_dir2 +"/"+prefix_lable+"/"+ str(i) + "_step_" +str(int(random.random()*100))
        #     cv2.imwrite(save_dir2 +"/"+prefix_lable+"/"+ str(count) + "_step_" +str(int(random.random()*100)) + ImgName, crop1)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(count) + "_step_" + str(int(random.random() * 100)) + ImgName,crop2)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(count) + "_step_" + str(int(random.random() * 100)) + ImgName,crop3)
        #     cv2.imwrite(save_dir2 + "/" + prefix_lable + "/" + str(count) + "_step_" + str(int(random.random() * 100)) + ImgName,crop4)
        #     count +=1
        # width_step = (width-96)/7
        # height_step = (height-96)/7
        # for i in range(7):
        #     for j in range(7):
        #         x_start = int(b[0])+ width_step*i
        #         y_start = int(b[1])+height_step*j
        #         # x_end = int(b[2])+width_step*i
        #         # y_end = int(b[3])+height_step*j
        #         crop = img[y_start:y_start+96,x_start:x_start+96]
        #         # print save_dir2+"/"+prefix_lable+str(i)+"_step_"+str(j)+ImgName
        #         cv2.imwrite(save_dir2+"/"+prefix_lable+"/"+str(i)+"_step_"+str(j)+ImgName,crop)
    # cv2.imshow("draw",draw)
    # cv2.waitKey(0)
    startPoint = 0
    print "parts",height_part,width_part
    # cv2.imshow("detection result", crop)
    # cv2.waitKey(0)
    # pass

def get_bbox(face_shape, bound):
    height, width = bound
    max_x, max_y = np.max(face_shape, axis=0)
    min_x, min_y = np.min(face_shape, axis=0)
    if 0 <= min_x < max_x < width and 0 <= min_y < max_y < height:
        return min_x, min_y, max_x, max_y
    else:
        return None

def get_newbox(facebox,bound):
    b = facebox
    height = bound[0]
    width = bound[1]
    new_x1 = int(b[0] - (b[2] - b[0]) * 0.2)
    new_y1 = int(b[1] - (b[3] - b[1]) * 0.6)
    new_x2 = int(b[2] + (b[2] - b[0]) * 0.2)
    new_y2 = int(b[3] + (b[3] - b[1]) * 0.15)
    if new_x1 < 0:
        new_x1 = 0
    if new_y1 < 0:
        new_y1 = 0
    if new_x2 > width:  # shape[1]===width shape[0]==height
        new_x2 = width
    if new_y2 > height:
        new_y2 =height
    return new_x1,new_y1,new_x2,new_y2

if __name__=="__main__":
    multiprocessing.freeze_support()
    detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(0), num_worker=4, accurate_landmark=False)
    print "==========step1 : process make list ========="
    workingdir= "/home/sbb/data/antiSpoof/MSU/original/"
    save_dir = "/home/sbb/data/antiSpoof/MSU_AUG/"
    recursive = True
    listprefix = "msu"
    train_prefix = "train"
    ListGenerateType = 0
    #0: for given working dir generate train or val list
    #1: for given working dir generate train and val and test list
    # make_list(workingdir,recursive=True,shuffle=True,train_ratio=1,prefix="msu",usePrefix=False)

    listName = listprefix+".lst"
    if os.path.exists(listName):
        print "==========Step1 : List Generate Completely ========="
    else:
        print "ImageList is not Generate"
    print "==========Step2 : Augument All Image for Training ========="
    list_r = read_list(listName)
    list_r = list(list_r)
    shuffle = True
    if shuffle is True:
        random.shuffle(list_r)
    N = len(list_r)
    train_ratio = 0.9
    train_num = int(N * train_ratio)
    aug_Num = 64
    className = ["alive","spoof"]
    for i,item in enumerate(list_r):
        if i ==train_num:
            train_prefix= "val"
        save_dir2 =save_dir+ "msu_"+train_prefix+"/"
        filePath = item[2]
        # print item,"****"
        print i/float(N)
        label = int(item[1])
        print workingdir,filePath
        # print filePath
        img = cv2.imread(workingdir+filePath)
        # img = cv2.imread("/home/sbb/data/gender/CASIA-WebFace/CASIA-WebFace/2450852/002.jpg")
        imgName = filePath.replace("/","_")
        # imgName = imgName.split("/")[-1]
        det_result= detector.detect_face(img)
        if det_result is not None:
            boxes, points = det_result
            try:
                patchSelect(img,boxes,points,label,imgName,save_dir2)
            except ValueError:  # raised if `y` is empty.
                print imgName
    #         chips = detector.extract_image_chips(img, points, 224, 0.37)
    #         for i, chip in enumerate(chips):
    #             for aug in xrange(aug_Num):
    #                 chip_aug = pro_ImageWithoutResize(chip, boxes[0][:4])
    #                 save_dir3 = save_dir2+className[label]+"/"+"web"+str(aug)+"_"+imgName
    #                 if chip_aug is not None:
    #                     cv2.imwrite(save_dir3,chip_aug)
    #
    # print "==========step2 : Augument Image Completely ========="
    # print "==========step3 : Generate list for Training and Val  ========="
    Training_dir  = save_dir+"msu_train/"
    Val_dir = save_dir+"msu_val/"
    make_list(Training_dir, recursive=True, shuffle=True, train_ratio=1, prefix="msu_train",usePrefix = True)
    make_list(Val_dir, recursive=True, shuffle=True, train_ratio=1, prefix="msu_val",usePrefix = True)
