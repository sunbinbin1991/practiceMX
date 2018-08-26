import numpy as np
import cv2
import numpy.matlib
import random

ratio_coloraug = 0.7
ratio_noise = 0.5
ratio_move = 0.3
ratio_gray = 0.3
ratio_scale_o = 1

ratio_mirror = 0.6
Trans_r = 0
Rotate_r = 0
# [0.5, 0.5, 0.5, 0.4]

def pro_Image(img,bbox):
    x1, y1, x2, y2 = bbox
    width = x2-x1
    height = y2-y1
    img_pro = img.copy()
    trans_flag = False
    if np.random.random()<Trans_r:
        scale = [1]
        Intersect_percent = 0.1
        random.shuffle(scale)
        ss = scale[0]
        x1_t = x1 - int(Intersect_percent * width * np.random.uniform(-1,1))
        y1_t = y1 - int(Intersect_percent * height *np.random.uniform(-1,1))
        img_pro = img_pro[y1_t:y1_t + int(height * ss), x1_t:x1_t + int(width * ss)]
        trans_flag = True

    if np.random.random() < Rotate_r:
        (h, w) = img.shape[:2]
        if trans_flag ==True:
            h1 = int(height * ss)
            w1 = int(width * ss)
        else:
            w1 = width
            h1 = height
        center = (w / 2, h / 2)
        angle_s=random.uniform(-10, 10)
        rec_scale = [0.8]
        random.shuffle(rec_scale)
        M = cv2.getRotationMatrix2D(center,angle_s,rec_scale[0])
        img_r = cv2.warpAffine(img,M,(h,w))
        # get new position of in rotate image
        new_position_x1 = int(M[0,0]*x1+M[0,1]*y1+M[0,2])
        new_position_y1 = int(M[1,0]*x1+M[1,1]*y1+M[1,2])
        new_position_x2 = int(M[0,0]*(x1+w1)+M[0,1]*y1+M[0,2])
        new_position_y2 = int(M[1,0]*(x1+w1)+M[1,1]*y1+M[1,2])
        new_position_x3 = int(M[0,0]*(x1)+M[0,1]*(y1+h1)+M[0,2])
        new_position_y3 = int(M[1,0]*(x1)+M[1,1]*(y1+h1)+M[1,2])
        new_position_x4 = int(M[0,0]*(x2)+M[0,1]*(y2)+M[0,2])
        new_position_y4 = int(M[1,0]*(x2)+M[1,1]*(y2)+M[1,2])

        a = [new_position_x1, new_position_x2, new_position_x3 ,new_position_x4]
        b = [new_position_y1, new_position_y2, new_position_y3, new_position_y4]
        new_min_x1 = np.min(a,axis=0)
        new_min_y1 = np.min(b,axis=0)
        new_max_x2 = np.max(a,axis=0)
        new_max_y2 = np.max(b,axis=0)
        img_pro = img_r[new_min_y1:new_max_y2, new_min_x1:new_max_x2]

    if np.random.random()< ratio_mirror:
        # print img_pro.size,img_pro.shape
        if img_pro.size ==0:
            return None
        img_pro = cv2.flip(img_pro, 1)

    if np.random.random()< ratio_coloraug:
        img_pro = img_pro * np.random.uniform(0.8,1.2)
        try:
            if img_pro.max() > 255:
                img_pro = img_pro / img_pro.max() * 255
        except ValueError:  # raised if `y` is empty.
            return None
        img_pro = img_pro.astype(np.uint8)
        # cv2.imshow("ori",img)
        # cv2.imshow("color_aug",img_pro)
        # cv2.waitKey(0)

    # if np.random.random()< ratio_noise:
    #     # img_pro = SaltAndPepper(img_pro,0.05)
    #     size_k = img.shape[0]
    #     img_pro = cv2.GaussianBlur(img_pro, (5, 5), sigmaX=1)
        
        # cv2.imshow("ori",img)
        # cv2.imshow("color_aug",img_pro)
        # cv2.imshow("gimg", gimg)
        # cv2.waitKey(0)

    if np.random.random()< ratio_move:
        size = 8
        # generating the kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        # applying the kernel to the input image
        img_pro = cv2.filter2D(img_pro, -1, kernel_motion_blur)
        # cv2.imshow("ori", img)
        # cv2.imshow('Motion Blur', img_pro)
        # cv2.waitKey(0)
    
    if np.random.random()<ratio_gray:
        img_gray = cv2.cvtColor(img_pro, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_pro = bgr
        # img_pro = np.matlib.repmat(bgr, 1, 3).reshape(img.shape[0], img.shape[1], 3, order='F')

    if np.random.random()<ratio_scale_o:
        scale = 1.2+ np.random.random()
        try:

            h_size = int(img_pro.shape[0]/scale)
            w_size = int(img_pro.shape[1]/scale)
            if img_pro is not None and img_pro.size!=0:
                img_pro_scale = cv2.resize(img_pro,(w_size,h_size), interpolation=cv2.INTER_CUBIC)
                img_pro = img_pro_scale
        except:
            print "hehe"
            return None

        # cv2.imshow("img_pro_scale", img_pro_scale)
        # cv2.imshow('img_pro', img_pro)
        # cv2.waitKey(0)
    return img_pro

def pro_ImageWithoutResize(img,bbox):



    pass


if __name__ == "__main__":
    img = cv2.imread("Emma.jpg")
    pro_Image(img)

