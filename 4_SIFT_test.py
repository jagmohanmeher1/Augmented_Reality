import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")

# def read_image1(path):
#     img1 = cv2.imread(path)
#     sift = cv2.SIFT_create( nfeatures =  200)
#     kp1, dsc1 = sift.detectAndCompute(img1, None)
#     img1 = cv2.drawKeypoints(img1, kp1, None)
#     cv2.imshow('Frame 1',img1)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()
    
# def read_image2(path):
#     img2 = cv2.imread(path)
#     sift = cv2.SIFT_create( nfeatures =  200)
#     kp1, dsc1 = sift.detectAndCompute(img2, None)
#     img2 = cv2.drawKeypoints(img2, kp1, None)
#     cv2.imshow('Frame 2',img2)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()
    
def display_SIFT_keypoints(path):
    img1 = cv2.imread(path)
    sift = cv2.SIFT_create()
    kp1, dsc1 = sift.detectAndCompute(img1, None)
    img1 = cv2.drawKeypoints(img1, kp1, None)
    cv2.imshow('Frame 1',img1)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
def SIFT_keypoint_matching(path1,path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    sift = cv2.SIFT_create()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, dsc1 = sift.detectAndCompute(img1, None)
    kp2, dsc2 = sift.detectAndCompute(img2, None)
    index = dict(algorithm = 0, trees =5 )
    search = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index,search)
    matches = flann.knnMatch(dsc1, dsc2, k=2)
    n_matches = len(matches)
    mask = [[0,0] for i in range(n_matches)]
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            mask[i] = [1,0]
    draw = dict(matchColor = (0,255,0), 
                singlePointColor = (0,0,255),
                matchesMask=mask,
                flags = 0)

    matches_img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, **draw)
    cv2.imshow('Frame',matches_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    
    
