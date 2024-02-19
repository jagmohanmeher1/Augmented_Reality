import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob

def disparity_map(path1,path2):
    imgL = cv2.imread(path1)
    imgR = cv2.imread(path2)
    gray_l = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(imgL, (699, 476))
    l = cv2.resize(imgR, (699, 476))
    stereo = cv2.StereoBM_create(256,25)
    disp = stereo.compute(gray_l,gray_r).astype(np.float32)/(16) 
    print(disp)
    cv2.setMouseCallback('image',draw_circle)
    cv2.imshow("figure",disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #stereoSBGM 
    
# mouse callback function
def draw_circle(event,x,y,flags,param):
    imgR = cv2.imread(path2)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2,(x,y),100,(255,0,0),-1)
    
