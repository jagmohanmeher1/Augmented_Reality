import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob

def show_corners(file):
    figsize = (20, 20)
    CHECKERBOARD = (11,8)
    frameSize=(2048,2048)      
    #images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points in real world space
    threedpoints = []
    # Vector for 2D points in the image plane
    twodpoints = []    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    intrinsic_matrix = 0
    dist_matrix = 0
    file_name = file+"/*.bmp"
    images = glob.glob(file_name)
    for filename in images:
        image = cv2.imread(filename)
        print(image)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
        if ret == False: 
            print('chessboard not found')
        elif ret == True:
            print('yes')
        threedpoints.append(objectp3d)
        corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1, -1), criteria)
        twodpoints.append(corners2)
        image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)   
        cv2.imshow('Frame',image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    
def displayIntrinsicMatrix(file):
    figsize = (20, 20)
    CHECKERBOARD = (11,8)
    frameSize=(2048,2048)      
    #images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points in real world space
    threedpoints = []
    # Vector for 2D points in the image plane
    twodpoints = []    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    intrinsic_matrix = 0
    dist_matrix = 0
    file_name = file+"/*.bmp"
    images = glob.glob(file_name)
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
        if ret == False: 
            print('chessboard not found')
        elif ret == True:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1, -1), criteria)
            twodpoints.append(corners2)
            image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)   
        
    #Camera calibration
    ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    print(intrinsic_matrix)
  
def displayDistortionMatrix(file):
    figsize = (20, 20)
    CHECKERBOARD = (11,8)
    frameSize=(2048,2048)      
    #images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points in real world space
    threedpoints = []
    # Vector for 2D points in the image plane
    twodpoints = []    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    intrinsic_matrix = 0
    dist_matrix = 0
    file_name = file+"/*.bmp"
    images = glob.glob(file_name)
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
        if ret == False: 
            print('chessboard not found')
        elif ret == True:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1, -1), criteria)
            twodpoints.append(corners2)
            image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)   
        
    #Camera calibration
    ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    print(dist_matrix)
    #SolvePnPRansac for extrinsic matrix
    
def displayExtrinsicMatrix(file, image_no):
    figsize = (20, 20)
    CHECKERBOARD = (11,8)
    frameSize=(2048,2048)      
    #images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points in real world space
    threedpoints = []
    # Vector for 2D points in the image plane
    twodpoints = []    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    intrinsic_matrix = 0
    dist_matrix = 0
    file_name = file+"/*.bmp"
    images = glob.glob(file_name)
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
        if ret == False: 
            print('chessboard not found')
        elif ret == True:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1, -1), criteria)
            twodpoints.append(corners2)
            image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)   
        
    #Camera calibration
    ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    #ret, rvecs, tvecs = cv2.solvePnP(objectp3d, corners2, intrinsic_matrix, dist_matrix)
    rvecs = np.array(rvecs)
    r_mtx = np.zeros([len(images),3,3])
    
    for idx , n in enumerate(rvecs):
        matrix, jacobian = cv2.Rodrigues(n)
        r_mtx[idx,:,:] = matrix
        
    ext_mtx = np.append(r_mtx, tvecs, axis =2)
    print(ext_mtx[image_no])
    
    
    
def displayUndistortedImages(file):
    figsize = (20, 20)
    CHECKERBOARD = (11,8)
    frameSize=(2048,2048)      
    #images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for 3D points in real world space
    threedpoints = []
    # Vector for 2D points in the image plane
    twodpoints = []    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                        * CHECKERBOARD[1],
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    intrinsic_matrix = 0
    dist_matrix = 0
    file_name = file+"/*.bmp"
    images = glob.glob(file_name)
    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)
        if ret == False: 
            print('chessboard not found')
        elif ret == True:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1, -1), criteria)
            twodpoints.append(corners2)
            image = cv2.drawChessboardCorners(image,
                                        CHECKERBOARD,
                                        corners2, ret)   
        
    #Camera calibration
    ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    
    for i, filename in enumerate(images):
            print(i)
            image = cv2.imread(filename)
            imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            dst = cv2.undistort(imgRGB, intrinsic_matrix, dist_matrix)
            stacked_result = np.hstack((image,dst))
            cv2.imshow('Frame',stacked_result)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()