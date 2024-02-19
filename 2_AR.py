import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
    
def AR_horizontal_words(file,str):
    fs = cv2.FileStorage("alphabet_lib_onboard.txt", cv2.FILE_STORAGE_READ)   
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
    shift = np.array([
        [7,5],[4,5],[1,5],[7,2],[4,2],[1,2]
    ])
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
            # #image = cv2.drawChessboardCorners(image,
            #                             CHECKERBOARD,
            #                             corners2, ret)   
        
            #Camera calibration
            print(threedpoints)
            ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
            ret, rvecs, tvecs = cv2.solvePnP(objectp3d, corners2, intrinsic_matrix, dist_matrix)
            for v,n in enumerate(str):
                ch = fs.getNode(n).mat()
                number_of_points = 2*len(ch[:])
                ch1 = np.resize(ch,(number_of_points,3))

                for i, x in enumerate(ch1):
                    ch1[i,0], ch1[i,1] = (ch1[i,0] + shift[v,0],ch1[i,1]+shift[v,1])
                
                # Project 3D points to image plane
                chessboard_corners = np.float32(ch1)
                imgpts, jac = cv2.projectPoints(chessboard_corners,rvecs, tvecs, intrinsic_matrix, dist_matrix)
                imgpts = imgpts.astype(int)
                imgpts = np.ravel(imgpts)
                q = []
                for i, n in enumerate(imgpts):
                    q.append(n)
                    if len(q)==4:
                        image = cv2.line(image,(q[0],q[1]),(q[2],q[3]),(0,0,255), 10)
                        q.clear()

        image = cv2.resize(image, (900,900))
        cv2.imshow('img',image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
def AR_vertical_words(file,str):
    fs = cv2.FileStorage("alphabet_lib_vertical.txt", cv2.FILE_STORAGE_READ)   
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
    shift = np.array([
        [7,5],[4,5],[1,5],[7,2],[4,2],[1,2]
    ])
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
            # #image = cv2.drawChessboardCorners(image,
            #                             CHECKERBOARD,
            #                             corners2, ret)   
        
            #Camera calibration
            print(threedpoints)
            ret, intrinsic_matrix, dist_matrix, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
            ret, rvecs, tvecs = cv2.solvePnP(objectp3d, corners2, intrinsic_matrix, dist_matrix)
            for v,n in enumerate(str):
                ch = fs.getNode(n).mat()
                number_of_points = 2*len(ch[:])
                ch1 = np.resize(ch,(number_of_points,3))

                for i, x in enumerate(ch1):
                    ch1[i,0], ch1[i,1] = (ch1[i,0] + shift[v,0],ch1[i,1]+shift[v,1])
                
                # Project 3D points to image plane
                chessboard_corners = np.float32(ch1)
                imgpts, jac = cv2.projectPoints(chessboard_corners,rvecs, tvecs, intrinsic_matrix, dist_matrix)
                imgpts = imgpts.astype(int)
                imgpts = np.ravel(imgpts)
                q = []
                for i, n in enumerate(imgpts):
                    q.append(n)
                    if len(q)==4:
                        image = cv2.line(image,(q[0],q[1]),(q[2],q[3]),(0,0,255), 10)
                        q.clear()

        image = cv2.resize(image, (900,900))
        cv2.imshow('img',image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

