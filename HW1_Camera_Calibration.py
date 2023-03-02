from sqlite3 import converters
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import glob

class chessboard():
    def __init__(self):
        figsize = (20, 20)
        self.CHECKERBOARD = (11,8)
        self.frameSize=(2048,2048)      
        #self.images = glob.glob("/Users/jagmohanmeher/Documents/NCKU/3rd sem/Computer vision course/Homework 1_code/Q1_Image/*.bmp")
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Vector for 3D points in real world space
        self.threedpoints = []
        # Vector for 2D points in the image plane
        self.twodpoints = []    
        #  3D points real world coordinates
        self.objectp3d = np.zeros((1, self.CHECKERBOARD[0]
                            * self.CHECKERBOARD[1],
                            3), np.float32) 
        self.objectp3d[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0],
                                    0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.prev_img_shape = None
        self.intrinsic_matrix = 0
        self.dist_matrix = 0


    def chess_calibration(self,file):
        file_name = file+"/*.bmp"
        self.images = glob.glob(file_name)
        for filename in self.images:
            image = cv2.imread(filename)
            print(image)
            self.grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            ret, corners = cv2.findChessboardCorners(self.grayColor, self.CHECKERBOARD, None)
            if ret == False: 
                print('chessboard not found')
            elif ret == True:
                print('yes')
                self.threedpoints.append(self.objectp3d)
                corners2 = cv2.cornerSubPix(self.grayColor, corners, (11,11), (-1, -1), self.criteria)
                self.twodpoints.append(corners2)
                image = cv2.drawChessboardCorners(image,
                                                self.CHECKERBOARD,
                                                corners2, ret)   
                cv2.imshow('Frame',image)
                cv2.waitKey(1000)
        print(file_name)

    
    def camera_calibration(self):
        #Camera calibration
        ret, self.intrinsic_matrix, self.dist_matrix, self.rvecs, self.tvecs = cv2.calibrateCamera(self.threedpoints, self.twodpoints, self.grayColor.shape[::-1], None, None)

    def intrinsic_parameter(self):
        #Print intrinsic matrix
        a = chessboard()
        ret, self.intrinsic_matrix, self.dist_matrix, self.rvecs, self.tvecs = cv2.calibrateCamera(self.threedpoints, self.twodpoints, self.grayColor.shape[::-1], None, None)
        print("Intrinsic matrix:\n", a.intrinsic_matrix)
        
    def extrinsic_parameter(self):
        extrinsic_matrix = cv2.cvFindExtrinsicCameraParams2(self.threedpoints,self.twodpoints, self.intrinsic_matrix,self.dist_matrix,self.rvec,self.tvec)

    def distortion_matrix(self):
        #Print distrotion matrix
        ret, self.intrinsic_matrix, self.dist_matrix, self.rvecs, self.tvecs = cv2.calibrateCamera(self.threedpoints, self.twodpoints, self.grayColor.shape[::-1], None, None)
        print("distortion matrix: ", self.dist_matrix.ravel())

    def undistort_the_images(self):
        plt.figure(figsize=self.figsize)

        #Undistorted images
        for i, filename in enumerate(self.images):
            print(i)
            image = cv2.imread(filename)
            imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            dst = cv2.undistort(imgRGB, self.intrinsic_matrix, self.dist_matrix)
            if i < 15:
                plt.subplot(4, 4, i + 1)
                plt.imshow(dst)

        plt.show()
        print("Done")

    cv2.destroyAllWindows() 



#if __name__ == "__main__":
    #chess = chessboard()
    #chess.chess_calibration()