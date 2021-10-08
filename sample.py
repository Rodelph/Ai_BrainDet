import cv2 as cv
import numpy as np
from skimage.util import img_as_float
import FuzzyCMeans as fm

def sample_fcm_image():
    
    img = cv.imread('uploads/pred2.jpg')
    img = img_as_float(img)
    x = np.reshape(img,(img.shape[0]*img.shape[1],3),order='F')
    
    cluster_n = 6
    expo = 6
    min_err = 0.001 
    max_iter = 500
    verbose = 0
    m,c = fm.fcm(x,cluster_n,expo,min_err,max_iter,verbose)
    m = np.reshape(m,(img.shape[0],img.shape[1]),order='F')
    
    simg = fm.calc_median(img,m,verbose)
    img_c = ((simg + 1) * 255 / 2).astype('uint8')
    cv.imshow("title", img_c)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    sample_fcm_image()