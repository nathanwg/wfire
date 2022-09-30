import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('img.jpg')
scale = 1
width = int(img.shape[1]*scale)
height = int(img.shape[0]*scale)
dim = (width,height)
img = cv.resize(img,dim)
img = img[:,:,0]
img_blur = cv.GaussianBlur(img,(3,3),sigmaX=0,sigmaY=0)
plt.imshow(img)
plt.get_current_fig_manager().window.state('zoomed')
x = np.linspace(0,639,640)
##for i in range(1,10):
##    y = np.linspace(i,i,640)*64
##    plt.plot(x,y,'w')
##    plt.plot(y,x,'w')
plt.xticks(ticks=[])
plt.yticks(ticks=[])
##plt.show()
p = plt.ginput(n=-1,timeout=-1,show_clicks=True)
plt.close()
print(p)


##threshold_1 = 0
##threshold_2 = 250
##edges = cv.Canny(img,threshold_1,threshold_2)
##other = cv.Laplacian(img_blur,cv.CV_64F)
##
##
##current = other
##imgs = np.concatenate((img,current),axis=1)
##
##cv.imshow('image',other)
##cv.waitKey(0)

