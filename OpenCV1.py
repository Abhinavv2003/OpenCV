import cv2 as cv
import numpy as np


import matplotlib.pyplot as plt


#img=cv.imread('/Users/abhinavv20/Downloads/cameraman.png')
img=cv.imread('/Users/abhinavv20/Documents/MATLAB/cameraman.tif')
#img=img.reshape(-1,img.shape[-1])
assert img is not None , "file could not be read check with os.path.exists()"
#kernel1=np.ones((5,5),np.float32)/25
#kernel2=np.ones((5,5),np.float64)
kernel=np.array([[1 ,4 ,7 ,4 ,1],[4 ,16 ,26 ,16 ,4],[7 ,26 ,41 ,26 ,7],[4 ,16 ,26, 16 ,4],[1, 4, 7, 4, 1]])/64
#kernel=np.array([[0,0,1,2,1,0,0],[0,3,13,22,13,3,0],[1,13,59,97,59,13,1],[2,22,97,159,97,22,2],[1,13,59,97,59,13,1],[0,3,13,22,13,3,0],[0,0,1,2,1,0,0]])/324
#kernel = kernel.astype(int)
print(kernel.dtype)
kernel = kernel.astype(np.float16)
print(kernel.dtype)

dst=cv.filter2D(img,-1,kernel)
#dst1=cv.filter2D(img,-1,kernel2)
plt.subplot(121),plt.imshow(img),plt.title('original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('filtered')
plt.xticks([]),plt.yticks([])
#plt.subplot(122),plt.imshow(dst1),plt.title('fp32')
#plt.xticks([]),plt.yticks([])
plt.show()

#print(img.shape)