import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

import cv2

import lauetools.LaueTools.IOimagefile as IOimage

def reduce(im, sx, sy, lp, hp):
    SX, SY = im.shape
    im_ = im.reshape(sx, SX // sx, sy,SY // sy).mean(-1).mean(1)
    ss=cv2.dct(im_)
    ss[:lp, :] = 0.
    ss[:, :lp] = 0.
    ss[hp:, :] = 0.
    ss[:, hp:] = 0.
    return cv2.idct(ss), ss[lp: hp, lp: hp].ravel()
"""
filename= "140_S027_C090.TIFF"
dirname="../XRD_images/2nd_saving/"
x=IOimage.readCCDimage(filename=filename,CCDLabel='TIFF Format',dirname=dirname)
im=x[0].astype(np.float32)
im=(im-im.mean())/im.std()

x=reduce(im,128,128,1,65)
plt.imshow(x[0])
plt.show()

quit()
"""
features=np.zeros((64,146,4096))
s_range=range(64)
c_range=range(146)
for i in s_range:
    for j in c_range:
        if (i >= 100):
            str_i=str(i)
        else:
            if (i >= 10):
                str_i='0'+str(i)
            else:
                str_i='00'+str(i)
        if (j >= 100):
            str_j=str(j)
        else:
            if (j >= 10):
                str_j='0'+str(j)
            else:
                str_j='00'+str(j)
        path="../XRD_images/2nd_saving/"+"140_S"+str_i+"_C"+str_j+".TIFF"
        im=IOimage.readCCDimage(path,CCDLabel='TIFF Format')[0]
        if (im.mean() > 1e-10):
            im=(im-im.mean())/im.std()
        features[i,j]=reduce(im,128,128,1,65)[1]
np.save('features',features)
"""
quit()

features=np.load('features.npy')
features=features.reshape((64*146,1024))
from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
kpca = KernelPCA(n_components=7, kernel='linear')
X_transformed = kpca.fit_transform(features)

np.save("7_linear",X_transformed)
"""
