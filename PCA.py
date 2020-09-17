import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



features=np.load('features.npy')
features=features.reshape((64*146,4096))
#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
y_pred = PCA(n_components=100, svd_solver='randomized').fit(features)

xxx=y_pred.components_

fig=plt.figure()
plt.subplot(2,3,1)
plt.hist(xxx[0,:],bins=100)
plt.subplot(2,3,2)
plt.hist(xxx[1,:],bins=100)
plt.subplot(2,3,3)
plt.hist(xxx[2,:],bins=100)

plt.subplot(2,3,4)
plt.hist(xxx[3,:],bins=100)
plt.subplot(2,3,5)
plt.hist(xxx[4,:],bins=100)
plt.subplot(2,3,6)
plt.hist(xxx[5,:],bins=100)

plt.show()
"""

grains=np.array(y_pred).reshape((64,146))

np.save('KMeans100',grains)

plt.imshow(grains)
plt.show()
"""
