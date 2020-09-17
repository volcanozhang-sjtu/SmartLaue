import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



features=np.load('features.npy')
features=features.reshape((64*146,4096))
#kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
y_pred = KMeans(n_clusters=100, random_state=170).fit_predict(features)

grains=np.array(y_pred).reshape((64,146))

np.save('KMeans100',grains)

plt.imshow(grains)
plt.show()
