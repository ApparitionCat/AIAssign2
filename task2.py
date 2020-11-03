#----------Task 2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimg

imgData = mpimg.imread('cat.png')
height, widthth, channels = imgData.shape
chanData = imgData.reshape((height,widthth*channels))
ipca = PCA(20).fit(chanData)
imgC = ipca.transform(chanData)
temp = ipca.inverse_transform(imgC)
temp = np.reshape(temp, ( height, widthth, channels))
plt.imshow(temp)
plt.show()
