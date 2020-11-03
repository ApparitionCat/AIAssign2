import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import datasets
from mnist import MNIST
import yellowbrick.cluster as yb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




#-------------iris------------------------------------------
def runIris():
    iris = datasets.load_iris()
    colormap=np.array(['Red','green','blue'])
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']
    model=KMeans(n_clusters=3, random_state=10, n_init = 10)
    model.fit(x)
    plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[model.labels_],s=40)
    plt.title('Iris')
    plt.show()
#---------------------------------------------------------------
#------synt-----------------------------------------------------
def runSynt():
    df = pd.DataFrame({
    'x': [12, 20, 28, 38, 50, 46, 31, 50, 11, 21, 45, 33, 33, 65, 75, 39, 33, 29, 61],
    'y': [39, 16, 30, 42, 34, 16, 65, 39, 73, 50, 76, 63, 68, 33, 4, 8, 19, 7, 24]})
    colmap = {1: 'r', 2: 'b', 3: 'g'}
    model = KMeans(n_clusters=3, init='random', n_init=10, random_state=0)
    model.fit(df)
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.scatter(df['x'], df['y'], c= model.labels_.astype(float), alpha=0.5)
    #elbow(df)
    #sil(df)
    plt.show()

#-------MNIST--------------------------------------------------
def runMNIST():
    mndata = MNIST("./MNIST")
    mndata.gz = True
    xTrain, yTrain = mndata.load_training()
    x = StandardScaler().fit_transform(xTrain)
    pcaConv= PCA(5)
    xConv = pcaConv.fit_transform(x)
    xConv.shape
    model = KMeans(n_clusters=10, init='random', n_init=10, random_state=0)
    model.fit(xConv, yTrain)
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.scatter(x=xConv[:, 0], y=xConv[:,1], c= model.labels_.astype(float), alpha=0.5)
    plt.show()
#-----------------------------------------------------------------------------------
#def RunStl():
    #i did not expect the binary files to be 2.5GB, im sorry for leaving this empty ;_;

#task 1.2----------------------------------------------------------------
def elbow(x):
    errDist =[]
    I = range(1, 10)
    for i in I:
        modelbow = KMeans(n_clusters = i)
        modelbow.fit(x)
        errDist.append(modelbow.inertia_)
    plt.plot(I, errDist, "bx-")
    plt.title('Elbow')
    plt.show()

def sil(x):
    I = range(2,6)
    for i in I:
        silModel = KMeans(n_clusters=i)
        visualizer = yb.SilhouetteVisualizer(silModel, colors="Blues")
        visualizer.fit(x)
        plt.title("Silhouette")
        plt.show()
    visualizer.poof()


#runSynt()
runIris()
#runMNIST()
#RunStl()
