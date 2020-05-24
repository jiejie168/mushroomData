__author__ = 'Jie'
'''
the data are from the course called: applied machine learning in python, week 2
the visualization of fit model boundary is quit interesting,which can be used for future
# play for the data. it is a classification problem.
# different classifiers are used for comparison.
# only two important features are adopted via principal component analysis (PCA)
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#######################################
# load the data
df=pd.read_csv("mushrooms.csv")
df1=pd.get_dummies(df)
df2=df1.sample(frac=0.08) # only select 8% of original data
x=df2.iloc[:,2:]
y=df2.iloc[:,1] # select the "class_p" as y ; 1: poisonous.

#perform Principal Component Analysis (PCA) on
#  the original training data to reduce the dimensionality to 2 dimensions.
pca=PCA(n_components=2).fit_transform(x) # return ndarray, only select 2 important features
X_train,X_test,y_train,y_test=train_test_split(pca,y,random_state=0)

###############################################################
# define a plot function to plot data with/without Poisonous
def plotPoisonous(X,y):
    # X: array; y : dataFrame
    plt.figure(figsize=(8,6))
    # X[y.values==0,0]: elements corresponding the edible mushroom in the first column
    plt.scatter(X[y.values==0,0],X[y.values==0,1],label="edible",alpha=0.6,s=4)
    plt.scatter(X[y.values==1,0],X[y.values==1,1],label="poisonous",alpha=0.6,s=4)
    plt.legend()
    plt.title("scatter plot of data\nfirst two principal components")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.show()

# plotPoisonous(pca,y)
########################################################################
# define a function to plot the boundary of fit model
def plot_mushroom_boundary(X,y,model):
    plt.figure(figsize=(10,5))
    for i,type in enumerate (['decision boundary','decision probability']):
        plt.subplot(1,2,i+1)
        steps=0.01
        x_min,x_max=X[:,0].min()-.1,X[:,0].max()+ .1  # range of 1st feature
        y_min,y_max=X[:,1].min()-.1,X[:,1].max()+ .1  # range of 2nd feature
        # this is used to create a grid matrix.
        # xx,yy=meshgrid(x,y): xx expands to ndarray (len(y),len(x)); yy expands to ndarray (len(x),len(y))
        # the purpose of meshgrid here is to produce a rectangular grid out of an array x, and an array of y.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, steps),
                             np.arange(y_min, y_max, steps))
        if i==0:
            # np.c_[]: join the two arrays by row. the rows should be equal.
            # xx.ravel(): squeeze the array to 1 dimension.
            # calculate the prediction values for data btw the arange
            Z=model.predict(np.c_[xx.ravel(),yy.ravel()])  # (len(xx.ravel(),2), its shape is equal to the shape of X_test
            # aa=np.c_[xx.ravel(),yy.ravel()]
            # print (xx.ravel().shape)
            # print (aa.shape)
            # print (aa)
        else:
            try:
                Z=model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
            except:
                plt.text(0.4, 0.5, 'Probabilities Unavailable', horizontalalignment='center',
                verticalalignment='center', transform = plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                break
        Z=Z.reshape(xx.shape)
        plt.scatter(X[y.values==0,0], X[y.values==0,1], alpha=0.4, label='Edible', s=5)
        plt.scatter(X[y.values==1,0], X[y.values==1,1], alpha=0.4, label='Posionous', s=5)
        #Display data as an image; i.e. on a 2D regular raster.
        #origin{'upper', 'lower'}, optional.  the original point of display the image: upper left, or lower left.
        # plot the prediction boundary of data inside the range of training data
        plt.imshow(Z,interpolation='nearest', cmap='RdYlBu_r', alpha=0.15,
                   extent=(x_min, x_max, y_min, y_max), origin='lower')
        plt.legend()
        plt.title(type + '\n' +
                  str(model).split('(')[0]+ ' Test Accuracy: ' + str(np.round(model.score(X, y), 5)))
        plt.gca().set_aspect('equal');
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.02)
    plt.show()

# classifier problem.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# model = LogisticRegression()
# model=KNeighborsClassifier(n_neighbors=20)
# model=DecisionTreeClassifier(max_depth=3)
# model=RandomForestClassifier()
# model=SVC(kernel='linear',C=2)
model=SVC(kernel='rbf',C=10)
model.fit(X_train,y_train)
plot_mushroom_boundary(X_test, y_test, model)
