#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
import math
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
from skimage import filter
from skimage import feature
from skimage import transform
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max


#my functions
import neuralnet
import gridsearch







# make graphics inline
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")



# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
 ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))
 
 
 
 
# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[5],"*.jpg"))[9]
print example_file
im = imread(example_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show()


# First we threshold the image by only taking values greater than the mean to reduce noise in the image
# to use later as a mask
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)


# calculate common region properties for each region within the segmentation
regions = measure.regionprops(labels)
# find the largest nonzero region
def getLargestRegion(props=regions, labelmap=labels, imagethres=imthr):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop
    
    
    
regionmax = getLargestRegion()
plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
plt.show()


print regionmax.minor_axis_length/regionmax.major_axis_length


def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio
    

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()
    
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def noise_dataset(X,Y):
    #add noise to images used to train the network

    

#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              numberofImages += 1
              

# rescale images to squares
maxPixel = 200

imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
#2 x canny edge detection + fourier mag + fourier phase
num_features = imageSize + 1 # for our ratio


# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()



print "Reading folders"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            

            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            files.append((nameFileImage, label))
            

    label += 1


print "Making features"

for nameFileImage, label in files:
    image = imread(nameFileImage, as_grey=True)

    #create other images here


    #keep this metric
    axisratio = getMinorMajorRatio(image)
    
    #fancy image processing goes here
    edges1 = filter.canny(image)
    
    edges2 = filter.canny(im, sigma=3)
	
    arr,hogIM=feature.hog(image, orientations=9, pixels_per_cell=(3, 3), cells_per_block=(3, 3), visualise=True, normalise=False)

    #resize to square
    image = resize(image, (maxPixel, maxPixel))
    
    
    # Store the rescaled image pixels and the axis ratio
    
    #create first layer of feature groups here
    X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
    

    X[i, imageSize] = axisratio
    
    # Store the classlabel
    y[i] = label
    i += 1
    # report progress for each 5% done  
    report = [int((j+1)*num_rows/20.) for j in range(20)]
    if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    
    
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features


scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);

print "Accuracy of all classes"
print np.mean(scores)
print classification_report(y, y_pred, target_names=namesClasses)
   
# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)

# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y),len(set(y))))


#init neural net
learning_rates ={}
n_components = {}
featureGroups ={}
trainY ={}

NN = neuralnet(n_components,learning_rates,featureGroups, trainY, X,Y)

while time < :
    #backprop and retrain

    #save network to keep training later



#verify classification

for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    
    #fit random forest to neural output
    clf = RF(n_estimators=100, n_jobs=3)
    
    X_train = NN.transformData(X_train)
    X_test = NN.transformData(X_test)

    clf.fit(X_train, y_train)
    
    y_pred[test] = clf.predict_proba(X_test)
    
    print classification_report(y_test, y_pred, target_names=namesClasses)
    
    scores = cross_validation.cross_val_score(clf, X_test , y_test, cv=5, n_jobs=1);
    
    print "Accuracy of all classes"
    
    print np.mean(scores)

    print 'log-loss'
    multiclass_log_loss(y, y_pred)


