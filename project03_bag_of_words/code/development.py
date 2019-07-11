import numpy as np
import matplotlib
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from collections import Counter


def get_tiny_images(image_paths):
    """Resizes a list of images and returns them as one NxD numpy array.

    Parameters
    ---------
    image_paths : list
        A 1-D Python list of strings. Each string is a complete path to an 
        image on the filesystem.

    Returns
    -------
    N x D Numpy array where where n is the number of images and d is the
    length of the tiny image representation vector. e.g. if the images
    are resized to 16x16, then d is 16 * 16 = 256.
    """
    size = 16
    images_array = np.empty(shape=(len(image_paths), size**2))
    for i, file_i in enumerate(image_paths):
        tmp = imread(file_i)
        tmp = rgb2grey(tmp)
        tmp = resize(tmp, (size, size), anti_aliasing=True)
        images_array[i, ] = tmp.reshape((1, -1))

    return images_array


def build_vocabulary(image_paths, vocab_size):
    """Build and Cluster HOG descriptors and their centers.

    Parameters
    ----------
    image_paths : list(str)
        Image path strings.
    vocab_size : int
        The number of words desired for the bag of words
        vocabulary set.

    Returns
    -------
    a vocab_size x (z*z*9) (see below) array which contains the cluster
    centers that result from the K Means clustering.
    """
    z = 4
    size = 100
    hog_all = list()
    for i, file_i in enumerate(image_paths):
        if i % 100 == 0:
            print("Reading image {} of {}".format(i, len(image_paths)))
        image = imread(file_i)
        image = rgb2grey(image)
        image = resize(image, (size, size), anti_aliasing=True)
        hog_features = hog(image, orientations=9, pixels_per_cell=(4, 4),
                           cells_per_block=(z, z)).reshape(-1, z*z*9)
        hog_all.append(hog_features)
    print("Kmeans clustering")
    kmeans = KMeans(n_clusters=vocab_size, max_iter=100, n_jobs=1)
    hog_all = np.vstack(hog_all)
    kmeans.fit(np.array(hog_all))

    return kmeans.cluster_centers_


def get_bags_of_words(image_paths):
    pass


def svm_classify():
    pass


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    """Predicts the class of each image using a KNN classifier.

    Parameters
    ----------
    train_image_feats : Numpy array (N x D)
        An nxd numpy array, where N is the number of training examples, and
        D is the image descriptor vector size.
    train_labels : Numpy array (N x 1)
        An nx1 Python list containing the corresponding ground truth labels
        for the training data.
    test_image_feats : Numpy array (M x D)
        An MxD numpy array, where m is the number of test images and d is the
        image descriptor vector size.

    Returns
    -------
    An mx1 numpy list of strings, where each string is the predicted label for
    the corresponding image in test_image_feats
    """
    k = 5
    train_labels = np.array(train_labels)
    dist = cdist(test_image_feats, train_image_feats)
    num_test = dist.shape[0]
    y_pred = list()
    for i in range(num_test):
        # A list of length k storing the labels of the k nearest
        # neighbors to the ith test point.
        closest_y = []
        # Find k closest labels
        closest_y = train_labels[np.argsort(dist[i])[:k]]
        # Find most common label and store it. Ties are resolved randomly
        y_pred.append(Counter(closest_y).most_common(1)[0][0])
    return y_pred
