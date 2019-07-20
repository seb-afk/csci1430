import numpy as np
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


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
    hog_all = get_hog_features(image_paths)
    hog_all = np.vstack(hog_all)
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, max_iter=100)
    n_features = hog_all.shape[0]
    n_sample = int(n_features * .1)
    print("Kmeans clustering of matrix with shape: {}".format(hog_all.shape))
    hog_all = hog_all[np.random.choice(n_features, n_sample, replace=False), :]
    print("Kmeans clustering of matrix with shape: {}".format(hog_all.shape))
    kmeans.fit(hog_all)
    return kmeans.cluster_centers_


def get_hog_features(image_paths):
    """Get HOG features for each image.

    Parameters
    ----------
    image_paths : list(str*())
        A Python list of strings, where each string is a complete path to one
        image on the disk.

    Returns
    -------
    list(array_1, array_2, array_i)
        List of Numpy arrays with shape (NxD) containing the Hog features for
        each image. N is the number of features and D the number of dimensions.
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
    return hog_all


def get_bags_of_words(image_paths):
    """Get bag of words histogram for each image.

    Parameters
    ----------
    image_paths : list(str())
        A Python list of strings, where each string is a complete path to one
        image on the disk.

    Returns
    -------
    bag_of_words : Numpy array (NxD)
        An (NxD) numpy matrix, where N is the number of images in image_paths
        and D is size of the histogram built for each image.
    """
    hog_all = get_hog_features(image_paths)
    vocab = np.load("../data/vocab.npy")
    vocab_size = vocab.shape[0]
    results = list()
    for hog_i in hog_all:
        dist = cdist(hog_i, vocab)
        num_features = dist.shape[0]
        y_pred = list()
        for i in range(num_features):
            closest_y = np.argsort(dist[i])[0]
            y_pred.append(closest_y)
        bincount = np.bincount(np.array(y_pred), minlength=vocab_size)
        results.append(bincount / sum(y_pred))
    return np.vstack(results)


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """Train SVC and classify test images.

    Parameters
    ----------
    train_image_feats : Numpy array (N x D)
        Feature matrix for training.
    train_labels : list()
        Python list of length N.
    test_image_feats : Numpy array (M x D)
        Feature matrix for testing.

    Returns
    -------
    Numpy array (M x 1) of strings.
        Predicted labels for the test images
    """
    scaler = StandardScaler()
    scaler.fit(train_image_feats)
    train_image_feats_sc = scaler.transform(train_image_feats)
    test_image_feats_sc = scaler.transform(test_image_feats)
    svc = LinearSVC()
    svc.fit(train_image_feats_sc, train_labels)
    y_predict = svc.predict(test_image_feats_sc)
    return list(y_predict)


def nearest_neighbor_classify(train_feats, train_labels, test_feats):
    """Predicts the class of each image using a KNN classifier.

    Parameters
    ----------
    train_feats : Numpy array (N x D)
        An nxd numpy array, where N is the number of training examples, and
        D is the image descriptor vector size.
    train_labels : Numpy array (N x 1)
        An nx1 Python list containing the corresponding ground truth labels
        for the training data.
    test_feats : Numpy array (M x D)
        An MxD numpy array, where m is the number of test images and d is the
        image descriptor vector size.

    Returns
    -------
    An mx1 numpy list of strings, where each string is the predicted label for
    the corresponding image in test_feats.
    """
    k = 5
    train_labels = np.array(train_labels)
    dist = cdist(test_feats, train_feats)
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
