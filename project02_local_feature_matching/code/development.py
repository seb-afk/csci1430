import numpy as np
from scipy.spatial.distance import cdist

def get_idx_2smallest_dist(distances):
    """Given a distance matrix it returns the index of the k
    smallest distances.

    Parameters
    ----------

    distances: 2d-array distance matrix

    Returns
    -------
    
    d1_ix, d2_ix: Tuple.
        For eact row of the distance matrix it returns the index of the smallest
        and second smallest distance.

    """
    n_rows = distances.shape[0]
    idxs = np.argsort(distances, axis=1)[:, :2]
    d1_ix = [np.arange(n_rows), idxs[:,0]]
    d2_ix = [np.arange(n_rows), idxs[:,1]]
    return d1_ix, d2_ix


def match_features(im1_features, im2_features):
    """ Returns the index of the common image features as well as the 
    confidence of a correct match.
    
    Parameters
    ----------
    
    im1_features: array, features of image 1.
    
    im2_features: array, features of image 2.
    
    Returns
    -------
    
    matches: array
        First column = matches for image 1.
        Second column = matches for image 2.
    
    confidences: list, confidence for each match identified. 
        We define it as:

            confidence = 1 - NNDR
        
        where:

        NDDR = \frac{d_1}{d_2} = \frac{||D_A - D_B||}{||D_A - D_C||}

        See Equation 4.18 in Section 4.1.3 of Szeliski.
    """
    distances = cdist(im1_features, im2_features, metric="euclidean")
    d1_ix, d2_ix = get_idx_2smallest_dist(distances)
    nddr = distances[d1_ix] / distances[d2_ix]

    matches = np.stack((d1_ix[0], d1_ix[1]), axis=1)
    confidences = 1 - nddr
    
    return matches, confidences

def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    
    Parameters
    ----------

    image: a grayscale or color image (your choice depending on your 
        implementation)
    x: np array of x coordinates of interest points
    y: np array of y coordinates of interest points
    feature_width: in pixels, is the local feature width. You can assume
        that feature_width will be a multiple of 4 (i.e. every cell of your
        local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    Returns
    -------

    features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    features = np.asarray([0])

    return features