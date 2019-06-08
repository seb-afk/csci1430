import unittest
import numpy as np
import development as dev

class TestGetIdx2smallestDist(unittest.TestCase):
    def test_correct_distances(self):
        distances = np.array([[0., 1., 2.],
                              [1., 4., 3.],
                              [5., 2., 2.]])
        d1_ix, d2_ix = dev.get_idx_2smallest_dist(distances)
        d1 = distances[d1_ix]
        d2 = distances[d2_ix]
        np.testing.assert_array_equal(d1, [0, 1, 2])
        np.testing.assert_array_equal(d2, [1, 3, 2])

class TestMatchFeatures(unittest.TestCase):
    def test_feature_matching(self):
        im1_features = np.array([[1], [200], [100]])
        im2_features = np.array([[1.1], [1.2], [202]])

        matches, confidences = dev.match_features(im1_features, im2_features)
        im1_matches, im2_matches = matches[:,0], matches[:,1]
        confidences_rounded = np.round(confidences, decimals=3)

        np.testing.assert_array_equal(im1_matches, [0, 1, 2])
        np.testing.assert_array_equal(im2_matches, [0, 2, 1])
        np.testing.assert_array_equal(confidences_rounded, [0.5  , 0.99 , 0.001])


if __name__ == '__main__':
    unittest.main()