# %% [markdown]
#
# **Question 1**
#
# In machine learning, what are bias and variance? When we evaluate a
# classifier, what are overfitting and underfitting, and how do these relate to
# bias and variance?
#
# **Answer 1**
#
# If we would test our hypothesis on many different datasets and take our
# average result the bias would tell us the difference between this average
# result and the true result. In contrast, if we measure how much our result
# differs everytime we test our hypothesis on a new dataset we get the variance.
# In other words bias measures how good we are on average and variance measures
# how consistent we are.
#
# Overfitting: we fit the noise in the data rather than the true underlying
# pattern. Our model is too complex. We have low bias (many degrees of freedom)
# but high variance. Underfitting: we are note able to match the complexity of
# the data at hand. We have high bias but in contrast low variance.

# %% [markdown]
#
# **Question 2**
#
# Given a linear classifier like SVM, how might we handle data that are not
# linearly separable? How does the \emph{kernel trick} help in these cases? (See
# hidden slides in supervised learning crash course deck, plus your own
# research.)
#
# **Answer 2**
#
# We could map the data to a non-linear or higher-dimensional feature space
# where the data does become linearly separable but the model still consists of
# a linear combination of the features.

# %% [markdown]
#
# **Question 3**
#
# Given a linear classifier such as SVM which separates two classes (binary
# decision), how might we use multiple linear classifiers to create a new
# classifier which separates $k$ classes?

# Below, we provide pseudocode for a linear classifier. It trains a model on a
# training set, and then classifies a new test example into one of two classes.
# Please convert this into a multi-class classifier. You can take either the one
# vs.~all (or one vs.~others) approach or the one vs.~one approach in the
# slides; please declare which approach you take.

# %%
def classify(train_feats, train_labels, test_feats):
    """

    Parameters
    ----------

    train_feats: N x d matrix
        N images each d descriptors long.

    train_labels: N x C matrix
        One hot encoded vectors specifying the target class of each 
        observation.

    test_feats: N x d matrix
        N images each d descriptors long for which we want to predict a class
        label.

    Returns
    -------

    test_label: N x 1 matrix
        Vector of integers denoting the predicted target class.
    """
    # Train classification hyperplane
    weights, bias = train_linear_classifier(train_feats, train_label)
    # Compute distance from hyperplane
    test_scores = weights * test_feats + bias
    test_label = np.amax(test_scores, axis=1)

    return test_label

# %% [markdown]
#
# **Question 4**
#
# Suppose we are creating a visual word dictionary using SIFT and k-means
# clustering for a scene recognition algorithm. Examining the SIFT features
# generated from our training database, we see that many are almost equidistant
# from two or more visual words. Why might this affect classification accuracy?
# 
# Given the situation, describe \emph{two} methods to improve classification
# accuracy, and explain why they would help.
# TODO
# If many features are ambiguous and do not discriminate well between the
# visual words it means they are uninformative. This can be a problem for
# algorithms that cannot distinguish between informative and uninformative
# features. For example the KNN classifier.
# %% [markdown]
#
# **Question 5**
#
# \paragraph{Q5:} The way that the bag of words representation handles the
# spatial layout of visual information can be both an advantage and a
# disadvantage. Describe an example scenario for each of these cases, plus
# describe a modification or additional algorithm which can overcome the
# disadvantage.

# How might we evaluate whether bag of words is a good model?