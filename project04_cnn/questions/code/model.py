import numpy as np
import hyperparameters as hp
import random

from sklearn.svm import LinearSVC

class Model:

    def __init__(self, train_images, train_labels, num_classes):
        self.input_size = train_images.shape[1]
        self.num_classes = num_classes
        self.learning_rate = hp.learning_rate
        self.batchSz = hp.batch_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.clf = LinearSVC(multi_class='ovr',dual=False)


        # sets up weights and biases...
        self.W = np.random.rand(self.input_size, self.num_classes)
        self.b = np.zeros((1, self.num_classes))

    def train_nn(self):
        """
        Simple neural network model.
        This function computes the forward pass, loss calculation, and 
        back propagation for this model. 
        The neural network contains one fully connected layer with a softmax unit, 
        and the loss function is cross-entropy loss.
        
        The basic structure of this part is a loop over the number of epoches 
        we wish to use (hp.num_epochs).
        For each epoch, we need to iterate through all training data.
        The batch size is 1, meaning that we should update the model parameters 
        for each training data at every epoch.
        """
        
        # This is our training data as indices into an image storage array
        indices = list(range(self.train_images.shape[0]))

        # These are our storage variables for our update gradients.
        # delta_W is the matrix of gradients on the weights of our neural network
        #     Each row is a different neuron (with its own weights)
        # delta_b is the vector of gradients on the biases (one per neuron)
        delta_W = np.zeros((self.input_size, self.num_classes))
        delta_b = np.zeros((1, self.num_classes))

        # Iterate over the number of epochs declared in the hyperparameters.py file
        for epoch in range(hp.num_epochs):
            # Overall per-epoch sum of the loss
            loss_sum = 0
            # Shuffle the data before training each epoch to remove ordering bias
            random.shuffle(indices)

            # For each image in the datset:
            for index in range(len(indices)):
                # Get input training image and ground truth label
                i = indices[index]
                img = self.train_images[i]
                gt_label = self.train_labels[i]

                ################
                # GENERAL ADVICE:
                # This is _precise work_ - we need very few lines of code.
                # At this point, we need not write any for loops.
                # As a guide, our solution has 14 lines of code.
                #
                # Further, please do not use any library functions; our solution uses none.

                ################
                # FORWARD PASS:
                # This is where we take our current estimate of the weights and biases
                # and compute the current error against the training data.

                # Step 1:
                # Compute the output response to this 'img' input for each neuron (linear unit).
                # Our current estimate for the weights and biases are stored in:
                #    self.W
                #    self.b
                # Remember: use matrix operations.
                # Our output will be a number for each neuron stored in a vector.


                # Step 2:
                # Convert these to probabilities by implementing the softmax function.              


                # Step 3:
                # Compute the error against the training label 'gt_label' using the cross-entropy loss
                # Remember:
                #     log has a potential divide by zero error

                #loss_sum = loss_sum + your_loss_over_all_classes
                

                ################
                # BACKWARD PASS (BACK PROPAGATION):            
                # This is where we find which direction to move in for gradient descent to 
                # optimize our weights and biases.
                # Use the derivations from the questions handout.

                # Step 4:
                # Compute the delta_W and delta_b gradient terms for the weights and biases
                # using the provided derivations in Eqs. 6 and 7 of the handout.
                # Remember:
                #    delta_W is a matrix the size of the input by the size of the classes
                #    delta_b is a vector
                # Note:
                #    By equation 6 and 7, we need to subtract 1 from p_j only if it is 
                #    the true class probability.


                # Step 5:
                # Update self.W and self.b using the gradient terms 
                # and the self.learning_rate hyperparameter.
                # Eqs. 4 and 5 in the handout.
                


                # Once trained, self.W and self.b will be used in accuracy_nn()
                # to evalute test performance.
                #
                # When running for 1 epoch:
                # > python3 main.py -data mnist -mode nn
                # 
                # Epoch 0: Total loss: ~210000
                # Training accuracy: ~88%.

                
            print( "Epoch " + str(epoch) + ": Total loss: " + str(loss_sum) )


    def train_svm(self):
        """
        Use the response from the learned weights and biases on the training data
        as input into an SVM. I.E., train an SVM on the multi-class hyperplane distance outputs.
        """
        # Step 1:
        # Compute the response of the hyperplane on the training image
        scores = np.dot(self.train_images, self.W) + self.b

        # Step 2:
        # Fit an SVM model to these. Uses LinearSVC function declare in self.clf
        # This will be used later on in accuracy_svm()
        self.clf.fit(scores,self.train_labels)

    def accuracy_nn(self, test_images, test_labels):
        """
        Computer the accuracy of the neural network model over the test set.
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = np.argmax(scores, axis=1)
        return np.mean(predicted_classes == test_labels)

    def accuracy_svm(self, test_images, test_labels):
        """
        Computer the accuracy of the svm model over the test set.
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = self.clf.predict(scores)
        return np.mean(predicted_classes == test_labels)
