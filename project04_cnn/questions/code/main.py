"""
CSCI 1430 Deep Learning project
By Ruizhao Zhu, Aaron Gokaslan, James Tompkin

This executable is used to launch the model on a given dataset. Additionally, it the data processing
parsing data into numpy arrays.

Usage: 
    python main.py -data [DATA] -mode [MODE]
        DATA | "mnist" or "scenerec"
        MODE | "nn" or "svm"

"""

import numpy as np
import hyperparameters as hp
import gzip
from skimage import io
from skimage.transform import resize
from model import Model
import os
import sys
import argparse

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("-data", help="Designates dataset to use. Valid values: mnist, scenerec")
parser.add_argument("-mode", help="Designates classifier to use. Valid values: nn, nn+svm")
parser.parse_args()

def load_data_scene(search_path, categories, size):
    images = np.zeros((size * hp.scene_class_count, hp.img_size * hp.img_size))
    labels = np.zeros((size * hp.scene_class_count,), dtype = np.int8)
    for label_no in range(hp.scene_class_count):
        img_path = search_path + categories[label_no]
        img_names = [f for f in os.listdir(img_path) if ".jpg" in f]
        for i in range(size):
            im = io.imread(img_path + "/" + img_names[i])
            im_vector = resize(im, (hp.img_size, hp.img_size)).reshape(1, hp.img_size * hp.img_size)
            index = size * label_no + i
            images[index, :] = im_vector
            labels[index] = label_no
    return images, labels

def get_categories_scene(search_path):
    dir_list = []
    for filename in os.listdir(search_path):
        if os.path.isdir(os.path.join(search_path, filename)):
            dir_list.append(filename)
    return dir_list

def format_data_scene_rec():
    train_path = "../../data/train/"
    test_path = "../../data/test/"
    categories = get_categories_scene(train_path)    
    train_images, train_labels = load_data_scene(train_path, categories, hp.num_train_per_category)
    test_images, test_labels = load_data_scene(test_path, categories, hp.num_test_per_category)
    return train_images, train_labels, test_images, test_labels

def format_data_mnist():
    # Reading in MNIST data.
    # Stolen from CS 1420

    # TODO: Update filepaths
    with open("../../data/train-images-idx3-ubyte.gz", 'rb') as f1, open("../../data/train-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
        train_images = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)
        train_images = np.where(train_images > 99, 1, 0)
        train_labels = np.frombuffer(buf2, dtype='uint8', offset=8)
    with open("../../data/t10k-images-idx3-ubyte.gz", 'rb') as f1, open("../../data/t10k-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        test_images = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        test_images = np.where(test_images > 99, 1, 0)
        test_labels = np.frombuffer(buf2, dtype='uint8', offset=8)

    return train_images, train_labels, test_images, test_labels

def main(argv):

    data = {"mnist", "scenerec"}
    mode = {"nn", "nn+svm"}

    if argv[1] not in data:
        raise ValueError("Data must be one of %r.", data)

    if argv[3] not in mode:
        raise ValueError("Mode must be one of %r.", mode)

    if argv[1] == "scenerec":
        train_images, train_labels, test_images, test_labels = format_data_scene_rec()
        num_classes = hp.scene_class_count
    else:
        train_images, train_labels, test_images, test_labels = format_data_mnist()
        num_classes = hp.mnist_class_count

    model = Model(train_images, train_labels, num_classes)

    if argv[3] == "nn":
        model.train_nn()
        accuracy = model.accuracy_nn(test_images, test_labels)
        print ('nn model training accuracy: {:.0%}'.format(accuracy))
    else:
        model.train_nn()
        model.train_svm()
        accuracy = model.accuracy_svm(test_images, test_labels)
        print ('nn+svm model training accuracy: {:.0%}'.format(accuracy))

if __name__ == '__main__':
    main(sys.argv[1:])
