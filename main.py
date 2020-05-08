"""
Project 6 - Model 3 
CS1430 - Computer Vision
Brown University
"""

import os
import argparse
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import datasets, model_selection
# import our VGG-16 Model
from model3 import VGGModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
To download the weights, run this line in the gcp terminal
within the virtual environment:
    $ gdown https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo
"""

def main():
    """ Main function. """
    # Load train/test data (Olivetti Faces) from sklearn 
    olivetti = datasets.fetch_olivetti_faces()
    #Split olivetti object into images and labels
    images = olivetti.images 
    labels = olivetti.target

    # Resize each image in Olivetti dataset to (224,224) as that is the size expected by VGG
    resized_oli = np.zeros((images.shape[0],224,224,3))
    for i in range(len(resized_oli)):
        img = cv2.resize(images[i], (224,224))
        img = np.stack([img, img, img], axis=-1)
        resized_oli[i] = img

    # Load model and weights
    model = VGGModel()
    model(tf.keras.Input(shape=(224, 224, 3)))
    model.load_weights('vgg_face_weights.h5', by_name=True)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    # Compute predictions on the resized Olivetti dataset
    prediction = model.predict(resized_oli)
    
    # Varying the number of images per person in the train-test split
    for i in range(1,10):
        r = 5 # random state
        test_size = 1-i/10.0 
        # Split our prediction into our training and testing sets
        train_ims, test_ims, train_labs, test_labs = model_selection.train_test_split(prediction.reshape(400,-1) 
            , labels, test_size=test_size, random_state=r, stratify=labels)

        # Fit the model to the training set and labels using KNearestNeighbors
        # Compute the mean accuracy on the test data and labels
        knn = KNeighborsClassifier()
        knn.fit(train_ims, train_labs)
        acc = knn.score(test_ims,test_labs)

        print("accuracy for ", i, " images and random state ", r, " = ", acc)

main()