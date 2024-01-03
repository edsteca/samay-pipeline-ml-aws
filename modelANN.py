# Ejemplo adaptado de https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/scikit_learn_inference_pipeline

from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Import TensorFlow and necessary modules for building a neural network
import tensorflow as tf
from tensorflow.keras import layers, models

# Import specific components from TensorFlow Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Import necessary libraries for callbacks and evaluation metrics
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    
    data = pd.read_csv(os.path.join(args.train,'data_to_train.csv'), index_col=0, engine="python",sep=';')
    sound_df = data
    
    X = np.array(data.drop(['Fracaso_o_exito'],1))
    y = np.array(data['Fracaso_o_exito'])
    
    print('Shape : ', X.shape)

    # Initialize the LabelEncoder
    le = LabelEncoder()
    
    # Fit the label encoder to the 'label' column and return the encoded labels
    encoded_labels = le.fit_transform(sound_df['label'])

    # Split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )
    # Calculate the number of features in the training data
    n_features = X_train.shape[1]
    
    # Calculate the number of unique labels in the training data
    n_classes = np.unique(y_train).shape[0]
    # Calculate the number of features in the training data
    n_features = X_train.shape[1]
    
    # Calculate the number of unique labels in the training data
    n_classes = np.unique(y_train).shape[0]

    # Print the number of features and the number of unique labels in the training data
    print("Number of features in training data:", n_features)
    print("Number of unique labels in training data:", n_classes)

    # Create the Sequential model
    modelANN = Sequential()
    
    # Add the first hidden layer with 128 units and ReLU activation
    modelANN.add(Dense(128, activation='relu', input_shape=(n_features,)))
    
    # Apply dropout regularization to the first hidden layer
    modelANN.add(Dropout(0.5))
    
    # Add the second hidden layer with 64 units and ReLU activation
    modelANN.add(Dense(64, activation='relu'))
    
    # Apply dropout regularization to the second hidden layer
    modelANN.add(Dropout(0.5))
    
    # Add the output layer with 'n_classes' units and softmax activation
    modelANN.add(Dense(n_classes, activation='softmax'))

    # Compile the Sequential neural network model
    modelANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(modelANN.summary())

    # Set the number of training epochs and batch size
    num_epochs = 50  # Number of training epochs
    num_batch_size = 32  # Batch size for training

    # Define callbacks for model training, including checkpointing
    callbacks = [
        ModelCheckpoint(
            filepath='modelannsamaytest_{epoch:02d}.h5',  # Filepath pattern for saving models
            monitor='val_accuracy',  # Monitor validation accuracy for saving
            save_best_only=True,  # Only save when there's an improvement
            verbose=1  # Print a message when saving
        )
    ]

    # Fit (train) the neural network model on the training data
    start = datetime.now()
    modelANN.fit(
        X_train,  # Training features
        y_train,  # Training labels
        batch_size=num_batch_size,  # Batch size
        epochs=num_epochs,  # Number of epochs
        validation_data=(X_test, y_test),  # Validation dataset
        callbacks=callbacks,  # Callbacks for model saving and monitoring
        verbose=1  # Display progress updates
    )
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    
    #Save the model to the location specified by args.model_dir
    joblib.dump(modelANN, os.path.join(args.model_dir, "model.joblib"))
    
    print("Model saved!")
    
    
"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to 
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are 
only going to accept text/csv and raise an error for all other formats.
"""    

    
def input_fn(request_body, request_content_type):
    if content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")

        
def output_fn(prediction, content_type):
    return '|'.join([INDEX_TO_LABEL[t] for t in prediction])


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a pca
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.predict(input_data)
    
    return features


def model_fn(model_dir):
    """Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
