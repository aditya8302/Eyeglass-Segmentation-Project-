import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from train import build_model

global image_h
global image_w


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print("path;",path)    

def load_dataset(path, split=0.1):
    """ Loading the images and masks """
    main_directory = path

    X = []
    Y = []
    files_path = os.path.join(main_directory, "files")
    mask_path= os.path.join(main_directory, "mask")
    
    for root, dirs, files in os.walk(files_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                X.append(image_path)
                
    for root, dirs, files in os.walk(files_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                Y.append(image_path)
    print("checkxy")

    """ Spliting the data into training and testing """
    split_size = float(len(X) * split)
    print("split_size", split_size)
    print("check X",X)
    print("check Y",Y)

    train_x, valid_x = train_test_split(X, test_size=split, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split, random_state=42)
    print("checkix")

    return (train_x, train_y), (valid_x, valid_y)
    

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    print("checklm")
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_w, image_h))
    x = x.astype(np.float32)    ## (h, w)
    x = np.expand_dims(x, axis=-1)  ## (h, w, 1)
    x = np.concatenate([x, x, x, x], axis=-1) ## (h, w, 4)
    print("checkpq")
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        print("checkb")
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([image_h, image_w, 3])
    y.set_shape([image_h, image_w, 4])
    print("checks")
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    print("checkj")
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    image_h = 512
    image_w = 512
    input_shape = (image_h, image_w, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 20

    """ Paths """
    dataset_path = "D:\ML_Project"
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_model(input_shape)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs, 
        callbacks=callbacks
    )
    #Perform Predictions
    y_pred = model.predict(valid_x)

    # 4. Evaluate Performance
    accuracy = accuracy_score(valid_y, y_pred)
    print("Accuracy:", accuracy)
