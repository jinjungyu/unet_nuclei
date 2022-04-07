import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATASET = os.path.join(ROOT_DIR,'dataset')

X_TRAIN_PATH=os.path.join(PATH_DATASET,'x_train.npy')
Y_TRAIN_PATH=os.path.join(PATH_DATASET,'y_train.npy')
X_TEST_PATH=os.path.join(PATH_DATASET,'x_test.npy')

X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)

model_path = os.path.join(ROOT_DIR,'model_unet_nuclei.h5')

model = tf.keras.models.load_model(model_path)
# Evaludate model
preds_train = model.predict(X_train)
preds_train_mask = (preds_train>0.5).astype(np.uint8)
preds_test = model.predict(X_test)
preds_test_mask = (preds_test>0.5).astype(np.uint8)

# Train Prediction and Ground Truth
indices = np.random.randint(0,X_train.shape[0],5)
for idx in indices:
    plt.figure(figsize=(9,3))
    plt.subplot(131)
    plt.title("Train Image")
    imshow(X_train[idx])
    plt.subplot(132)
    plt.title("Train Prediction")
    imshow(np.squeeze(preds_train_mask[idx]),cmap="gray")
    plt.subplot(133)
    plt.title("Ground Truth")
    imshow(Y_train[idx].astype(np.uint8),cmap="gray")
    plt.tight_layout()
    plt.show()

# Visualize Test Prediction
indices = np.random.randint(0,X_test.shape[0],5)
for idx in indices:
    plt.subplot(121)
    plt.title("Test Image")
    imshow(X_test[idx])
    plt.subplot(122)
    plt.title("Test Prediction")
    imshow(np.squeeze(preds_test_mask[idx]),cmap="gray")
    plt.tight_layout()
    plt.show()
