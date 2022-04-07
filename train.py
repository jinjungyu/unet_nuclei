import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPool2D,concatenate,Dropout,Lambda

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATASET = os.path.join(ROOT_DIR,'dataset')

X_TRAIN_PATH=os.path.join(PATH_DATASET,'x_train.npy')
Y_TRAIN_PATH=os.path.join(PATH_DATASET,'y_train.npy')
X_TEST_PATH=os.path.join(PATH_DATASET,'x_test.npy')


# Preparing Data
X_train = np.load(X_TRAIN_PATH)
Y_train = np.load(Y_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)

# Build Model (U-Net)
img_width = 128
img_height = 128
img_channels = 3

drop_rate=0.2
##### Encoder ######
inputs = Input((img_width,img_height,img_channels))
scaled_inputs = Lambda(lambda x: x / 255)(inputs)
# Block 1 input size : 128 128 3
conv1 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block1_conv1")(scaled_inputs)
conv1 = Dropout(drop_rate)(conv1)
conv1 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block1_conv2")(conv1)
pool1 = MaxPool2D((2,2),name="block1_poll1")(conv1)
# Block 2 input size : 64 64 16
conv2 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block2_conv1")(pool1)
conv2 = Dropout(drop_rate)(conv2)
conv2 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block2_conv2")(conv2)
pool2 = MaxPool2D((2,2),name="block2_poll1")(conv2)
# Block 3 input size : 32 32 32
conv3 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block3_conv1")(pool2)
conv3 = Dropout(drop_rate)(conv3)
conv3 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block3_conv2")(conv3)
pool3 = MaxPool2D((2,2),name="block3_poll1")(conv3)
# Block 4 input size : 16 16 64
conv4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block4_conv1")(pool3)
conv4 = Dropout(drop_rate)(conv4)
conv4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block4_conv2")(conv4)
pool4 = MaxPool2D((2,2),name="block4_poll1")(conv4)
# Block 5 input size : 8 8 128
conv5 = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block5_conv1")(pool4)
conv5 = Dropout(drop_rate)(conv5)
conv5 = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block5_conv2")(conv5)

##### Decoder ######
# Block 6 input size : 8 8 256
convt6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same',name="block6_upsample1")(conv5) # 8 8 256 -> 16 16 128
concat6 = concatenate([conv4,convt6]) # 16 16 128 + 16 16 128 = 16 16 256
conv6 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block6_conv1")(concat6)
conv6 = Dropout(drop_rate)(conv6)
conv6 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block6_conv2")(conv6)
# Block 7 input size : 16 16 128
convt7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same',name="block7_upsample1")(conv6) # 16 16 128 -> 32 32 64
concat7 = concatenate([conv3,convt7]) # 32 32 64 + 32 32 64 = 32 32 128
conv7 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block7_conv1")(concat7)
conv7 = Dropout(drop_rate)(conv7)
conv7 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block7_conv2")(conv7)
# Block 8 input size : 32 32 64
convt8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same',name="block8_upsample1")(conv7) # 32 32 64 -> 64 64 32
concat8 = concatenate([conv2,convt8]) # 64 64 32 + 64 64 32 = 64 64 64
conv8 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block8_conv1")(concat8)
conv8 = Dropout(drop_rate)(conv8)
conv8 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block8_conv2")(conv8)
# Block 9 input size : 64 64 32
convt9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same',name="block9_upsample1")(conv8) # 64 64 32 -> 128 128 16
concat9 = concatenate([conv1,convt9]) # 128 128 16 + 128 128 16 = 128 128 32
conv9 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block9_conv1")(concat9)
conv9 = Dropout(drop_rate)(conv9)
conv9 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same',name="block9_conv2")(conv9)

outputs = Conv2D(1,(1,1),activation='sigmoid',name="output")(conv9) # 128 128 1

model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.summary()
####################

model_best_path = os.path.join(ROOT_DIR,'unet_nuclei_best.h5')
# Set Callback, Checkpoint
callbacks = [
             tf.keras.callbacks.ModelCheckpoint(model_best_path,monitor='val_loss',save_best_only=True),
             tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir=os.path.join(ROOT_DIR,'logs'))
]
# Train model
history = model.fit(X_train,Y_train,validation_split=0.1,batch_size=16,epochs=50,callbacks=callbacks)

model.load_weights(model_best_path)
# Evaludate model
preds_train = model.predict(X_train)
preds_train_mask = (preds_train>0.5).astype(np.uint8)
preds_test = model.predict(X_test)
preds_test_mask = (preds_test>0.5).astype(np.uint8)

# Train Prediction and Ground Truth
indices = np.random.randint(0,X_train.shape[0],10)
for idx in indices:
  plt.subplot(121)
  plt.title("Train Image")
  imshow(X_train[idx])
  plt.subplot(122)
  plt.title("Ground Truth")
  imshow(np.squeeze(preds_train_mask[idx]),cmap="gray")
  plt.show()

# Visualize Test Prediction
indices = np.random.randint(0,X_test.shape[0],10)
for idx in indices:
  plt.subplot(121)
  plt.title("Test Image")
  imshow(X_test[idx])
  plt.subplot(122)
  plt.title("Test Prediction")
  imshow(np.squeeze(preds_test_mask[idx]),cmap="gray")
  plt.show()
