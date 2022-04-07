# 데이터 셋 : https://www.kaggle.com/c/data-science-bowl-2018/data
import zipfile
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread,imshow
from skimage.transform import resize

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path_zipfile = os.path.join(ROOT_DIR,'data-science-bowl-2018.zip')
PATH_DATASET = os.path.join(ROOT_DIR,'dataset')

# 압축 파일 풀기
print("\nExtract data-science-bowl-2018.zip.....",end="")
with zipfile.ZipFile(path_zipfile, 'r') as zip_ref:
  zip_ref.extractall(PATH_DATASET)
print("Done!\n")

for file in ['stage1_train.zip','stage1_test.zip']:
  path_dst = os.path.join(PATH_DATASET,file.split('.')[0])
  path_zipfile = os.path.join(PATH_DATASET,file)
  with zipfile.ZipFile(path_zipfile, 'r') as zip_ref:
    print(f"Extract {path_zipfile}.....",end="")
    zip_ref.extractall(path_dst)
    print("Done!")
print()

TRAIN_DIR = os.path.join(PATH_DATASET,'stage1_train')
TEST_DIR = os.path.join(PATH_DATASET,'stage1_test')

# Preparing Data
img_width = 128
img_height = 128
img_channels = 3

train_folders = next(os.walk(TRAIN_DIR))[1] # stage1_train 내의 folder list
test_folders = next(os.walk(TEST_DIR))[1] # stage1_test 내의 folder list

X_train = np.zeros((len(train_folders),img_height,img_width,img_channels),dtype=np.uint8)
Y_train = np.zeros((len(train_folders),img_height,img_width,1),dtype=np.bool_)

print("Prepare Train Image and Ground Truth")
for i, data_id in tqdm.tqdm(enumerate(train_folders),total=len(train_folders)):
  img_path = os.path.join(TRAIN_DIR,data_id,'images',data_id+'.png')
  img = imread(img_path)[:,:,:img_channels]
  img = resize(img,(img_height,img_width), mode='constant', preserve_range=True)
  X_train[i] = img
  mask = np.zeros((img_height,img_width,1),dtype=np.bool_)
  mask_path = os.path.join(TRAIN_DIR,data_id,'masks')
  for mask_file in next(os.walk(mask_path))[-1]:
    sample = imread(os.path.join(mask_path,mask_file))
    sample = resize(sample,(img_height,img_width), mode='constant', preserve_range=True)
    sample = np.expand_dims(sample,axis=-1).astype(np.bool_)
    mask = np.maximum(mask,sample)
  Y_train[i] = mask

print("Prepare Test Image")
X_test = np.zeros((len(test_folders),img_height,img_width,img_channels),dtype=np.uint8)
for i, data_id in tqdm.tqdm(enumerate(test_folders),total=len(test_folders)):
  img_path = os.path.join(TEST_DIR,data_id,'images',data_id+'.png')
  img = imread(img_path)[:,:,:img_channels]
  img = resize(img,(img_height,img_width), mode='constant', preserve_range=True)
  X_test[i] = img

# k = np.random.randint(0,i)
# plt.subplot(121)
# imshow(X_train[k])
# plt.subplot(122)
# imshow(np.squeeze(Y_train[k].astype(np.uint8)),cmap='gray')
# plt.show()

X_TRAIN_PATH=os.path.join(PATH_DATASET,'x_train.npy')
Y_TRAIN_PATH=os.path.join(PATH_DATASET,'y_train.npy')
X_TEST_PATH=os.path.join(PATH_DATASET,'x_test.npy')
np.save(X_TRAIN_PATH, X_train)
print(f"X_train is saved at {X_TRAIN_PATH}.")
np.save(Y_TRAIN_PATH, Y_train)
print(f"Y_train is saved at {Y_TRAIN_PATH}.")
np.save(X_TEST_PATH, X_test)
print(f"X_test is saved at {X_TEST_PATH}.")
