## Semantic Segmentation Practice 1
nuclei Dataset : https://www.kaggle.com/c/data-science-bowl-2018/data
### How to Setup and Train
1. clone repository and download dataset on clone directory
```
git clone https://github.com/realJun9u/unet_nuclei.git
```
2. Setup Dataset (Extract and Prepare Train,Test Data)
```
python setup.py
```
3. Train Model
```
python train.py
```
4. Analyze Result
```
tensorboard --log_dir=logs --host={host} --port={port}
```
