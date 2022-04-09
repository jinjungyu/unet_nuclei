from tensorflow.utils import Sequence
import numpy as np

class DataLoader(Sequence):
    def __init__(self,data_dir,batch_size,shuffle=True):
        self.x, self,y = X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self): # Number of Batch
        return np.ceil(len(self.x)/self.batch_size)

    def __getitem__(self,idx): # Generate Batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        x_batch = np.array([self.x[i] for i in indices])
        y_batch = np.array([self.y[i] for i in indices])

        return x_batch, y_batch

    def on_epoch_end(self): # if self.shuffle == True, Shuffle per Epoch
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)