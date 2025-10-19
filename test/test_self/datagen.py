import h5py
import numpy as np
file = h5py.File('hdf5_sample.h5','w')
sample_data = np.random.randn(16,16,16,16).astype("float32")
dataset = file.create_dataset("dataset",(16,16,16,16), data=sample_data)
file.close()