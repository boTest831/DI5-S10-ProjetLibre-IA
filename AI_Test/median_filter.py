from scipy.ndimage import median_filter
import numpy as np

def data_modify_suitable_train(data_set = None, type=True):
    if data_set is not None:
        data = []
        if type is True:
            np.random.shuffle(data_set)
            data = data_set[:, 0: data_set.shape[1] - 1]
        else:
            data = data_set
    data = np.array([np.reshape(i, (40,40)) for i in data])
    data = np.array([median_filter(i, size=(3, 3)) for i in data])
    data = np.array([(i>10) * 100 for i in data])
    data = np.array([np.reshape(i, (i.shape[0], i.shape[1],1)) for i in data])
    return data