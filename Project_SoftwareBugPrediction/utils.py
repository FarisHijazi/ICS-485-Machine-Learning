import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt                        

from IPython.display import display, HTML

import torch
import torch.nn as nn
import torch.nn.functional as F

def disp_df(df):
    display(HTML(df.to_html()))
    return df

# dictionary but with attributes instead of key/value pairs (no more annoying strings)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_tensor(array, dtype=torch.FloatTensor):
    # convert numpy array to a float tensor
    return torch.tensor(array).type(dtype)



def build_loaders(data, batch_size=64, dtype=None):
    """
    :param data: 
    :returns a dictionary containins {'train', 'val', 'test'} each with a dataset
    """
    data = vars(data) # turn it to a dictionary
    loaders = {}
    
    for loader_name in ['train', 'val', 'test']:
        loaders[loader_name] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                to_tensor(data[f'{loader_name}x']), # loader_name will be {'train', 'val', 'test'}
                to_tensor(data[f'{loader_name}y'], dtype=dtype),
            ),
            batch_size=batch_size, shuffle=False
        )
    
    return loaders


def get_train_val_test_split(X, Y, TRAIN_PERC=0.7, VAL_PERC=0.15, TEST_PERC=0.15):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    
    
    np.random.seed(0)
    perm = np.random.permutation(X.shape[0])
    
    TRAIN_SIZE = int(TRAIN_PERC*X.shape[0])
    VAL_SIZE = int(VAL_PERC*X.shape[0])
    TEST_SIZE = int(TEST_PERC*X.shape[0])

    trainx = X[perm[0:TRAIN_SIZE],:]
    trainy = Y[perm[0:TRAIN_SIZE]]

    valx =   X[perm[TRAIN_SIZE:          TRAIN_SIZE+VAL_SIZE],:]
    valy =   Y[perm[TRAIN_SIZE:          TRAIN_SIZE+VAL_SIZE]]

    testx =  X[perm[TRAIN_SIZE+VAL_SIZE: TRAIN_SIZE+VAL_SIZE+TEST_SIZE],:]
    testy =  Y[perm[TRAIN_SIZE+VAL_SIZE: TRAIN_SIZE+VAL_SIZE+TEST_SIZE]]
    
    data_splits = AttrDict({
        'trainx': trainx,
        'trainy': trainy,
        
        'valx': valx,
        'valy': valy,

        'testx': testx,
        'testy': testy,
        
        'X': X,
        'Y': Y,
    })
    
    return data_splits



def get_sample_weights(Y):
    from collections import Counter
    c = Counter(Y.values.squeeze())
    
    # finding reciprical of frequency map for each Y value
    Y_weights = np.array([1.0/float(c[y]) for y in Y.values.T.squeeze()])
    
    return Y_weights / Y_weights.sum()


def get_class_weights(Y):
    from collections import Counter

    classes = np.unique(Y)
    c = Counter(Y.squeeze())
    
    # finding reciprical of frequency map for each Y value
    Y_weights = np.array([1.0/float(c[y]) for y in classes.squeeze()])
    
    return Y_weights / Y_weights.sum()

def undersample_df(df, target_label='bugs'):
    """
    returns a balanced dataframe with equal number of samples for each class
    this is done by understampling (taking the minimum frequency class)
    """
    df = df.sample(frac=1, random_state=4) # shuffle
    
    indeces, subframes = zip(*df.groupby(['bugs']))
    label_lengths = list(map(lambda df_:len(df_), subframes))
    
    smallest_class = np.min(label_lengths)
    
    return pd.concat([
        sdf.sample(n=smallest_class, axis=0) for sdf in subframes
    ]).reset_index().drop(['index'], axis=1)\
    .sample(frac=1, random_state=4) # shuffle again

