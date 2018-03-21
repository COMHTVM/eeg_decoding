
import numpy as np
import h5py
import os
import math
from keras.utils import to_categorical
from librosa.core import stft


# Output shape (num_examples, 22, 1000, 1)
# No cropping
def get_data(personId, numTests):
    if os.path.exists("normal_xtrain.npy") and os.path.exists("normal_xtest.npy") and os.path.exists("normal_ytrain.npy") and os.path.exists("normal_ytest.npy"):
        X_tr = np.load("normal_xtrain.npy")
        X_te = np.load("normal_xtest.npy")
        y_tr = np.load("normal_ytrain.npy")
        y_te = np.load("normal_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 1000, 22, 1))
    X_te = np.zeros((0, 1000, 22, 1))
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        cur_y = cur_y[0, 0:X.shape[0]:1]
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X.transpose(0, 2, 1)
        X = X[:, :, :, np.newaxis]
        X_tr = np.concatenate((X_tr, X[idx[:-numTests]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("normal_xtrain.npy", X_tr)
    np.save("normal_xtest.npy", X_te)
    np.save("normal_ytrain.npy", y_tr)
    np.save("normal_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
# Output shape (num_examples, 22, 1000)
# No cropping    
def get_2d_data(personId, numTests):
    if os.path.exists("2d_xtrain.npy") and os.path.exists("2d_xtest.npy") and os.path.exists("2d_ytrain.npy") and os.path.exists("2d_ytest.npy"):
        X_tr = np.load("2d_xtrain.npy")
        X_te = np.load("2d_xtest.npy")
        y_tr = np.load("2d_ytrain.npy")
        y_te = np.load("2d_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 1000, 22))
    X_te = np.zeros((0, 1000, 22))
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        cur_y = cur_y[0, 0:X.shape[0]:1]
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X.transpose(0, 2, 1)
        X_tr = np.concatenate((X_tr, X[idx[:-numTests]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("2d_xtrain.npy", X_tr)
    np.save("2d_xtest.npy", X_te)
    np.save("2d_ytrain.npy", y_tr)
    np.save("2d_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
# Output shape (num_examples, 22, 500)
# No cropping    
def get_down2_2d_data(personId, numTests):
    if os.path.exists("down2_2dxtrain.npy") and os.path.exists("down2_2dxtest.npy") and os.path.exists("down2_2dytrain.npy") and os.path.exists("down2_2dytest.npy"):
        X_tr = np.load("down2_2dxtrain.npy")
        X_te = np.load("down2_2dxtest.npy")
        y_tr = np.load("down2_2dytrain.npy")
        y_te = np.load("down2_2dytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 500, 22))
    X_te = np.zeros((0, 500, 22))
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    down_factor = 2
    for idx, val in enumerate(personId):
        np.random.seed(val)
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        X = X[:, :, ::down_factor]
        cur_y = np.copy(A['type'])
        cur_y = cur_y[0, 0:X.shape[0]:1]
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X.transpose(0, 2, 1)
        X_tr = np.concatenate((X_tr, X[idx[:-numTests]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("down2_2dxtrain.npy", X_tr)
    np.save("down2_2dxtest.npy", X_te)
    np.save("down2_2dytrain.npy", y_tr)
    np.save("down2_2dytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
# Output shape (num_examples, 22, 500, 1)
# Cropping with frame_skip 100  
def get_cropped_data(personId, numTests):
    if os.path.exists("cropped_xtrain.npy") and os.path.exists("cropped_xtest.npy") and os.path.exists("cropped_ytrain.npy") and os.path.exists("cropped_ytest.npy"):
        X_tr = np.load("cropped_xtrain.npy")
        X_te = np.load("cropped_xtest.npy")
        y_tr = np.load("cropped_ytrain.npy")
        y_te = np.load("cropped_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 500, 22, 1)).astype(np.float32)
    X_te = np.zeros((0, 500, 22, 1)).astype(np.float32)
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        
        # Load matrix
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        num_trials = X.shape[0]
        cur_y = cur_y[0, 0:num_trials:1]
        
        # Split into frames 
        frame_skip = 100
        X, num_repeat = split_into_crops(X, frame_skip)
        cur_y = np.repeat(cur_y, num_repeat)
        
        # Remove NaNs
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        
        # Shuffle indices
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        # Concatenate matrices
        X = X[:, :, :, np.newaxis]
        X_tr = np.concatenate((X_tr, X[idx[:-numTests*num_repeat]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests*num_repeat:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests*num_repeat]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests*num_repeat:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("cropped_xtrain.npy", X_tr)
    np.save("cropped_xtest.npy", X_te)
    np.save("cropped_ytrain.npy", y_tr)
    np.save("cropped_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
# Output shape (num_examples, 22, 500, 1)
# Cropping with frame_skip 32    
def get_cropped2_data(personId, numTests):
    if os.path.exists("cropped2_xtrain.npy") and os.path.exists("cropped2_xtest.npy") and os.path.exists("cropped2_ytrain.npy") and os.path.exists("cropped2_ytest.npy"):
        X_tr = np.load("cropped2_xtrain.npy")
        X_te = np.load("cropped2_xtest.npy")
        y_tr = np.load("cropped2_ytrain.npy")
        y_te = np.load("cropped2_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 500, 22, 1)).astype(np.float32)
    X_te = np.zeros((0, 500, 22, 1)).astype(np.float32)
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        
        # Load matrix
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        num_trials = X.shape[0]
        cur_y = cur_y[0, 0:num_trials:1]
        
        # Split into frames 
        frame_skip = 32
        X, num_repeat = split_into_crops(X, frame_skip)
        cur_y = np.repeat(cur_y, num_repeat)
        
        # Remove NaNs
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        
        # Shuffle indices
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        # Concatenate matrices
        X = X[:, :, :, np.newaxis]
        X_tr = np.concatenate((X_tr, X[idx[:-numTests*num_repeat]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests*num_repeat:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests*num_repeat]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests*num_repeat:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("cropped2_xtrain.npy", X_tr)
    np.save("cropped2_xtest.npy", X_te)
    np.save("cropped2_ytrain.npy", y_tr)
    np.save("cropped2_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te

# Output shape(num_examples, num_fbanks * num_channels = 24*22)    
# No cropping
def get_feat_data(personId, numTests):
    if os.path.exists("feat_xtrain.npy") and os.path.exists("feat_xtest.npy") and os.path.exists("feat_ytrain.npy") and os.path.exists("feat_ytest.npy"):
        X_tr = np.load("feat_xtrain.npy")
        X_te = np.load("feat_xtest.npy")
        y_tr = np.load("feat_ytrain.npy")
        y_te = np.load("feat_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 513, 22)).astype(np.float32)
    X_te = np.zeros((0, 513, 22)).astype(np.float32)
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        
        # Load matrix
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        num_trials = X.shape[0]
        cur_y = cur_y[0, 0:num_trials:1]
        
        # Remove NaNs
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2])) 
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        
        # Get features
        X = crop_features(X)
        X = X.transpose(0, 2, 1)
        
        # Shuffle indices
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        # Concatenate matrices
        X_tr = np.concatenate((X_tr, X[idx[:-numTests]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("feat_xtrain.npy", X_tr)
    np.save("feat_xtest.npy", X_te)
    np.save("feat_ytrain.npy", y_tr)
    np.save("feat_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
    
def get_spec_data(personId, numTests):
    if os.path.exists("spec_xtrain.npy") and os.path.exists("spec_xtest.npy") and os.path.exists("spec_ytrain.npy") and os.path.exists("spec_ytest.npy"):
        X_tr = np.load("spec_xtrain.npy")
        X_te = np.load("spec_xtest.npy")
        y_tr = np.load("spec_ytrain.npy")
        y_te = np.load("spec_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 8, 257, 22)).astype(np.float32)
    X_te = np.zeros((0, 8, 257, 22)).astype(np.float32)
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        
        # Load matrix
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        num_trials = X.shape[0]
        cur_y = cur_y[0, 0:num_trials:1]
        
        # Remove NaNs
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2])) 
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        
        # Get features
        X = get_spectrogram_features(X)
        
        # Shuffle indices
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        # Concatenate matrices
        X_tr = np.concatenate((X_tr, X[idx[:-numTests]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("spec_xtrain.npy", X_tr)
    np.save("spec_xtest.npy", X_te)
    np.save("spec_ytrain.npy", y_tr)
    np.save("spec_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te
    
    
# Output shape (num_examples, 22, 500)
# Cropping with frame_skip 32    
def get_cropped2d_data(personId, numTests):
    if os.path.exists("cropped2d_xtrain.npy") and os.path.exists("cropped2d_xtest.npy") and os.path.exists("cropped2d_ytrain.npy") and os.path.exists("cropped2d_ytest.npy"):
        X_tr = np.load("cropped2d_xtrain.npy")
        X_te = np.load("cropped2d_xtest.npy")
        y_tr = np.load("cropped2d_ytrain.npy")
        y_te = np.load("cropped2d_ytest.npy")
        return X_tr, X_te, y_tr, y_te
    files = [os.path.join('project_datasets', file) for file in os.listdir('project_datasets/') if file.endswith('.mat')]
    X_tr = np.zeros((0, 500, 22)).astype(np.float32)
    X_te = np.zeros((0, 500, 22)).astype(np.float32)
    y_tr = np.zeros(0)
    y_te = np.zeros(0)
    for idx, val in enumerate(personId):
        np.random.seed(val)
        
        # Load matrix
        A = h5py.File(files[val], 'r')
        X = np.copy(A['image'])[:,:22, :]
        cur_y = np.copy(A['type'])
        num_trials = X.shape[0]
        cur_y = cur_y[0, 0:num_trials:1]
        
        # Split into frames 
        frame_skip = 32
        X, num_repeat = split_into_crops(X, frame_skip)
        cur_y = np.repeat(cur_y, num_repeat)
        
        # Remove NaNs
        xs = X.shape
        X = X.reshape(X.shape[0], -1)
        nan_rows = np.isnan(X).any(axis=1)
        X = X[~nan_rows]
        X = np.reshape(X, (-1, xs[1], xs[2]))
        y  = np.asarray(cur_y, dtype=np.int32)
        y = y[~nan_rows]
        
        # Shuffle indices
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        
        # Concatenate matrices
        X_tr = np.concatenate((X_tr, X[idx[:-numTests*num_repeat]]), axis=0)
        X_te = np.concatenate((X_te, X[idx[-numTests*num_repeat:]]), axis=0)
        y_tr = np.concatenate((y_tr, y[idx[:-numTests*num_repeat]]), axis=0)
        y_te = np.concatenate((y_te, y[idx[-numTests*num_repeat:]]), axis=0)
        
    print_dims(X_tr, X_te, y_tr, y_te)
    
    # Save matrices
    np.save("cropped2d_xtrain.npy", X_tr)
    np.save("cropped2d_xtest.npy", X_te)
    np.save("cropped2d_ytrain.npy", y_tr)
    np.save("cropped2d_ytest.npy", y_te)
    return X_tr, X_te, y_tr, y_te

    
    
# def split_into_crops(X, frame_skip):
    # frame_len = 500
    # signal_len = X.shape[2]
    # num_channels = X.shape[1]
    # num_frames = (signal_len - frame_len) // frame_skip
    # frames = np.zeros((0, frame_len, num_channels))
    # for t in np.arange(X.shape[0]):
        # for i in np.arange(0, signal_len - frame_len, frame_skip):
            # frames = np.concatenate((frames, X[t, :, i:i+frame_len].T[np.newaxis, :, :]), axis=0)
    # return frames, num_frames
    
def split_into_crops(X, frame_skip):
    frame_len = 500
    signal_len = X.shape[2]
    num_channels = X.shape[1]
    num_trials = X.shape[0]
    num_frames = math.ceil((signal_len - frame_len) / frame_skip )
    frames = np.zeros((num_frames * num_trials, frame_len, num_channels))
    for t in np.arange(num_trials):
        for i in np.arange(0, signal_len - frame_len, frame_skip):
            frames[t * num_frames + i//frame_skip] = X[t, :, i:i+frame_len].T[np.newaxis, :, :]
    return frames, num_frames
    
       
def crop_features(X):
    fs = 250
    num_trials = X.shape[0]
    num_channels = X.shape[1]
    signal_len = X.shape[2]
    n_fft = 1024
    X = np.abs(np.fft.fft(X, n=1024, axis=2))[:, :, :(1 + n_fft//2)]
    return X
    
def get_spectrogram_features(X):
    fs = 250
    num_trials = X.shape[0]
    num_channels = X.shape[1]
    signal_len = X.shape[2]
    frame_len = 2*fs
    frame_skip = frame_len//8
    n_fft = 512
    out_len = 1 + n_fft//2
    end_freq = 100
    end_freq_idx = end_freq*2
    num_frames = math.floor((signal_len - frame_len) / frame_skip )
    frames = np.zeros((num_trials, num_frames, out_len, num_channels))
    for t in np.arange(num_trials):
        for c in np.arange(num_channels):
            window = X[t, c]
            spec = np.abs(stft(window, n_fft=n_fft, win_length=frame_len, hop_length=frame_skip, center=False))
            frames[t, :, :, c] = spec.T
    return frames

    
def normalize(X):
    energy = np.sqrt(np.mean(X**2, axis=2))
    X /= energy[:, :, np.newaxis]
    return X
    
    
def print_dims(X_train, X_test, y_train, y_test):
    print("X_train size: ", X_train.shape)
    print("X_test size: ", X_test.shape)
    print("y_train size: ", y_train.shape)
    print("y_test size: ", y_test.shape)
    
def categorize_labels(y_train, y_test, num_classes):
    min_label_val = min(np.concatenate((y_train, y_test)))
    y_train = y_train - min_label_val
    y_test = y_test - min_label_val
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_test
    
