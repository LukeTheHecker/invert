import numpy as np
from scipy.sparse.csgraph import laplacian
import mne
from copy import deepcopy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sys; sys.path.insert(0, '../../esinet')

from esinet.util import unpack_fwd

def contextualize(stc_instant, leadfield, lstm_look_back=80, 
                num_units=128, num_epochs=100, steps_per_ep=None, 
                batch_size=32, optimizer="adam", loss="mean_squared_error",
                verbose=0):

    # leadfield = unpack_fwd(fwd)[1]
    _, n_dipoles = leadfield.shape
    # stc_instant_unscaled = deepcopy(stc_instant)
    stc_instant = standardize_2(stc_instant)
    stc_epochs_train = deepcopy(stc_instant)[np.newaxis]
    x_train, y_train = prepare_training_data(stc_epochs_train, lstm_look_back)
    # time axis must be second-to-last
    x_train = np.swapaxes(x_train, 1,2)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),]

    model = Sequential()
    model.add(LSTM(num_units, activation='tanh', return_sequences=False, input_shape=(lstm_look_back, n_dipoles)))
    model.add(Dense(n_dipoles, activation='linear'))

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.losses.CosineSimilarity()])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, 
                                steps_per_epoch=steps_per_ep, validation_split=0.15, 
                                shuffle=True, epochs=num_epochs, callbacks=callbacks)

    stc_lstm =  np.zeros(stc_instant.shape)
    stc_cmne =  np.zeros(stc_instant.shape)

    stc_lstm[:, :lstm_look_back] = stc_instant[:, :lstm_look_back]
    stc_cmne[:, :lstm_look_back] = stc_instant[:, :lstm_look_back]

    steps = stc_instant.shape[1] - lstm_look_back

    for i in range(steps):
        print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2))
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm[:, i+lstm_look_back] = stc_pred
        stc_cmne[:, i+lstm_look_back] = stc_instant[:, i+lstm_look_back] * stc_pred

    return stc_cmne

def contextualize_bd(stc_instant, leadfield, lstm_look_back=80, 
                num_units=128, num_epochs=100, steps_per_ep=None, 
                batch_size=32, optimizer="adam", loss="mean_squared_error",
                verbose=0):

    # leadfield = unpack_fwd(fwd)[1]
    _, n_dipoles = leadfield.shape
    # stc_instant_unscaled = deepcopy(stc_instant)
    stc_instant_forward = standardize_2(stc_instant)
    stc_instant_backwards = stc_instant_forward[:, ::-1]

    stc_epochs_train_forward = deepcopy(stc_instant_forward)[np.newaxis]
    stc_epochs_train_backwards = deepcopy(stc_epochs_train_forward[:, :, ::-1])
    stc_epochs_train = np.concatenate([stc_epochs_train_forward, stc_epochs_train_backwards], axis=0)

    x_train, y_train = prepare_training_data(stc_epochs_train, lstm_look_back)
    # time axis must be second-to-last
    x_train = np.swapaxes(x_train, 1,2)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),]

    model = Sequential()
    model.add(LSTM(num_units, activation='tanh', return_sequences=False, input_shape=(lstm_look_back, n_dipoles)))
    model.add(Dense(n_dipoles, activation='linear'))

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.losses.CosineSimilarity()])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, 
                                steps_per_epoch=steps_per_ep, validation_split=0.15, 
                                shuffle=True, epochs=num_epochs, callbacks=callbacks)

    stc_lstm_forward =  np.zeros(stc_instant_forward.shape)
    stc_cmne_forward =  np.zeros(stc_instant_forward.shape)

    stc_lstm_forward[:, :lstm_look_back] = stc_instant_forward[:, :lstm_look_back]
    stc_cmne_forward[:, :lstm_look_back] = stc_instant_forward[:, :lstm_look_back]

    steps = stc_instant_forward.shape[1] - lstm_look_back
    print("Forward Steps:")
    for i in range(steps):
        print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne_forward[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2))
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm_forward[:, i+lstm_look_back] = stc_pred
        stc_cmne_forward[:, i+lstm_look_back] = stc_instant_forward[:, i+lstm_look_back] * stc_pred

    stc_lstm_backwards =  np.zeros(stc_instant_backwards.shape)
    stc_cmne_backwards =  np.zeros(stc_instant_backwards.shape)

    stc_lstm_backwards[:, :lstm_look_back] = stc_instant_backwards[:, :lstm_look_back]
    stc_cmne_backwards[:, :lstm_look_back] = stc_instant_backwards[:, :lstm_look_back]

    steps = stc_instant_backwards.shape[1] - lstm_look_back
    print("Forward Steps:")
    for i in range(steps):
        print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne_backwards[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2))
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm_backwards[:, i+lstm_look_back] = stc_pred
        stc_cmne_backwards[:, i+lstm_look_back] = stc_instant_backwards[:, i+lstm_look_back] * stc_pred

    stc_cmne_backwards_reverse = stc_cmne_backwards[:, ::-1]
    stc_cmne = deepcopy(stc_cmne_forward)
    stc_cmne[:, :lstm_look_back] = stc_cmne_backwards_reverse[:, :lstm_look_back]

    return stc_cmne

def inverse_dspm(M, leadfield):
    alpha = 0.001
    n_chans, n_dipoles = leadfield.shape
    noise_cov = np.identity(n_chans) + np.random.rand(n_chans) * 0.01
    source_cov = np.identity(n_dipoles)

    M_norm = (1/np.sqrt(noise_cov)) @ M
    G_norm = (1/np.sqrt(noise_cov)) @ leadfield

    K = source_cov @ G_norm.T @ np.linalg.inv(G_norm @ source_cov @ G_norm.T + alpha**2 * np.identity(n_chans))
    W_dSPM = np.diag(np.sqrt(1/np.diagonal(K @ noise_cov @ K.T)))
    K_dSPM = W_dSPM @ K
    D_dSPM = K_dSPM @ M_norm

    # rectify & normalize
    # Q = np.stack([rectify_norm(x) for x in D_dSPM.T], axis=1)
    return D_dSPM

def inverse_loreta(M, leadfield, fwd):
    alpha = 0.001
    adjacency = mne.spatial_src_adjacency(fwd['src']).toarray()
    B = np.diag(np.linalg.norm(leadfield, axis=0))
    laplace_operator = laplacian(adjacency)
    D_LOR = np.linalg.inv(leadfield.T @ leadfield + alpha * B @ laplace_operator.T @ laplace_operator @ B) @ leadfield.T @ M
    return D_LOR

def rectify_norm(x):
    return (x-abs(x).mean()) / abs(x).std()

def prepare_training_data(stc, lstm_look_back=20):
    assert len(stc.shape) == 3, "stc must be 3D numpy.ndarray"
    n_samples, _, n_time = stc.shape
    x = []
    y = []
    for i in range(n_samples):
        for j in range(n_time-lstm_look_back-1):
            x.append( stc[i, :, j:j+lstm_look_back] )
            y.append( stc[i, :, j+lstm_look_back+1] )
            
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    return x, y

def standardize(mat, mean=None, std=None):
    """
    0 center and scale data
    Standardize an np.array to the array mean and standard deviation or specified parameters
    See https://en.wikipedia.org/wiki/Feature_scaling
    """
    if mean is None:
        mean = np.mean(mat, axis=1)

    if std is None:
        std = np.std(mat, axis=1)
    print(mean, std)

    # data_normalized = (mat.transpose() - mean).transpose()
    # data = (data_normalized.transpose() / std).transpose()
    # return data
    return np.transpose((mat.T - mean) /std)

def standardize_2(mat):
    mat_scaled = deepcopy(mat)
    for t, slice in enumerate(mat_scaled.T):
        # mat_scaled[:, t] = (slice - slice.mean()) / slice.std()
        mat_scaled[:, t] = slice / abs(slice).max()
    return mat_scaled

