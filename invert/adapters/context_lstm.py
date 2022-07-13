import numpy as np
from copy import deepcopy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def contextualize(stc_instant, forward, lstm_look_back=80, 
                num_units=128, num_epochs=100, steps_per_ep=25, 
                batch_size=32, fast=True, optimizer="adam", loss="mean_squared_error",
                verbose=0):
    """
    Temporal contextualization of inverse solutions using Long-Short Term Memory
    (LSTM) Networks as described in [1].

    Parameters
    ----------
    stc_instant : mne.sourceEstimate
        The instantaneous source estimate object which shall be contextualized.
    forward : mne.Forward
        The forward model
    lstm_look_back : int
        Number of time points to consider as context
    num_units : int
        Number of LSTM cells (units)
    num_epochs : int
        Number of epochs to train the LSTM network.
    steps_per_ep : int
        Iterations per epoch during optimization.
    batch_size : int
        Batch size for training.
    optimizer : str/ tensorflow keras optimizer object
        The optimizer for gradient descent
    loss : str/ tensorflow keras loss function
        Computes the error between predicted source and true source for
        backpropagation.
    verbose : int
        Controls verbosity of the program
    
    Return
    ------
    stc_context : mne.SourceEstimate
        The contextualized source.

    References
    ----------
    
    [1] Dinh, C., Samuelsson, J. G., Hunold, A., Hämäläinen, M. S., & Khan, S.
        (2021). Contextual MEG and EEG source estimates using spatiotemporal
        LSTM networks. Frontiers in neuroscience, 15, 552666.
    """
    leadfield = forward['sol']['data']
    _, n_dipoles = leadfield.shape
    stc_instant_unscaled = deepcopy(stc_instant.data)
    stc_instant_scaled = standardize_2(stc_instant.data)
    stc_epochs_train = deepcopy(stc_instant_scaled)[np.newaxis]
    x_train, y_train = prepare_training_data(stc_epochs_train, lstm_look_back)
    # time axis must be second-to-last
    x_train = np.swapaxes(x_train, 1,2)

    if fast:
        num_epochs = 50
        num_units = 64
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_cosine_similarity", min_delta=0.01),]
    else:
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),]

    model = Sequential()
    model.add(LSTM(num_units, activation='tanh', return_sequences=False, input_shape=(lstm_look_back, n_dipoles)))
    model.add(Dense(n_dipoles, activation='linear'))

        # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.losses.CosineSimilarity()])
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, 
                                steps_per_epoch=steps_per_ep, validation_split=0.15, 
                                shuffle=True, epochs=num_epochs, callbacks=callbacks, 
                                verbose=verbose)

    stc_lstm =  np.zeros(stc_instant_scaled.shape)
    stc_cmne =  np.zeros(stc_instant_scaled.shape)

    stc_lstm[:, :lstm_look_back] = np.ones((n_dipoles, lstm_look_back))
    stc_cmne[:, :lstm_look_back] = stc_instant_scaled[:, :lstm_look_back]

    steps = stc_instant_scaled.shape[1] - lstm_look_back

    for i in range(steps):
        # print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2), verbose=0)
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm[:, i+lstm_look_back] = stc_pred
        stc_cmne[:, i+lstm_look_back] = stc_instant_scaled[:, i+lstm_look_back] * stc_pred

    stc_context_data = stc_instant_unscaled * stc_lstm
    stc_context = stc_instant.copy()
    stc_context.data = stc_context_data

    return stc_context

def contextualize_bd(stc_instant, forward, lstm_look_back=80, 
                num_units=128, num_epochs=100, steps_per_ep=25, 
                batch_size=32, fast=True, optimizer="adam", 
                loss="mean_squared_error", verbose=0):
    """
    Bi-directional temporal contextualization of inverse solutions using
    Long-Short Term Memory (LSTM) Networks as described in [1] using both past
    and future time points.

    Parameters
    ----------
    stc_instant : mne.sourceEstimate
        The instantaneous source estimate object which shall be contextualized.
    forward : mne.Forward
        The forward model
    lstm_look_back : int
        Number of time points to consider as context
    num_units : int
        Number of LSTM cells (units)
    num_epochs : int
        Number of epochs to train the LSTM network.
    steps_per_ep : int
        Iterations per epoch during optimization.
    batch_size : int
        Batch size for training.
    optimizer : str/ tensorflow keras optimizer object
        The optimizer for gradient descent
    loss : str/ tensorflow keras loss function
        Computes the error between predicted source and true source for
        backpropagation.
    verbose : int
        Controls verbosity of the program
    
    Return
    ------
    stc_context : mne.SourceEstimate
        The contextualized source.

    References
    ----------
    
    [1] Dinh, C., Samuelsson, J. G., Hunold, A., Hämäläinen, M. S., & Khan, S.
        (2021). Contextual MEG and EEG source estimates using spatiotemporal
        LSTM networks. Frontiers in neuroscience, 15, 552666.
    """
    leadfield = forward['sol']['data']
    _, n_dipoles = leadfield.shape
    
    stc_instant_forward = standardize_2(stc_instant.data)
    stc_instant_backwards = stc_instant_forward[:, ::-1]

    stc_epochs_train_forward = deepcopy(stc_instant_forward)[np.newaxis]
    stc_epochs_train_backwards = deepcopy(stc_epochs_train_forward[:, :, ::-1])
    stc_epochs_train = np.concatenate([stc_epochs_train_forward, stc_epochs_train_backwards], axis=0)

    x_train, y_train = prepare_training_data(stc_epochs_train, lstm_look_back)
    # time axis must be second-to-last
    x_train = np.swapaxes(x_train, 1,2)

    if fast:
        num_epochs = 50
        num_units = 64
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_cosine_similarity", min_delta=0.01),]
    else:
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),]

    model = Sequential(name="Contextual_LSTM")
    model.add(LSTM(num_units, activation='tanh', return_sequences=False, input_shape=(lstm_look_back, n_dipoles)))
    model.add(Dense(n_dipoles, activation='linear'))

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.losses.CosineSimilarity()])
    if verbose>0:
        model.summary()
    # print(x_train.shape, y_train.shape)
    model.fit(x_train, y_train, batch_size=batch_size, 
                                steps_per_epoch=steps_per_ep, validation_split=0.15, 
                                shuffle=True, epochs=num_epochs, callbacks=callbacks,
                                verbose=verbose)

    stc_lstm_forward =  np.zeros(stc_instant_forward.shape)
    stc_cmne_forward =  np.zeros(stc_instant_forward.shape)

    stc_lstm_forward[:, :lstm_look_back] = np.ones((n_dipoles, lstm_look_back))
    stc_cmne_forward[:, :lstm_look_back] = stc_instant_forward[:, :lstm_look_back]

    steps = stc_instant_forward.shape[1] - lstm_look_back
    if verbose>0:
        print("Forward Steps:")
    for i in range(steps):
        # if verbose>0:
        #     print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne_forward[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2), verbose=0)
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm_forward[:, i+lstm_look_back] = stc_pred
        stc_cmne_forward[:, i+lstm_look_back] = stc_instant_forward[:, i+lstm_look_back] * stc_pred


    stc_lstm_backwards =  np.zeros(stc_instant_backwards.shape)
    stc_cmne_backwards =  np.zeros(stc_instant_backwards.shape)

    stc_lstm_backwards[:, :lstm_look_back] = np.ones((n_dipoles, lstm_look_back))
    stc_cmne_backwards[:, :lstm_look_back] = stc_instant_backwards[:, :lstm_look_back]

    steps = stc_instant_backwards.shape[1] - lstm_look_back
    if verbose>0:
        print("Backwards Steps:")
    for i in range(steps):
        # if verbose>0:
        #     print(f"Time Step {i}/{steps}")
        stc_prior = np.expand_dims(stc_cmne_backwards[:, i:i+lstm_look_back], axis=0)
        stc_pred = model.predict(np.swapaxes(stc_prior, 1,2), verbose=0)
        stc_pred = abs(stc_pred) / abs(stc_pred).max()
        stc_lstm_backwards[:, i+lstm_look_back] = stc_pred
        stc_cmne_backwards[:, i+lstm_look_back] = stc_instant_backwards[:, i+lstm_look_back] * stc_pred

    
    stc_lstm_combined = deepcopy(stc_lstm_forward)
    stc_lstm_backwards_rev = stc_lstm_backwards[:, ::-1]
    stc_lstm_combined[:, :lstm_look_back] = stc_lstm_backwards_rev[:, :lstm_look_back]


    # stc_cmne = deepcopy(stc_cmne_forward)
    # stc_cmne[:, :lstm_look_back] = stc_cmne_backwards_reverse[:, :lstm_look_back]
    stc_context_data = deepcopy(stc_instant.data)

    stc_context_data *= stc_lstm_combined
    stc_context = stc_instant.copy()
    stc_context.data = stc_context_data

    # if verbose>0:
    #     avg_attenuation = 100*(1 -stc_lstm_combined.mean())
    #     print(f"Temporal context attenuated {avg_attenuation:.1f} % of the source signal.")

    return stc_context


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
    # print(mean, std)

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

