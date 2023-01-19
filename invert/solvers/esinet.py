from copy import deepcopy
# from esinet import Simulation, Net
from .base import BaseSolver, InverseOperator
import colorednoise as cn
from scipy.sparse.csgraph import laplacian
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix, vstack
from time import time
import mne
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize_scalar
from tensorflow.keras.layers import (Conv1D, Conv2D, Dense, Flatten, Lambda, multiply,
                                    Reshape, AveragePooling2D, TimeDistributed,
                                    Bidirectional, LSTM)
import tensorflow.keras.backend as K
from ..util import find_corner
tf.keras.backend.set_image_data_format('channels_last')

class SolverCNN(BaseSolver):
    ''' Class for the Convolutional Neural Network (CNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="CNN", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_filters="auto", 
                            activation_function="tanh", batch_size="auto", 
                            n_timepoints=20, batch_repetitions=5, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            epsilon=0.25, snr_range=(1,100), patience=300,
                            add_forward_error=False, forward_error=0.1,
                            alpha="auto", **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        n_filters : int
            Number of filters in the convolution layer.
        activation_function : str
            The activation function of the hidden layers.
        batch_size : ["auto", int]
            The batch_size used during training. If "auto", the batch_size
            defaults to the number of dipoles in the source/ forward model.
            Choose a smaller batch_size (e.g., 1000) if you run into memory
            problems (RAM or GPU memory).
        n_timepoints : int
            The number of time points to simulate and ultimately train the
            neural network on.
        batch_repetitions : int
            The number of learning repetitions on the same batch of training
            data until a new batch is simulated.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        n_sources : int
            The maximum number of sources to simulate for the training data.
        n_orders : int
            Controls the maximum smoothness of the sources.
        size_validation_set : int
            The size of validation data set.
        epsilon : float
            The threshold at which to select sources as "active". 0.25 -> select
            all sources that are active at least 25 % of the maximum dipoles.
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data. (1,
            10) -> Samples have varying SNR from 1 to 10.
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape
        
        if batch_size == "auto":
            batch_size = n_dipoles
        if n_filters == "auto":
            n_filters = int(n_channels*4)
            
        # Store Parameters
        # Architecture
        self.n_filters = n_filters
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        self.add_forward_error = add_forward_error
        self.forward_error = forward_error
        # Inference
        self.epsilon = epsilon

        self.create_generator()
        self.build_model()
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data)
        stc = self.source_to_object(source_mat)

        return stc

    def apply_model(self, data) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''
        y = deepcopy(data)
        y -= y.mean(axis=0)
        n_channels, n_times = y.shape
        
        # Scaling
        y /= np.linalg.norm(y, axis=0)
        y /= np.max(abs(y))
        # Reshape for keras model
        y = y.T[np.newaxis, :, :, np.newaxis]
        
        # Add empty batch and (color-) channel dimension
        gammas = self.model.predict(y, verbose=self.verbose)[0]
        gammas /= gammas.max()


        # Select dipole indices
        gammas[gammas<self.epsilon] = 0
        dipole_idc = np.where(gammas!=0)[0]
        print("Active dipoles: ", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        y = deepcopy(data)
        y -= y.mean(axis=0)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        return x_hat        
            
    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=self.generator.__next__(), callbacks=callbacks)

    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape
        
        inputs = tf.keras.Input(shape=(None, n_channels, 1), name='Input')


        cnn1 = TimeDistributed(Conv1D(self.n_filters, n_channels,
                    activation=self.activation_function, padding="valid",
                    name='CNN1'))(inputs)
        
        # cnn1 = Conv2D(self.n_filters, (1, n_channels),
        #             activation=self.activation_function, padding="valid",
        #             name='CNN1')(inputs)
        reshape = Reshape((self.n_timepoints, self.n_filters))(cnn1)
        maxpool = Bidirectional(LSTM(128, return_sequences=False))(reshape)

        flat = Flatten()(maxpool)

        # hl1 = Dense(300, 
        #     activation=self.activation_function, 
        #     name='HL1')(flat)

        out = Dense(n_dipoles, 
            activation="relu", 
            name='Output')(flat)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        # if self.verbose > 0:
        model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=False, return_mask=True, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,)
        self.generator = generator(self.forward, **gen_args)

class SolverCovCNN(BaseSolver):
    ''' Class for the Covariance-based Convolutional Neural Network (CovCNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="Cov-CNN", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_filters="auto", 
                            activation_function="tanh", batch_size="auto", 
                            n_timepoints=20, batch_repetitions=10, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            epsilon=0.25, snr_range=(1,100), patience=100,
                            add_forward_error=False, forward_error=0.1,
                            alpha="auto", **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        n_filters : int
            Number of filters in the convolution layer.
        activation_function : str
            The activation function of the hidden layers.
        batch_size : ["auto", int]
            The batch_size used during training. If "auto", the batch_size
            defaults to the number of dipoles in the source/ forward model.
            Choose a smaller batch_size (e.g., 1000) if you run into memory
            problems (RAM or GPU memory).
        n_timepoints : int
            The number of time points to simulate and ultimately train the
            neural network on.
        batch_repetitions : int
            The number of learning repetitions on the same batch of training
            data until a new batch is simulated.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        n_sources : int
            The maximum number of sources to simulate for the training data.
        n_orders : int
            Controls the maximum smoothness of the sources.
        size_validation_set : int
            The size of validation data set.
        epsilon : float
            The threshold at which to select sources as "active". 
            0.25 -> select all sources that are active at least 25 % of the
            maximum dipoles.
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data.
            (1, 10) -> Samples have varying SNR from 1 to 10.
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape
        
        if batch_size == "auto":
            batch_size = n_dipoles
        if n_filters == "auto":
            n_filters = n_channels
            
        # Store Parameters
        # Architecture
        self.n_filters = n_filters
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        self.add_forward_error = add_forward_error
        self.forward_error=forward_error
        # Inference
        self.epsilon = epsilon
        print("Create Generator:..")
        self.create_generator()
        print("Build Model:..")
        self.build_model()
        print("Train Model:..")
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data)
        stc = self.source_to_object(source_mat)

        return stc

    def apply_model(self, data) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''

        y = deepcopy(data)
        y -= y.mean(axis=0)
        # y_norm = y / np.linalg.norm(y, axis=0)
        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = y@y.T
        # Scale
        C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]

        # Get prior source covariance from model
        gammas = self.model.predict(C, verbose=self.verbose)[0]
        # gammas = np.maximum(gammas, 0)
        gammas /= gammas.max()
        gammas[gammas<self.epsilon] = 0
        source_covariance = np.diag(gammas)

        # Perform inversion
        # L_s = self.leadfield @ source_covariance
        # L = self.leadfield
        # W = np.diag(np.linalg.norm(L, axis=0)) 
        # x_hat = source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T @ y

        # # Select dipole indices
        # gammas[gammas<self.epsilon] = 0
        # dipole_idc = np.where(gammas!=0)[0]
        # print("Active dipoles: ", len(dipole_idc))

        # # 1) Calculate weighted minimum norm solution at active dipoles
        # n_dipoles = len(gammas)
        # x_hat = np.zeros((n_dipoles, n_times))
        # L = self.leadfield[:, dipole_idc]
        # W = np.diag(np.linalg.norm(L, axis=0))
        # x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        # Bayes-like inversion
        Gamma = source_covariance
        Sigma_y = self.leadfield @ Gamma @ self.leadfield.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ self.leadfield.T @ Sigma_y_inv
        x_hat = inverse_operator @ y
        return x_hat        
         
    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator (and clear all repetitions with loop)
        for _ in range(self.batch_repetitions):
            x_val, y_val = self.generator.__next__()
        
        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(n_channels, n_channels, 1), name='Input')

        cnn1 = Conv2D(self.n_filters, (1, n_channels),
                    activation=self.activation_function, padding="valid",
                    name='CNN1')(inputs)

        flat = Flatten()(cnn1)

        fc1 = Dense(300, 
            activation=self.activation_function, 
            name='FC1')(flat)

        out = Dense(n_dipoles, 
            activation="elu", 
            name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')

        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=True, return_mask=True, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error)
        self.generator = generator(self.forward, **gen_args)
        
class SolverFC(BaseSolver):
    ''' Class for the Fully-Connected Neural Network (FC) for 
        EEG inverse solutions.
    
    Attributes
    ----------

    '''

    def __init__(self, name="Fully-Connected", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_dense_units=300, 
                            activation_function="tanh", 
                            batch_size="auto", n_timepoints=20, 
                            batch_repetitions=10, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            snr_range=(1,100), patience=100, alpha="auto", 
                            add_forward_error=False, forward_error=0.1,
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        activation_function : str
            The activation function of the hidden layers.
        batch_size : ["auto", int]
            The batch_size used during training. If "auto", the batch_size
            defaults to the number of dipoles in the source/ forward model.
            Choose a smaller batch_size (e.g., 1000) if you run into memory
            problems (RAM or GPU memory).
        n_timepoints : int
            The number of time points to simulate and ultimately train the
            neural network on.
        batch_repetitions : int
            The number of learning repetitions on the same batch of training
            data until a new batch is simulated.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        n_sources : int
            The maximum number of sources to simulate for the training data.
        n_orders : int
            Controls the maximum smoothness of the sources.
        size_validation_set : int
            The size of validation data set.
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data.
            (1, 10) -> Samples have varying SNR from 1 to 10.
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, verbose=self.verbose, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape
        
        if batch_size == "auto":
            batch_size = n_dipoles

            
        # Store Parameters
        # Architecture
        self.n_dense_units = n_dense_units
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        self.add_forward_error = add_forward_error
        self.forward_error = forward_error
        # MISC
        self.verbose = verbose
        # Inference
        print("Create Generator:..")
        self.create_generator()
        print("Build Model:..")
        self.build_model()
        print("Train Model:..")
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data)
        stc = self.source_to_object(source_mat)

        return stc

    def apply_model(self, data) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''

        y = deepcopy(data)
        y -= y.mean(axis=0)
        y /= np.linalg.norm(y, axis=0)
        y /= abs(y).max()

        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        
        # Add empty batch and (color-) channel dimension
        y = y.T[np.newaxis]
        # Predict source(s)
        source_pred = self.model.predict(y, verbose=self.verbose)
        source_pred = np.swapaxes(source_pred, 1, 2)

        # Rescale sources
        y_original = deepcopy(data)
        y_original = y_original[np.newaxis]
        source_pred_scaled = solve_p_wrap(self.leadfield, source_pred, y_original)
        
        return source_pred_scaled[0]
        
        
    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator
        x_val, y_val = self.generator.__next__()
        x_val = x_val[:self.size_validation_set]
        y_val = y_val[:self.size_validation_set]

        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(None, n_channels), name='Input')

        dense = TimeDistributed(Dense(self.n_dense_units, 
                activation=self.activation_function), name=f'FC1')(inputs)    
        
        dense = TimeDistributed(Dense(self.n_dense_units, 
                activation=self.activation_function), name=f'FC2')(dense)

        out = Dense(n_dipoles, 
            activation="sigmoid", 
            name='Output')(dense)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='FC')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=False, return_mask=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,)
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()

class SolverLSTM(BaseSolver):
    ''' Class for the Long-Short Term Memory Neural Network (LSTM) for 
        EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="LSTM", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_dense_units=300,
                            n_lstm_units=75,  
                            activation_function="tanh", 
                            batch_size="auto", n_timepoints=20, 
                            batch_repetitions=10, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            snr_range=(1,100), patience=100, alpha="auto", 
                            add_forward_error=False, forward_error=0.1,
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        n_dense_units : int
            The number of neurons in the fully-connected hidden layers.
        n_lstm_units : int
            The number of neurons in the LSTM hidden layers.
        activation_function : str
            The activation function of the hidden layers.
        batch_size : ["auto", int]
            The batch_size used during training. If "auto", the batch_size
            defaults to the number of dipoles in the source/ forward model.
            Choose a smaller batch_size (e.g., 1000) if you run into memory
            problems (RAM or GPU memory).
        n_timepoints : int
            The number of time points to simulate and ultimately train the
            neural network on.
        batch_repetitions : int
            The number of learning repetitions on the same batch of training
            data until a new batch is simulated.
        epochs : int
            The number of epochs to train.
        learning_rate : float
            The learning rate of the optimizer that trains the neural network.
        loss : str
            The loss function of the neural network.
        n_sources : int
            The maximum number of sources to simulate for the training data.
        n_orders : int
            Controls the maximum smoothness of the sources.
        size_validation_set : int
            The size of validation data set.
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data. (1,
            10) -> Samples have varying SNR from 1 to 10.
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, verbose=self.verbose, **kwargs)
        n_channels, n_dipoles = self.leadfield.shape
        
        if batch_size == "auto":
            batch_size = n_dipoles
            
        # Store Parameters
        # Architecture
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.activation_function = activation_function
        # Training
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.size_validation_set = size_validation_set
        self.patience = patience
        # Training Data
        self.n_timepoints = n_timepoints
        self.n_sources = n_sources
        self.n_orders = n_orders
        self.batch_repetitions = batch_repetitions
        self.snr_range = snr_range
        self.add_forward_error = add_forward_error
        self.forward_error = forward_error
        # MISC
        self.verbose = verbose
        # Inference
        print("Create Generator:..")
        self.create_generator()
        print("Build Model:..")
        self.build_model()
        # self.build_model2()
        print("Train Model:..")
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, mne_obj) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        
        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data)
        stc = self.source_to_object(source_mat)

        return stc

    def apply_model(self, data) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''
        y = deepcopy(data)
        y -= y.mean(axis=0)
        y /= np.linalg.norm(y, axis=0)
        y /= abs(y).max()

        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        
        # Add empty batch and (color-) channel dimension
        y = y.T[np.newaxis]
        # Predict source(s)
        source_pred = self.model.predict(y, verbose=self.verbose)
        source_pred = np.swapaxes(source_pred, 1, 2)

        # Rescale sources
        y_original = deepcopy(data)
        y_original = y_original[np.newaxis]
        source_pred_scaled = solve_p_wrap(self.leadfield, source_pred, y_original)
        
        return source_pred_scaled[0]
               
    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator
        x_val, y_val = self.generator.__next__()
        x_val = x_val[:self.size_validation_set]
        y_val = y_val[:self.size_validation_set]

        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model2(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(None, n_channels), name='Input')

        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units)), 
            name='LSTM1')(inputs)

        lstm2 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units)), 
            name='LSTM2')(lstm1)

        # lstm3 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
        #     input_shape=(None, self.n_dense_units)), 
        #     name='LSTM3')(lstm2)
        
        out = TimeDistributed(Dense(n_dipoles, activation="relu"))(lstm2)
        
        model = tf.keras.Model(inputs=inputs, outputs=out, name='LSTM2')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(None, n_channels), name='Input')

        dense = TimeDistributed(Dense(self.n_dense_units, 
                activation=self.activation_function), name=f'FC1')(inputs)    
        
        direct_out = TimeDistributed(Dense(n_dipoles, 
                activation="linear"), name=f'FC2')(dense)

        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units)), 
            name='LSTM1')(dense)
        mask = TimeDistributed(Dense(n_dipoles, 
                    activation="sigmoid"), 
                    name='Mask')(lstm1)

        out = multiply([direct_out, mask], name="multiply")

        model = tf.keras.Model(inputs=inputs, outputs=out, name='LSTM')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=False, return_mask=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,)
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()

def rms(x):
        return np.sqrt(np.mean(x**2))
    
def add_white_noise(X_clean, snr):
    ''' '''
    # Inter-channel correlations
    coeff_mat = (np.random.rand(X_clean.shape[0], X_clean.shape[0])-0.5) * 2
    # Make matrix symmetric
    coeff_mat = (coeff_mat + coeff_mat.T)/2
    
    # Random partially correlated noise
    X_noise = coeff_mat @ np.random.randn(*X_clean.shape)

    rms_clean = rms(X_clean)
    scaler = rms_clean / snr

    X_full = X_clean + X_noise*scaler
    X_full -= X_full.mean(axis=0)
    return X_full

def add_error(leadfield, forward_error, gradient):
    n_chans, n_dipoles = leadfield.shape
    noise = np.random.uniform(-1, 1, (n_chans, n_dipoles)) @ gradient
    leadfield_mix = leadfield / np.linalg.norm(leadfield) + forward_error * noise / np.linalg.norm(noise)
    return leadfield_mix

def generator(fwd, use_cov=True, batch_size=1284, batch_repetitions=30, n_sources=10, 
              n_orders=2, amplitude_range=(0.001,1), n_timepoints=20, 
              snr_range=(1, 100), n_timecourses=5000, beta_range=(0, 3),
              return_mask=True, scale_data=True, return_info=False,
              add_forward_error=False, forward_error=0.1, verbose=0):
    

    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose)
    # Convert to sparse matrix for speedup
    adjacency = csr_matrix(adjacency)
    gradient = abs(laplacian(adjacency))
    leadfield = deepcopy(fwd["sol"]["data"])
    leadfield_original = deepcopy(fwd["sol"]["data"])
    
    del adjacency
    
    # Convert to sparse matrix for speedup
    gradient = csr_matrix(gradient)

    leadfield -= leadfield.mean()
    # Normalize columns of the leadfield
    leadfield /= np.linalg.norm(leadfield, axis=0)

    n_chans, n_dipoles = leadfield.shape


    sources = csr_matrix(np.identity(n_dipoles))
    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = n_orders
    else:
        min_order = 0
        max_order = n_orders

    for i in range(max_order-1):
        # new_sources = sources[-n_dipoles:, -n_dipoles:] @ gradient
        # new_sources /= new_sources.max(axis=0)
        # sources = np.concatenate( [sources, new_sources], axis=0 )
    
        new_sources = csr_matrix(sources.toarray()[-n_dipoles:, -n_dipoles:]) @ gradient
        row_maxes = new_sources.max(axis=0).toarray().flatten()
        new_sources = new_sources / row_maxes[np.newaxis]

        # new_sources /= new_sources.max(axis=0)
        if i >= min_order-1:
            # sources = np.concatenate( [sources, new_sources], axis=0 )
            sources = vstack([sources, new_sources])

    
    if min_order>0:
        start_idx = int(n_dipoles*min_order)
        sources = csr_matrix(sources.toarray()[start_idx:, :])
    sources = csr_matrix(sources)
    # Pre-compute random time courses
    betas = np.random.uniform(*beta_range,n_timecourses)
    time_courses = np.stack([cn.powerlaw_psd_gaussian(beta, n_timepoints) for beta in betas], axis=0)

    # Normalize time course to max(abs()) == 1
    time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T

    n_candidates = sources.shape[0]
    while True:
        if add_forward_error:
            leadfield = add_error(leadfield_original, forward_error, gradient)
        # print("yeet")
        # select sources or source patches
        n_sources_batch = np.random.randint(1, n_sources+1, batch_size)
        selection = [np.random.randint(0, n_candidates, n) for n in n_sources_batch]

        # Assign each source (or source patch) a time course
        amplitude_values = [np.random.uniform(*amplitude_range, n) for n in n_sources_batch]
        amplitudes = [time_courses[np.random.choice(n_timecourses, n)].T * amplitude_values[i] for i, n in enumerate(n_sources_batch)]
        # y = np.stack([(amplitudes[i] @ sources.toarray()[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)
        y = np.stack([(amplitudes[i] @ sources[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)

        # Project simulated sources through leadfield
        x = np.stack([leadfield @ yy.T for yy in y], axis=0)

        # Add white noise to clean EEG
        snr_levels = np.random.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        x = np.stack([add_white_noise(xx, snr_level) for (xx, snr_level) in zip(x, snr_levels)], axis=0)


        # Apply common average reference
        x = np.stack([xx - xx.mean(axis=0) for xx in x], axis=0)
        # Scale eeg
        # if scale_data:
        #     x = np.stack([xx / np.linalg.norm(xx, axis=0) for xx in x], axis=0)
        
        if use_cov:
            # Calculate Covariance
            x = np.stack([xx@xx.T for xx in x], axis=0)

            # Normalize Covariance to abs. max. of 1
            x = np.stack([C / np.max(abs(C)) for C in x], axis=0)
            x = np.expand_dims(x, axis=-1)
        
        else:
            if scale_data:
                # normalize all time points to unit length
                x = np.stack([xx / np.linalg.norm(xx, axis=0) for xx in x], axis=0)
                # normalize each sample to max(abs()) == 1
                x = np.stack([xx / np.max(abs(xx)) for xx in x], axis=0)
            # Reshape
            x = np.swapaxes(x, 1,2)

        if return_mask:    
            # Calculate mean source activity
            # y = abs(y).mean(axis=1)
            # # Masking the source vector (1-> active, 0-> inactive)
            # y = (y>0).astype(float)
            # y = np.stack([np.diagonal(yy.T@yy) for yy in y], axis=0)
            y = np.mean(y**2, axis=1)
        
        else:
            if scale_data:
                y = np.stack([ (yy.T / np.max(abs(yy), axis=1)).T for yy in y], axis=0)
            

        
        # Return same batch multiple times:
        if return_info:
            info = pd.DataFrame(dict(n_sources=n_sources_batch, amplitudes=amplitude_values, snr=snr_levels))
            output = (x, y, info)
        else:
            output = (x, y)

        for _ in range(batch_repetitions):
            yield output

def solve_p_wrap(leadfield, y_est, x_true):
    ''' Wrapper for parallel (or, alternatively, serial) scaling of 
    predicted sources.
    '''


    y_est_scaled = deepcopy(y_est)

    for trial, _ in enumerate(x_true):
        for time in range(x_true[trial].shape[-1]):
            scaled = solve_p(leadfield, y_est[trial][:, time], x_true[trial][:, time])
            y_est_scaled[trial][:, time] = scaled

    return y_est_scaled

def solve_p(leadfield, y_est, x_true):
    '''
    Parameters
    ---------
    y_est : numpy.ndarray
        The estimated source vector.
    x_true : numpy.ndarray
        The original input EEG vector.
    
    Return
    ------
    y_scaled : numpy.ndarray
        The scaled estimated source vector.
    
    '''
    # Check if y_est is just zeros:
    if np.max(y_est) == 0:
        return y_est
    y_est = np.squeeze(np.array(y_est))
    x_true = np.squeeze(np.array(x_true))
    # Get EEG from predicted source using leadfield
    x_est = np.matmul(leadfield, y_est)

    # optimize forward solution
    tol = 1e-9
    options = dict(maxiter=1000, disp=False)

    # base scaling
    rms_est = np.mean(np.abs(x_est))
    rms_true = np.mean(np.abs(x_true))
    base_scaler = rms_true / rms_est

    
    opt = minimize_scalar(correlation_criterion, args=(leadfield, y_est* base_scaler, x_true), \
        bounds=(0, 1), options=options, tol=tol)


    scaler = opt.x
    y_scaled = y_est * scaler * base_scaler
    return y_scaled


def correlation_criterion(scaler, leadfield, y_est, x_true):
    ''' Perform forward projections of a source using the leadfield.
    This is the objective function which is minimized in Net::solve_p().
    
    Parameters
    ----------
    scaler : float
        scales the source y_est
    leadfield : numpy.ndarray
        The leadfield (or sometimes called gain matrix).
    y_est : numpy.ndarray
        Estimated/predicted source.
    x_true : numpy.ndarray
        True, unscaled EEG.
    '''

    x_est = np.matmul(leadfield, y_est) 
    error = np.abs(pearsonr(x_true-x_est, x_true)[0])
    return error

def emd_loss(distances):
    def loss(y_true, y_pred):
        y_true = y_true / K.sum(y_true)
        y_pred = y_pred / K.sum(y_pred)
        
        y_shape = tf.shape(y_true)
        if len(y_shape)==3:
            # y_true = tf.reshape(y_true, [tf.reduce_prod([y_shape[0],y_shape[1]]), y_shape[2]])
            # y_pred = tf.reshape(y_pred, [tf.reduce_prod([y_shape[0],y_shape[1]]), y_shape[2]])
            y_true = tf.reshape(y_true, [y_shape[0]*y_shape[1], y_shape[2]])
            y_pred = tf.reshape(y_pred, [y_shape[0]*y_shape[1], y_shape[2]])

        emd_score = tf.linalg.matmul( distances, tf.transpose(K.square(y_true-y_pred)))
        emd_score = K.sum(emd_score)
        return emd_score
    return loss