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
from ..simulate import generator

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
                            epsilon=0.25, snr_range=(-5,5), patience=300,
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
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
                            n_dense_units=300, n_dense_layers=2, 
                            activation_function="tanh", inter_source_correlation=(0, 1), 
                            output_activation="sigmoid", batch_size="auto", n_timepoints=20, 
                            batch_repetitions=10, 
                            epochs=300, learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=(1, 10), n_orders=(0, 10), size_validation_set=256,
                            epsilon=0., snr_range=(-5,5), patience=100,
                            add_forward_error=False, correlation_mode=None, noise_color_coeff=0, 
                            forward_error=0.1, cov_type="basic",
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
        cov_type : str
            The type of covariance matrix to use. 
            "basic" -> Use the basic
            "SSM" -> use the Signal Subspace Matching (SSM) covariance
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
        self.forward = forward
        # Store Parameters
        # Architecture
        self.n_filters = n_filters
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
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
        self.inter_source_correlation = inter_source_correlation
        self.cov_type = cov_type
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
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

    def compute_covariance(self, Y: np.ndarray) -> np.ndarray:
        ''' Compute the covariance matrix of the data.

        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.
        
        Return
        ------
        C : numpy.ndarray
            The covariance matrix.
        '''
        if self.cov_type == "basic":
            C = Y @ Y.T
        elif self.cov_type == "SSM":
            n_time = Y.shape[1]
            M_Y = Y.T @ Y
            YY = M_Y + 0.001 * (50/n_time) * np.trace(M_Y) * np.eye(n_time)
            P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
            C = P_Y.T @ P_Y
            print("yes its new")

        else:
            msg = "Covariance type not recognized. Use 'basic', 'SSM' or provide a custom covariance matrix."
            raise ValueError(msg)

        return C

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
        # y -= y.mean(axis=0)
        # y_norm = y / np.linalg.norm(y, axis=0)
        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = self.compute_covariance(y)
        # Scale
        C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]

        # Get prior source covariance from model
        gammas = self.model.predict(C, verbose=self.verbose)[0]
        self.gammas = deepcopy(gammas)
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
        
        self.gammas = gammas
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
        
        self.history = self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)
    
    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(n_channels, n_channels, 1), name='Input')
        # leadfield = self.forward["sol"]["data"]
        # leadfield /= abs(leadfield).max(axis=0)
        # cnn1 = CustomConv2D(leadfield, leadfield.shape[1])(inputs)
        cnn1 = Conv2D(self.n_filters, (1, n_channels),
                    activation=self.activation_function, padding="valid",
                    name='CNN1')(inputs)
        
        flat = Flatten()(cnn1)

        for i in range(self.n_dense_layers):
            flat = Dense(self.n_dense_units, 
                activation=self.activation_function, 
                name=f'FC{i+1}')(flat)

        out = Dense(n_dipoles, 
            activation=self.output_activation, # softmax
            name='Output')(flat)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))#, metrics=["cosine_similarity",])
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=True, return_mask=True, scale_data=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,
                inter_source_correlation=self.inter_source_correlation, correlation_mode=self.correlation_mode, cov_type=self.cov_type,
                noise_color_coeff=self.noise_color_coeff)
        self.generator = generator(self.forward, **gen_args)
        

class SolverCovTriCNN(BaseSolver):
    ''' Class for the Covariance-based Convolutional Neural Network (CovTriCNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="Cov-Tri-CNN", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_filters="auto", 
                            n_dense_units=300, n_dense_layers=2, 
                            activation_function="tanh", inter_source_correlation=(0, 1), 
                            output_activation="sigmoid", batch_size="auto", n_timepoints=20, 
                            batch_repetitions=10, 
                            epochs=300, learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=(1, 10), n_orders=(0, 10), size_validation_set=256,
                            epsilon=0., snr_range=(-5,5), patience=100,
                            add_forward_error=False, correlation_mode=None, noise_color_coeff=0, 
                            forward_error=0.1, cov_type="basic",
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
        cov_type : str
            The type of covariance matrix to use. 
            "basic" -> Use the basic
            "SSM" -> use the Signal Subspace Matching (SSM) covariance
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
        self.forward = forward
        # Store Parameters
        # Architecture
        self.n_filters = n_filters
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
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
        self.inter_source_correlation = inter_source_correlation
        self.cov_type = cov_type
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
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

    def compute_covariance(self, Y: np.ndarray) -> np.ndarray:
        ''' Compute the covariance matrix of the data.

        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.
        
        Return
        ------
        C : numpy.ndarray
            The covariance matrix.
        '''
        if self.cov_type == "basic":
            C = Y @ Y.T
        elif self.cov_type == "SSM":
            n_time = Y.shape[1]
            M_Y = Y.T @ Y
            YY = M_Y + 0.001 * (50/n_time) * np.trace(M_Y) * np.eye(n_time)
            P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
            C = P_Y.T @ P_Y
            print("yes its new")

        else:
            msg = "Covariance type not recognized. Use 'basic', 'SSM' or provide a custom covariance matrix."
            raise ValueError(msg)

        return C

    def apply_model(self, data, n_sources, refine_solution=True, max_iter=6, alpha=0.01) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.
        n_sources : int
            The number of sources to estimate.
        refine_solution : bool
            Whether to refine the solution using the estimated sources. 
        max_iter : int
            The maximum number of iterations to refine the solution.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''

        Y = deepcopy(data)
        n_channels, n_times = Y.shape
        _, n_dipoles = self.leadfield.shape
        # print(alpha)
        # Compute Data Covariance Matrix
        C =self.get_diag_and_lower(self.compute_covariance(Y))
        # Scale
        C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        self.estimates = dict()
        estimated_idc = []
        # Add all sources
        for i_iter in range(n_sources):
            # print(f"iter {i_iter+1}")
            X = self.model.predict(C, verbose=self.verbose)[0]
            # print(f"\tmax_val iter {i_iter} = {X.max()}")
            # X /= abs(X).max()
            X /= X.max()
            self.estimates[f"iter_{i_iter}"] = X
            X[np.array(estimated_idc).astype(int)] = 0  # prevent double selection
            next_index = np.argmax(X)
            estimated_idc.append( next_index )

            # # Outproject on whole data:
            current_data = self.outproject_from_data(Y.copy(), self.leadfield, estimated_idc, alpha=alpha)
            C = self.get_diag_and_lower(self.compute_covariance(current_data))
            C /= abs(C).max()
            C = C[np.newaxis]
            

        
        estimated_idc = np.array(estimated_idc)
        # print(f"before refinement: {estimated_idc}")
        # Phase 2: refinement
        candidates_2 = deepcopy(estimated_idc)
        list_of_sets = [set(estimated_idc)]
        if len(estimated_idc) > 1 and refine_solution:
            candidates_2_prev = deepcopy(candidates_2)
            for j in range(max_iter):#
                # print(f"\titeration {j+1}")
                # A_q_j = A_q.copy()
                for qq in range(n_sources):
                    candidates_2 = np.delete(candidates_2, qq, axis=0) # delete the current source
                    
                    # # Outproject on whole data
                    # current_data = self.outproject_from_data(Y.copy(), self.leadfield, candidates_2, alpha=alpha)
                    # C = self.compute_covariance(current_data)
                    # C /= abs(C).max()
                    # C = C[np.newaxis, :, :, np.newaxis]
                    
                    # Outproject on data covariance:
                    C = self.outproject_from_data_cov(Y.copy(), self.leadfield, candidates_2, alpha=alpha)
                    C /= abs(C).max()
                    C = C[np.newaxis, :, :, np.newaxis]

                    X = self.model.predict(C, verbose=self.verbose)[0]
                    # print(f"\tmax_val iter {j}/{qq} = {X.max()}")
                    X /= abs(X).max()
                    X[np.array(candidates_2).astype(int)] = 0  # prevcent double selection
                    next_index = np.argmax(X)
                    candidates_2 = np.insert(candidates_2, qq, next_index)
                    # print(f"\tinserting {next_index} at {qq} (replaces {estimated_idc[qq]}): {candidates_2}") 

                
                if set(candidates_2) in list_of_sets:#set(candidates_2_prev):
                    # print(f"\t\tNo change after {j+1} iteration(s)")
                    break
                # else:
                #     print(candidates_2_prev, " ==> ", candidates_2)
                candidates_2_prev = deepcopy(candidates_2)
                list_of_sets.append(set(candidates_2))
                
        # print(f"after refinement: {candidates_2}")
        # Get prior source covariance from model
        L = self.leadfield[:, candidates_2]
        gradients = np.zeros((n_dipoles, n_sources))
        x_hat = np.zeros((n_dipoles, n_times))
        x_hat[candidates_2, :] = L.T @ np.linalg.pinv(L @ L.T) @ data
        
        return x_hat
         
    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator (and clear all repetitions with loop)
        for _ in range(self.batch_repetitions):
            x_val, y_val = self.generator.__next__()
        
        self.history = self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)
    
    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_chans, n_dipoles = self.leadfield.shape
        input_size = int((n_chans**2 - n_chans) / 2)
        input_shape = (input_size,)  # Specify the input shape based on your data

        model = tf.keras.Sequential([
            # Reshape input to match the expected input for Conv2D
            tf.layers.Reshape((1, input_size, 1), input_shape=input_shape),
            
            # Convolutional layer
            tf.layers.Conv2D(self.n_filters, (1, input_size), activation='tanh'),
            
            # Flatten the output from the Conv2D layer
            tf.layers.Flatten(),
            
            # Fully connected layers
            tf.layers.Dense(self.n_dense_units, activation='tanh'),
            
            tf.layers.Dense(n_dipoles, activation='sigmoid')
        ])

        model.build(input_shape=(None, input_size))  # Build the model with the specified input shape

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss)  # Specify the loss function and optimizer
        model.build()
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=True, return_mask=True, scale_data=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,
                inter_source_correlation=self.inter_source_correlation, correlation_mode=self.correlation_mode, cov_type=self.cov_type,
                noise_color_coeff=self.noise_color_coeff)
        self.generator = generator(self.forward, **gen_args)
    
    @staticmethod
    def get_diag_and_lower(matrix):
        """
        This function takes a square matrix and returns a flattened array
        containing its diagonal and lower diagonal values.
        
        Parameters:
        matrix (np.ndarray): A square matrix.

        Returns:
        np.ndarray: A flattened array of the diagonal and lower diagonal values.
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The input matrix must be square.")
        
        diag_and_lower = matrix[np.tril_indices(matrix.shape[0])]
        
        return diag_and_lower

    
    @staticmethod
    def outproject_from_data(data, leadfield, idc: np.array, alpha=0.):
        """
        Projects away the leadfield components at the indices idc from the EEG data.

        Parameters:
        data (np.array): Observed M/EEG data (n_chans x n_time).
        leadfield (np.array): Leadfield matrix (n_chans x n_dipoles).
        idc (np.array): Indices to project away from the leadfield.

        Returns:
        np.array: Data with the specified leadfield components removed.
        """
        # Select the columns of the leadfield matrix corresponding to the indices
        L_idc = leadfield[:, idc]

        # Compute the projection matrix
        # P = I - L(L.TL)^-1L.T
        # where L = L_idc
        L_idc_T = L_idc.T
        projection_matrix = np.eye(leadfield.shape[0]) - L_idc @ np.linalg.pinv(L_idc_T @ L_idc + np.identity(len(idc)) * alpha) @ L_idc_T

        # Apply the projection matrix to the data
        data_without_idc = projection_matrix @ data

        return data_without_idc


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
                            snr_range=(-5,5), patience=100, alpha="auto", 
                            add_forward_error=False, forward_error=0.1, 
                            correlation_mode=None, noise_color_coeff=0,
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
            The range of signal to noise ratios (SNRs) in the training data (in dB).
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        correlation_mode : None/str
            None implies no correlation between the noise in different channels.
            'bounded' : Colored bounded noise, where channels closer to each other will be more correlated.
            'diagonal' : Some channels have varying degrees of noise.
        noise_color_coeff : float
            The magnitude of spatial coloring of the noise.
        
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
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
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
        # y -= y.mean(axis=0)
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

        # dense = Bidirectional(LSTM(self.n_dense_units, return_sequences=True))(inputs)
        # dense = Bidirectional(LSTM(self.n_dense_units, return_sequences=True))(dense)

        out = Dense(n_dipoles, 
            activation="linear", 
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
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, 
                forward_error=self.forward_error, correlation_mode=self.correlation_mode, 
                noise_color_coeff=self.noise_color_coeff, scale_data=True)
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()

class SolverCovLSTM(BaseSolver):
    ''' Class for the Covariance-based Long-Short Term Memory Neural Network (LSTM) for 
        EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="Cov-LSTM", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_dense_units=300,
                            n_lstm_units=25,
                            activation_function="tanh", 
                            output_activation="sigmoid",
                            batch_size="auto", n_timepoints=20,
                            batch_repetitions=10, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            snr_range=(-5,5), patience=100, alpha="auto", 
                            add_forward_error=False, forward_error=0.1,
                            inter_source_correlation=(0, 1), correlation_mode=None, 
                            noise_color_coeff=0, epsilon=0.0, verbose=0, **kwargs):
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
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
        self.output_activation = output_activation
        self.epsilon = epsilon
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
        self.inter_source_correlation = inter_source_correlation
        self.correlation_mode = correlation_mode 
        self.noise_color_coeff = noise_color_coeff
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
        n_channels, n_times = y.shape
        
        
        # Compute Data Covariance Matrix
        # C = y@y.T
        # # Scale
        # y /= (np.trace(C) / n_channels)
        y /= np.max(abs(y))

        # Add empty batch and (color-) channel dimension
        y = y.T

        # Get prior source covariance from model
        gammas = self.model.predict(y[np.newaxis], verbose=self.verbose)[0]
        # gammas = np.maximum(gammas, 0)
        gammas /= gammas.max()
        gammas[gammas<self.epsilon] = 0
        self.gammas = gammas
        source_covariance = np.diag(gammas)
        print(source_covariance.shape)

        # Bayes-like inversion
        Gamma = source_covariance
        Sigma_y = self.leadfield @ Gamma @ self.leadfield.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ self.leadfield.T @ Sigma_y_inv
        x_hat = inverse_operator @ y.T
        return x_hat      
               
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

        # lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=False, 
        #     input_shape=(None, n_channels)), 
        #     name='LSTM1')(inputs)

        # Transformer Encoder Layer
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
            x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(inputs.shape[-1])(x)
            return x + res
        
        x = inputs
        for _ in range(2):
            x = transformer_encoder(inputs, head_size=128, num_heads=8, ff_dim=self.n_dense_units, dropout=0.)

        # transformer = transformer_encoder(inputs, head_size=64, num_heads=8, ff_dim=self.n_dense_units, dropout=0.)

        # Global average pooling to reduce dimensions
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Fully connected layer
        fc1 = Dense(self.n_dense_units, 
                    activation=self.activation_function, 
                    name='FC1')(x)

        # Output layer
        out = Dense(n_dipoles, 
                    activation=self.output_activation, 
                    name='Output')(fc1)

        # fc1 = Dense(self.n_dense_units, 
        #     activation=self.activation_function, 
        #     name='FC1')(lstm1)

        # out = Dense(n_dipoles, 
        #     activation="sigmoid", 
        #     name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='LSTM')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(
            use_cov=False, 
            return_mask=True, 
            batch_size=self.batch_size, 
            batch_repetitions=self.batch_repetitions,
            n_sources=self.n_sources, 
            n_orders=self.n_orders, 
            n_timepoints=self.n_timepoints,
            snr_range=self.snr_range, 
            add_forward_error=self.add_forward_error, 
            forward_error=self.forward_error, 
            remove_channel_dim=True,
            inter_source_correlation=self.inter_source_correlation,
            correlation_mode=self.correlation_mode, 
            noise_color_coeff=self.noise_color_coeff,
            )
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()


class SolverCovLSTM2(BaseSolver):
    ''' Class for the Covariance-based Long-Short Term Memory Neural Network (LSTM) for 
        EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="Cov-LSTM2", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_dense_units=300, n_filters=64,
                            n_lstm_units=25,
                            activation_function="tanh", 
                            output_activation="sigmoid",
                            batch_size="auto", n_timepoints=20,
                            batch_repetitions=10, epochs=300,
                            learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=10, n_orders=2, size_validation_set=256,
                            snr_range=(-5,5), patience=100, alpha="auto", 
                            add_forward_error=False, forward_error=0.1,
                            inter_source_correlation=(0, 1), correlation_mode=None, 
                            noise_color_coeff=0, epsilon=0.0, verbose=0, **kwargs):
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
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
        self.n_filters = n_filters
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.epsilon = epsilon
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
        self.inter_source_correlation = inter_source_correlation
        self.correlation_mode = correlation_mode 
        self.noise_color_coeff = noise_color_coeff
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
        n_channels, n_times = y.shape
        
        
        # Compute Data Covariance Matrix
        # C = y@y.T
        # # Scale
        # y /= (np.trace(C) / n_channels)
        y /= np.max(abs(y))

        # Add empty batch and (color-) channel dimension
        y = y.T
        print(y.shape)
        # Get prior source covariance from model
        gammas = self.model.predict(y[np.newaxis], verbose=self.verbose)[0]
        # gammas = np.maximum(gammas, 0)
        gammas /= gammas.max()
        gammas[gammas<self.epsilon] = 0
        self.gammas = gammas
        source_covariance = np.diag(gammas)
        print(source_covariance.shape)

        # Bayes-like inversion
        Gamma = source_covariance
        Sigma_y = self.leadfield @ Gamma @ self.leadfield.T
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        inverse_operator = Gamma @ self.leadfield.T @ Sigma_y_inv
        x_hat = inverse_operator @ y.T
        return x_hat      
               
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

        inputs = tf.keras.Input(shape=(None, n_channels,1), name='Input')

        x = TimeDistributed(tf.keras.layers.Conv1D(self.n_filters, kernel_size=n_channels,
                    activation=self.activation_function, padding="valid",
                    name='CNN1'))(inputs)
        x = tf.keras.layers.Reshape((-1, self.n_filters))(x)

        # lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=False, 
        #     input_shape=(None, n_channels)), 
        #     name='LSTM1')(inputs)

        # Transformer Encoder Layer
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
            x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(inputs.shape[-1])(x)
            return x + res
        
        

        for _ in range(1):
            x = transformer_encoder(x, head_size=self.n_lstm_units, num_heads=8, ff_dim=self.n_dense_units, dropout=0.)
        

        # Global average pooling to reduce dimensions
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # x = Lambda(lambda y: tf.reduce_mean(tf.abs(y), axis=1))(x)

        # Fully connected layer
        fc1 = Dense(self.n_dense_units, 
                    activation=self.activation_function, 
                    name='FC1')(x)

        # Output layer
        out = Dense(n_dipoles, 
                    activation=self.output_activation, 
                    name='Output')(fc1)

        # fc1 = Dense(self.n_dense_units, 
        #     activation=self.activation_function, 
        #     name='FC1')(lstm1)

        # out = Dense(n_dipoles, 
        #     activation="sigmoid", 
        #     name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='LSTM')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(
            use_cov=False, 
            return_mask=True, 
            batch_size=self.batch_size, 
            batch_repetitions=self.batch_repetitions,
            n_sources=self.n_sources, 
            n_orders=self.n_orders, 
            n_timepoints=self.n_timepoints,
            snr_range=self.snr_range, 
            add_forward_error=self.add_forward_error, 
            forward_error=self.forward_error, 
            remove_channel_dim=True,
            inter_source_correlation=self.inter_source_correlation,
            correlation_mode=self.correlation_mode, 
            noise_color_coeff=self.noise_color_coeff,
            )
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
                            snr_range=(-5,5), patience=100, alpha="auto", 
                            inter_source_correlation=(0, 1), add_forward_error=False, forward_error=0.1, correlation_mode=None, 
                            noise_color_coeff=0, verbose=0, **kwargs):
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
            The range of signal to noise ratios (SNRs) in the training data (in dB).
        patience : int
            Stopping criterion for the training.
        alpha : float
            The regularization parameter.
        correlation_mode : None/str
            None implies no correlation between the noise in different channels.
            'bounded' : Colored bounded noise, where channels closer to each other will be more correlated.
            'diagonal' : Some channels have varying degrees of noise.
        noise_color_coeff : float
            The magnitude of spatial coloring of the noise.
        
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
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
        self.inter_source_correlation = inter_source_correlation

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

        self.history = self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
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
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, 
                forward_error=self.forward_error, correlation_mode=self.correlation_mode, 
                noise_color_coeff=self.noise_color_coeff, inter_source_correlation=self.inter_source_correlation)
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()


class SolverRAPNN(BaseSolver):
    ''' Class for the Covariance-based Convolutional Neural Network (CovCNN) for EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''

    def __init__(self, name="RAP-NN", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, *args, n_filters="auto", 
                            n_dense_units=300, n_dense_layers=2, 
                            activation_function="tanh", inter_source_correlation=(0, 1), 
                            output_activation="sigmoid", batch_size="auto", n_timepoints=20, 
                            batch_repetitions=10, 
                            epochs=300, learning_rate=1e-3, loss="cosine_similarity",
                            n_sources=(1, 10), n_orders=(0, 10), size_validation_set=256,
                            epsilon=0., snr_range=(-5,5), patience=100,
                            add_forward_error=False, correlation_mode=None, noise_color_coeff=0, 
                            forward_error=0.1, cov_type="basic", verbose=0,
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
        cov_type : str
            The type of covariance matrix to use. 
            "basic" -> Use the basic
            "SSM" -> use the Signal Subspace Matching (SSM) covariance
        snr_range : tuple
            The range of signal to noise ratios (SNRs) in the training data (in dB).
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
        self.forward = forward
        # Store Parameters
        # Architecture
        self.n_filters = n_filters
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
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
        self.inter_source_correlation = inter_source_correlation
        self.cov_type = cov_type
        self.correlation_mode = correlation_mode
        self.noise_color_coeff = noise_color_coeff
        self.verbose = verbose
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

    def apply_inverse_operator(self, mne_obj, n_sources, refine_solution=True, max_iter=6, alpha=0.01) -> mne.SourceEstimate:
        ''' Apply the inverse operator.
        
        Parameters
        ----------
        mne_obj : [mne.Evoked, mne.Epochs, mne.io.Raw]
            The MNE data object.
        n_sources : int
            The number of sources to estimate.
        refine_solution : bool
            Whether to refine the solution using the estimated sources.
        max_iter : int
            The maximum number of iterations to refine the solution.
        alpha : float
            The regularization parameter.

        Return
        ------
        stc : mne.SourceEstimate
            The mne Source Estimate object.
        '''
        data = self.unpack_data_obj(mne_obj)

        source_mat = self.apply_model(data, n_sources, refine_solution=refine_solution, 
                                      max_iter=max_iter, alpha=alpha)
        stc = self.source_to_object(source_mat)

        return stc

    def compute_covariance(self, Y: np.ndarray) -> np.ndarray:
        ''' Compute the covariance matrix of the data.

        Parameters
        ----------
        Y : numpy.ndarray
            The data matrix.
        
        Return
        ------
        C : numpy.ndarray
            The covariance matrix.
        '''
        if self.cov_type == "basic":
            C = Y @ Y.T
        elif self.cov_type == "SSM":
            n_time = Y.shape[1]
            M_Y = Y.T @ Y
            YY = M_Y + 0.001 * (50/n_time) * np.trace(M_Y) * np.eye(n_time)
            P_Y = (Y @ np.linalg.inv(YY)) @ Y.T
            C = P_Y.T @ P_Y
            print("yes its new")

        else:
            msg = "Covariance type not recognized. Use 'basic', 'SSM' or provide a custom covariance matrix."
            raise ValueError(msg)

        return C

    def apply_model(self, data, n_sources, refine_solution=True, max_iter=6, alpha=0.01) -> np.ndarray:
        ''' Compute the inverse solution of the M/EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The M/EEG data matrix.
        n_sources : int
            The number of sources to estimate.
        refine_solution : bool
            Whether to refine the solution using the estimated sources. 
        max_iter : int
            The maximum number of iterations to refine the solution.

        Return
        ------
        x_hat : numpy.ndarray
            The source esimate.

        '''

        Y = deepcopy(data)
        n_channels, n_times = Y.shape
        _, n_dipoles = self.leadfield.shape
        # print(alpha)
        # Compute Data Covariance Matrix
        C = self.compute_covariance(Y)
        # Scale
        C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        self.estimates = dict()
        estimated_idc = []
        # Add all sources
        for i_iter in range(n_sources):
            # print(f"iter {i_iter+1}")
            X = self.model.predict(C, verbose=self.verbose)[0]
            # print(f"\tmax_val iter {i_iter} = {X.max()}")
            # X /= abs(X).max()
            X /= X.max()
            self.estimates[f"iter_{i_iter}"] = X
            X[np.array(estimated_idc).astype(int)] = 0  # prevent double selection
            next_index = np.argmax(X)
            estimated_idc.append( next_index )

            # # Outproject on whole data:
            # current_data = self.outproject_from_data(Y.copy(), self.leadfield, estimated_idc, alpha=alpha)
            # C = self.compute_covariance(current_data)
            # C /= abs(C).max()
            # C = C[np.newaxis, :, :, np.newaxis]
            
            # Outproject on data covariance:
            C = self.outproject_from_data_cov(Y.copy(), self.leadfield, estimated_idc, alpha=alpha)
            C /= abs(C).max()
            C = C[np.newaxis, :, :, np.newaxis]
        
        estimated_idc = np.array(estimated_idc)
        # print(f"before refinement: {estimated_idc}")
        # Phase 2: refinement
        candidates_2 = deepcopy(estimated_idc)
        list_of_sets = [set(estimated_idc)]
        if len(estimated_idc) > 1 and refine_solution:
            candidates_2_prev = deepcopy(candidates_2)
            for j in range(max_iter):#
                # print(f"\titeration {j+1}")
                # A_q_j = A_q.copy()
                for qq in range(n_sources):
                    candidates_2 = np.delete(candidates_2, qq, axis=0) # delete the current source
                    
                    # # Outproject on whole data
                    # current_data = self.outproject_from_data(Y.copy(), self.leadfield, candidates_2, alpha=alpha)
                    # C = self.compute_covariance(current_data)
                    # C /= abs(C).max()
                    # C = C[np.newaxis, :, :, np.newaxis]
                    
                    # Outproject on data covariance:
                    C = self.outproject_from_data_cov(Y.copy(), self.leadfield, candidates_2, alpha=alpha)
                    C /= abs(C).max()
                    C = C[np.newaxis, :, :, np.newaxis]

                    X = self.model.predict(C, verbose=self.verbose)[0]
                    # print(f"\tmax_val iter {j}/{qq} = {X.max()}")
                    X /= abs(X).max()
                    X[np.array(candidates_2).astype(int)] = 0  # prevcent double selection
                    next_index = np.argmax(X)
                    candidates_2 = np.insert(candidates_2, qq, next_index)
                    # print(f"\tinserting {next_index} at {qq} (replaces {estimated_idc[qq]}): {candidates_2}") 

                
                if set(candidates_2) in list_of_sets:#set(candidates_2_prev):
                    # print(f"\t\tNo change after {j+1} iteration(s)")
                    break
                # else:
                #     print(candidates_2_prev, " ==> ", candidates_2)
                candidates_2_prev = deepcopy(candidates_2)
                list_of_sets.append(set(candidates_2))
                
        # print(f"after refinement: {candidates_2}")
        # Get prior source covariance from model
        L = self.leadfield[:, candidates_2]
        gradients = np.zeros((n_dipoles, n_sources))
        x_hat = np.zeros((n_dipoles, n_times))
        x_hat[candidates_2, :] = L.T @ np.linalg.pinv(L @ L.T) @ data
        
        return x_hat
    
    @staticmethod
    def outproject_from_data_cov(data, leadfield, idc: np.array, alpha=0.1):
        """
        Projects away the leadfield components at the indices idc from the EEG data covariance matrix.

        Parameters:
        data (np.array): Observed M/EEG data (n_chans x n_time).
        leadfield (np.array): Leadfield matrix (n_chans x n_dipoles).
        idc (np.array): Indices to project away from the leadfield.

        Returns:
        np.array: Data covariance matrix with the specified leadfield components removed.
        """
        # Compute the data covariance matrix
        C = data @ data.T
        
        # Select the columns of the leadfield matrix corresponding to the indices
        L_idc = leadfield[:, idc]

        # Compute the projection matrix
        # P = I - L(L.TL)^-1L.T
        # where L = L_idc
        L_idc_T = L_idc.T
        projection_matrix = np.eye(leadfield.shape[0]) - L_idc @ np.linalg.pinv(L_idc_T @ L_idc + np.identity(len(idc)) * alpha) @ L_idc_T

        # Apply the projection matrix to the covariance matrix
        C_projected = projection_matrix @ C @ projection_matrix.T

        return C_projected

    # @staticmethod
    # def outproject_from_data_cov(data, leadfield, idc: np.array, alpha=0.1):
    #     """
    #     Projects away the leadfield components at the indices idc from the EEG data.

    #     Parameters:
    #     data (np.array): Observed M/EEG data (n_chans x n_time).
    #     leadfield (np.array): Leadfield matrix (n_chans x n_dipoles).
    #     idc (np.array): Indices to project away from the leadfield.

    #     Returns:
    #     np.array: Data with the specified leadfield components removed.
    #     """
    #     print("now cov based")
    #     # Select the columns of the leadfield matrix corresponding to the indices
    #     C = data @ data.T
    #     A = leadfield[:, idc]
    #     P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
    #     Q = np.identity(leadfield.shape[0]) - P_A
    #     C_out = (Q @ C @ Q) / Q
    #     print(C_out.shape)
    #     return C_out
    
    @staticmethod
    def outproject_from_data(data, leadfield, idc: np.array, alpha=0.1):
        """
        Projects away the leadfield components at the indices idc from the EEG data.

        Parameters:
        data (np.array): Observed M/EEG data (n_chans x n_time).
        leadfield (np.array): Leadfield matrix (n_chans x n_dipoles).
        idc (np.array): Indices to project away from the leadfield.

        Returns:
        np.array: Data with the specified leadfield components removed.
        """
        # Select the columns of the leadfield matrix corresponding to the indices
        L_idc = leadfield[:, idc]

        # Compute the projection matrix
        # P = I - L(L.TL)^-1L.T
        # where L = L_idc
        L_idc_T = L_idc.T
        projection_matrix = np.eye(leadfield.shape[0]) - L_idc @ np.linalg.pinv(L_idc_T @ L_idc + np.identity(len(idc)) * alpha) @ L_idc_T

        # Apply the projection matrix to the data
        data_without_idc = projection_matrix @ data

        return data_without_idc

    def train_model(self,):
        ''' Train the neural network model.
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator (and clear all repetitions with loop)
        for _ in range(self.batch_repetitions):
            x_val, y_val = self.generator.__next__()
        
        self.history = self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)
    
    def build_model(self,):
        ''' Build the neural network model.
        '''
        n_channels, n_dipoles = self.leadfield.shape

        # inputs = tf.keras.Input(shape=(n_channels, n_channels, 1), name='Input')
        # # leadfield = self.forward["sol"]["data"]
        # # leadfield /= abs(leadfield).max(axis=0)
        # # cnn1 = CustomConv2D(leadfield, leadfield.shape[1])(inputs)
        # cnn1 = Conv2D(self.n_filters, (1, n_channels),
        #             activation=self.activation_function, padding="valid",
        #             name='CNN1')(inputs)
        
        # flat = Flatten()(cnn1)

        # for i in range(self.n_dense_layers):
        #     flat = Dense(self.n_dense_units, 
        #         activation=self.activation_function, 
        #         name=f'FC{i+1}')(flat)

        # out = Dense(n_dipoles, 
        #     activation=self.output_activation, # softmax
        #     name='Output')(flat)
        # model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')


        # Define the neural network architecture
        input_shape = (n_channels, n_channels, 1)  # Specify the input shape based on your data
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_channels, (1, n_channels), 
                activation="tanh", padding="valid",
                input_shape=input_shape,
                name='CNN1'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='tanh'),
            tf.keras.layers.Dense(n_dipoles, activation='sigmoid')
        ])

        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.AdamW(learning_rate=self.learning_rate))#, metrics=["cosine_similarity",])
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        ''' Creat the data generator used for the simulations.
        '''
        gen_args = dict(use_cov=True, return_mask=True, scale_data=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range, add_forward_error=self.add_forward_error, forward_error=self.forward_error,
                inter_source_correlation=self.inter_source_correlation, correlation_mode=self.correlation_mode, cov_type=self.cov_type,
                noise_color_coeff=self.noise_color_coeff)
        self.generator = generator(self.forward, **gen_args)
        

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


class Compressor:
    ''' Compression using Graph Fourier Transform
    '''
    def __init__(self):
        pass
    def fit(self, fwd, k=600):
        A = mne.spatial_src_adjacency(fwd["src"], verbose=0).toarray()
        # D = np.diag(A.sum(axis=0))
        # L = D-A
        L = laplacian(A)
        U, s, V = np.linalg.svd(L)

        self.U = U[:, -k:]
        self.s = s[-k:]
        self.V = V[:, -k:]
        # self.U = U[:, :k]
        # self.s = s[:k]
        # self.V = V[:, :k]
        return self
        
    def encode(self, X):
        ''' Encodes a true signal X
        Parameters
        ----------
        X : numpy.ndarray
            True signal
        
        Return
        ------
        X_comp : numpy.ndarray
            Compressed signal
        '''
        X_comp = self.U.T @ X

        return X_comp

    def decode(self, X_comp):
        ''' Decodes a compressed signal X

        Parameters
        ----------
        X : numpy.ndarray
            Compressed signal
        
        Return
        ------
        X_unfold : numpy.ndarray
            Decoded signal
        '''
        X_unfold = self.U @ X_comp

        return X_unfold
    
class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, eeg_matrix, output_channels, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        # Reshape eeg_matrix to the shape [1, kernel_width, 1, output_channels]
        self.eeg_kernels = np.reshape(eeg_matrix, (1, eeg_matrix.shape[0], 1, output_channels))

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=self.eeg_kernels.shape,
            initializer=tf.constant_initializer(self.eeg_kernels),
            trainable=False
        )

    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='VALID')


def hybrid_loss(y_true, y_pred, lambda_reg=0.01):
    # Cosine similarity component
    cosine_loss = 1 - tf.reduce_sum(tf.nn.l2_normalize(y_true, 0) * tf.nn.l2_normalize(y_pred, 0))
    
    # Normalized MAE component
    mae_loss = tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.norm(y_true)
    
    # Combine losses
    total_loss = cosine_loss + lambda_reg * mae_loss
    return total_loss

def sparse_cosine_similarity_loss(y_true, y_pred):
    # Calculate the signs and minima of corresponding elements
    signs = tf.sign(y_true) * tf.sign(y_pred)
    minima = tf.minimum(tf.abs(y_true), tf.abs(y_pred))
    
    # Modified dot product using signs and minima
    modified_dot_product = tf.reduce_sum(signs * minima)
    
    # Compute the norms of the true and predicted vectors
    norm_true = tf.norm(y_true)
    norm_pred = tf.norm(y_pred)
    
    # Calculate the cosine similarity using the modified dot product
    cosine_similarity = modified_dot_product / (norm_true * norm_pred)
    
    # Since we need a loss (lower is better), subtract the cosine similarity from 1
    loss = 1 - cosine_similarity
    return loss