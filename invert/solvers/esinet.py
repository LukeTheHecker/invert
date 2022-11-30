from copy import deepcopy
from esinet import Simulation, Net
from .base import BaseSolver, InverseOperator
from scipy.sparse.csgraph import laplacian
from scipy.stats import pearsonr
import mne
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Lambda, multiply,
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
                            epsilon=0.25, snr_range=(1,100), patience=10,
                            alpha="auto", verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
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
        # Inference
        self.epsilon = epsilon

        self.create_generator()
        self.build_model()
        self.train_model()

        self.inverse_operators = []
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
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

        
        

        # L-Curve for Epsilon Decision:
        # iters = np.arange(len(gammas))
        # sort_idx = np.argsort(gammas)[::-1]
        # gammas_sorted = gammas[sort_idx]
        # zero_idx = np.where(gammas_sorted<1e-3)[0][0]
        # n_comp = find_corner(iters[:zero_idx], gammas_sorted[:zero_idx])
        # self.epsilon = gammas_sorted[n_comp]
        # print("new eps: ", self.epsilon)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(iters, gammas_sorted, '*k')
        # plt.title("Gammas")
        # plt.plot(iters[n_comp], gammas_sorted[n_comp], 'og')
        


        # Select dipole indices
        gammas[gammas<self.epsilon] = 0
        dipole_idc = np.where(gammas!=0)[0]
        print("Active dipoles: ", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        
        return x_hat        
            
    def train_model(self,):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=self.generator.__next__(), callbacks=callbacks)


    def build_model(self,):
        n_channels, n_dipoles = self.leadfield.shape
        
        inputs = tf.keras.Input(shape=(self.n_timepoints, n_channels, 1), name='Input')


        cnn1 = Conv2D(self.n_filters, (1, n_channels),
                    activation=self.activation_function, padding="valid",
                    name='CNN1')(inputs)
        cnn1 = Lambda(lambda x: K.abs(x))(cnn1)
        # reshape = Reshape((self.n_timepoints, self.n_filters, 1))(cnn1)
        maxpool = AveragePooling2D(pool_size=(self.n_timepoints, 1), strides=None, padding="valid")(cnn1)

        flat = Flatten()(maxpool)

        hl1 = Dense(300, 
            activation=self.activation_function, 
            name='HL1')(flat)

        out = Dense(n_dipoles, 
            activation="relu", 
            name='Output')(hl1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        gen_args = dict(use_cov=False, return_mask=True, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
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
                            epsilon=0.5, snr_range=(1,100), patience=100,
                            alpha="auto", verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
        y -= y.mean(axis=0)
        # y_norm = y / np.linalg.norm(y, axis=0)
        n_channels, n_times = y.shape

        # Compute Data Covariance Matrix
        C = y@y.T
        # Scale
        C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        gammas = self.model.predict(C, verbose=self.verbose)[0]
        gammas /= gammas.max()

        # Select dipole indices
        gammas[gammas<self.epsilon] = 0
        dipole_idc = np.where(gammas!=0)[0]
        print("Active dipoles: ", len(dipole_idc))

        # 1) Calculate weighted minimum norm solution at active dipoles
        n_dipoles = len(gammas)
        x_hat = np.zeros((n_dipoles, n_times))
        L = self.leadfield[:, dipole_idc]
        W = np.diag(np.linalg.norm(L, axis=0))
        x_hat[dipole_idc, :] = np.linalg.inv(L.T @ L + W.T@W) @ L.T @ y

        return x_hat        
        
        
    def train_model(self,):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator
        x_val, y_val = self.generator.__next__()
        x_val = x_val[:256]
        y_val = y_val[:256]
        
        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model(self,):
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
            activation="relu", 
            name='Output')(fc1)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        gen_args = dict(use_cov=True, return_mask=True, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
        self.generator = generator(self.forward, **gen_args)
        

class SolverFC(BaseSolver):
    ''' Class for the Fully-Connected Neural Network (FC) for 
        EEG inverse solutions.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
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
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
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
        y_original = deepcopy(evoked.data)
        y_original = y_original[np.newaxis]
        source_pred_scaled = solve_p_wrap(self.leadfield, source_pred, y_original)
        
        return source_pred_scaled[0]
        
        
    def train_model(self,):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator
        x_val, y_val = self.generator.__next__()
        x_val = x_val[:self.size_validation_set]
        y_val = y_val[:self.size_validation_set]

        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model(self,):
        n_channels, n_dipoles = self.leadfield.shape

        inputs = tf.keras.Input(shape=(None, n_channels), name='Input')

        dense = TimeDistributed(Dense(self.n_dense_units, 
                activation=self.activation_function), name=f'FC1')(inputs)    
        
        dense = TimeDistributed(Dense(self.n_dense_units, 
                activation=self.activation_function), name=f'FC2')(dense)

        out = Dense(n_dipoles, 
            activation="linear", 
            name='Output')(dense)

        model = tf.keras.Model(inputs=inputs, outputs=out, name='CovCNN')
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        if self.verbose > 0:
            model.summary()
        
        self.model = model

    def create_generator(self,):
        gen_args = dict(use_cov=False, return_mask=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
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
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
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

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        source_mat = self.apply_model(evoked)
        stc = self.source_to_object(source_mat, evoked)

        return stc

    def apply_model(self, evoked) -> np.ndarray:
        y = deepcopy(evoked.data)
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
        y_original = deepcopy(evoked.data)
        y_original = y_original[np.newaxis]
        source_pred_scaled = solve_p_wrap(self.leadfield, source_pred, y_original)
        
        return source_pred_scaled[0]
        
        
    def train_model(self,):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),]
        
        # Get Validation data from generator
        x_val, y_val = self.generator.__next__()
        x_val = x_val[:self.size_validation_set]
        y_val = y_val[:self.size_validation_set]

        self.model.fit(x=self.generator, epochs=self.epochs, steps_per_epoch=self.batch_repetitions, 
                validation_data=(x_val, y_val), callbacks=callbacks)

    def build_model(self,):
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
        gen_args = dict(use_cov=False, return_mask=False, batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
        self.generator = generator(self.forward, **gen_args)
        self.generator.__next__()

class SolverFullyConnected(BaseSolver):
    ''' Class for the Fully-Connected (FC) neural network's inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Fully-Connected", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', 
                            n_simulations=5000, settings=None, activation_function="tanh", 
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        info = evoked.info
        if settings is None:
            settings = dict(duration_of_trial=0., )
        sim = Simulation(forward, info, settings=settings, verbose=verbose).simulate(n_simulations)

        model_args = dict(model_type="FC", activation_function=activation_function, )
        inverse_operator = InverseOperator(Net(forward, **model_args, verbose=verbose).fit(sim), self.name)
        self.inverse_operators = [inverse_operator,]
        
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

def make_fullyconnected_inverse_operator(fwd, info, n_samples=5000, settings=None, verbose=0):
    """ Calculate the inverse operator using the Fully-Connected artificial neural network model.

    Parameters
    ----------
    leadfield : mne.Foward
        The forward model object.
    info : mne.Info
        The mne info object.

    Return
    ------
    inverse_operator : esinet.Net
        The neural network model object from the esinet package.

    """
    if settings is None:
        settings = dict(duration_of_trial=0.)
    sim = Simulation(fwd, info, settings=settings, verbose=verbose).simulate(n_samples)

    model_args = dict(model_type="FC", activation_function="tanh")
    inverse_operator = Net(fwd, **model_args, verbose=verbose).fit(sim)

    return inverse_operator


def make_lstm_inverse_operator(fwd, info, n_samples=5000, settings=None, verbose=0):
    """ Calculate the inverse operator using the Long-Short Term Memory
    artificial neural network model.

    Parameters
    ----------
    leadfield : mne.Foward
        The forward model object.
    info : mne.Info
        The mne info object.

    Return
    ------
    inverse_operator : esinet.Net
        The neural network model object from the esinet package.

    """
    if settings is None:
        settings = dict(duration_of_trial=0.)
    sim = Simulation(fwd, info, settings=settings, verbose=verbose).simulate(n_samples)

    model_args = dict(model_type="LSTM")
    inverse_operator = Net(fwd, **model_args, verbose=verbose).fit(sim)

    return inverse_operator

def rms(x):
        return np.sqrt(np.mean(x**2))
    
def add_white_noise(X_clean, snr):
    ''' '''
    X_noise = np.random.randn(*X_clean.shape)

    rms_clean = rms(X_clean)
    scaler = rms_clean / snr

    X_full = X_clean + X_noise*scaler
    X_full -= X_full.mean(axis=0)
    return X_full
    
def generator(fwd, use_cov=True, batch_size=1284, batch_repetitions=30, n_sources=10, 
              n_orders=2, amplitude_range=(0.001,1), n_timepoints=20, 
              snr_range=(1, 100), n_timecourses=5000, beta_range=(0, 3),
              return_mask=True, verbose=0):
    import colorednoise as cn

    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose).toarray()
    gradient = abs(laplacian(adjacency))
    leadfield = fwd["sol"]["data"]
    leadfield -= leadfield.mean()
    # Normalize columns of the leadfield
    leadfield /= np.linalg.norm(leadfield, axis=0)

    n_chans, n_dipoles = leadfield.shape


    sources = np.identity(n_dipoles)
    for _ in range(n_orders-1):
        new_sources = sources[-n_dipoles:, -n_dipoles:] @ gradient
        new_sources /= new_sources.max(axis=0)
        sources = np.concatenate( [sources, new_sources], axis=0 )

    # Pre-compute random time courses
    betas = np.random.uniform(*beta_range,n_timecourses)
    # time_courses = np.stack([np.random.randn(n_timepoints) for _ in range(n_timecourses)], axis=0)
    time_courses = np.stack([cn.powerlaw_psd_gaussian(beta, n_timepoints) for beta in betas], axis=0)
    # Normalize time course to max(abs()) == 1
    time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T



    n_candidates = sources.shape[0]
    while True:
        # print("yeet")
        # select sources or source patches
        n_sources_batch = np.random.randint(1, n_sources+1, batch_size)
        selection = [np.random.randint(0, n_candidates, n) for n in n_sources_batch]

        # Assign each source (or source patch) a time course
        amplitudes = [time_courses[np.random.choice(n_timecourses, n)].T * np.random.uniform(*amplitude_range, n) for n in n_sources_batch]
        y = np.stack([(amplitudes[i] @ sources[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)
        
        # Project simulated sources through leadfield
        x = np.stack([leadfield @ yy.T for yy in y], axis=0)

        # Add white noise to clean EEG
        snr_levels = np.random.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        x = np.stack([add_white_noise(xx, snr_level) for (xx, snr_level) in zip(x, snr_levels)], axis=0)


        # Apply common average reference
        x = np.stack([xx - xx.mean(axis=0) for xx in x], axis=0)
        # Scale eeg
        x = np.stack([xx / np.linalg.norm(xx, axis=0) for xx in x], axis=0)
        
        if use_cov:
            # Calculate Covariance
            x = np.stack([xx@xx.T for xx in x], axis=0)

            # Normalize Covariance to abs. max. of 1
            x = np.stack([C / np.max(abs(C)) for C in x], axis=0)
            x = np.expand_dims(x, axis=-1)
        
        else:
            # normalize all time points to unit length
            x = np.stack([xx / np.linalg.norm(xx, axis=0) for xx in x], axis=0)
            # normalize each sample to max(abs()) == 1
            x = np.stack([xx / np.max(abs(xx)) for xx in x], axis=0)
            # Reshape
            x = np.swapaxes(x, 1,2)
            # x = x[:, :, :, np.newaxis]

        if return_mask:    
            # Calculate mean source activity
            y = abs(y).mean(axis=1)
            # Masking the source vector (1-> active, 0-> inactive)
            y = (y>0).astype(float)
        else:
            y = np.stack([ (yy.T / np.max(abs(yy), axis=1)).T for yy in y], axis=0)
        
        # Return same batch multiple times:
        for _ in range(batch_repetitions):
            yield (x, y)

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
        bounds=(0, 1), method='bounded', options=options, tol=tol)


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