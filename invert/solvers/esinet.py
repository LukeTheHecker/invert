from copy import deepcopy
from esinet import Simulation, Net
from .base import BaseSolver, InverseOperator
from scipy.sparse.csgraph import laplacian
import mne
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Reshape, AveragePooling2D
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
                validation_data=self.generator_val.__next__(), callbacks=callbacks)


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
        gen_args = dict(batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
        self.generator = generator(self.forward, "standard", **gen_args)
        
        gen_args["batch_size"] = self.size_validation_set
        self.generator_val = generator(self.forward, "standard", **gen_args)


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
        
        # Eigenvalue-based Covariance
        # # D, U = np.linalg.eig(C)
        # U, D, _ = np.linalg.svd(y)
        # D_ = D/D.max()
        # n = np.where( abs(np.diff(D_)) < 0.001 )[0][0] 
        # print("Eigenvalues: ", n)
        # U_n = np.zeros(U.shape)
        # U_n[:, :n] = U[:, :n]
        # U_n -= U_n.mean(axis=0)
        # y_proj = U_n @ np.diag(D) @  y
        # y_proj -= y_proj.mean(axis=0)
        # C = y_proj @ y_proj.T
        # C /= abs(C).max()

        # Add empty batch and (color-) channel dimension
        C = C[np.newaxis, :, :, np.newaxis]
        gammas = self.model.predict(C, verbose=self.verbose)[0]
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
        gen_args = dict(batch_size=self.batch_size, batch_repetitions=self.batch_repetitions, 
                n_sources=self.n_sources, n_orders=self.n_orders, n_timepoints=self.n_timepoints,
                snr_range=self.snr_range)
        self.generator = generator(self.forward, "cov", **gen_args)
        

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
    
def generator(fwd, method, batch_size=1284, batch_repetitions=30, n_sources=2, n_orders=3, amplitude_range=(0.001,1), n_timepoints=20, snr_range=(1, 100), verbose=0):
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

    time_courses = np.stack([np.random.randn(n_timepoints) for _ in range(1000)], axis=0)

    

    n_candidates = sources.shape[0]

    while True:
        # select sources
        n_sources_batch = np.random.randint(1, n_sources+1, batch_size)
        selection = [np.random.randint(0, n_candidates, n) for n in n_sources_batch]

        # Give them a time course
        amplitudes = [time_courses[np.random.choice(np.arange(time_courses.shape[0]), n)].T * np.random.uniform(*amplitude_range, n) for n in n_sources_batch]
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

        if method == "cov":
            # Calculate Covariance
            x = np.stack([xx@xx.T for xx in x], axis=0)

            # Normalize Covariance to abs. max. of 1
            x = np.stack([C / np.max(abs(C)) for C in x], axis=0)
            x = np.expand_dims(x, axis=-1)
        elif method == "standard":
            x = np.stack([xx / np.linalg.norm(xx, axis=0) for xx in x], axis=0)
            x = np.stack([xx / np.max(abs(xx)) for xx in x], axis=0)
            x = np.swapaxes(x, 1,2)
            x = x[:, :, :, np.newaxis]
            
        # Calculate mean source activity
        y = abs(y).mean(axis=1)
        # Masking the source vector (1-> active, 0-> inactive)
        y = (y>0).astype(float)
        
        # Return same batch multiple times:
        for _ in range(batch_repetitions):
            yield (x, y)
