import sys; sys.path.insert(0, '../') 
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import mne

from invert.forward import get_info, create_forward_model
from invert.util import pos_from_forward
from invert.evaluate import eval_mean_localization_error
from invert.simulate import generator

sampling = "ico2"
info = get_info(kind='biosemi32')
fwd = create_forward_model(info=info, sampling=sampling)
fwd["sol"]["data"] /= np.linalg.norm(fwd["sol"]["data"], axis=0) 
pos = pos_from_forward(fwd)
leadfield = fwd["sol"]["data"]
n_chans, n_dipoles = leadfield.shape

source_model = fwd['src']
vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=0)
distance_matrix = cdist(pos, pos)




sim_params = dict(
    use_cov=False,
    return_mask=False,
    batch_repetitions=1,
    batch_size=1,
    n_sources=(1, 10),
    n_orders=(0, 0),
    snr_range=(0.2, 10),
    amplitude_range=(0.1, 1),
    n_timecourses=200,
    n_timepoints=(5, 50),
    scale_data=False,
    add_forward_error=False,
    forward_error=0.1,
    inter_source_correlation=(0, 1),
    return_info=True,
    diffusion_parameter=0.1,
    correlation_mode="cholesky",
    noise_color_coeff=(0, 0.99),
    
    random_seed=None)

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers

# Assuming we have a function to generate initial EEG data and true dipoles
def generate_initial_data(gen):
    # This function should return initial EEG data
    # and the true dipole parameters that generated the data.

    # Generate random dipole parameters
    x, y, _ = gen.__next__()
    x = np.swapaxes(x, 1, 2)
    y = np.swapaxes(y, 1, 2)
    true_indices = [np.where(yy[:, 0]!=0)[0] for yy in y]
    return x, true_indices, y

# def outproject_from_data(data, leadfield, idc):
#     L = leadfield[:, idc]
#     # Y_est = L.T @ np.linalg.pinv(L @ L.T + np.identity(L.shape[0])*0.1) @ data
#     # or simply:
#     Y_est = np.linalg.pinv(L) @ data
#     return data - L@Y_est
#     # return L@Y_est - data

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

def wrap_outproject_from_data(current_data, leadfield, estimated_dipole_idc, alpha=0.1):
    # Wrapper function to outproject dipoles from the data
    n_samples = current_data.shape[0]
    new_data = np.zeros_like(current_data)
    for i in range(n_samples):
        new_data[i] = outproject_from_data(current_data[i], leadfield, np.array(estimated_dipole_idc[i]), alpha=alpha)
    return new_data

def predict(model, current_covs):
    # Predict source estimate

    # Predict the sources using the model
    estimated_sources = model.predict(current_covs)  # Model's prediction
    return estimated_sources
    
    # return new_data, estimated_dipole_idc

# Function to compute residuals or stopping condition
def compute_residual(current_data, new_data):
    # Placeholder function to compute residual to decide when to stop the iteration
    return tf.norm(current_data - new_data)

from scipy.optimize import linear_sum_assignment
import tensorflow as tf

def spatially_weighted_cosine_loss(pos, sigma=10.0):
    """
    Returns a loss function that combines cosine similarity with a spatial weighting
    based on the positions of dipoles in the brain.
    
    Parameters:
    - pos: numpy array of shape (n, 3) containing the positions of each dipole.
    - sigma: controls the spread of the spatial influence (lower value -> steeper).

    Returns:
    - A loss function compatible with Keras.
    """
    # Convert positions to a tensor and compute pairwise squared Euclidean distances
    pos_tensor = tf.constant(pos, dtype=tf.float32)
    pos_diff = tf.expand_dims(pos_tensor, 0) - tf.expand_dims(pos_tensor, 1)
    sq_dist_matrix = tf.reduce_sum(tf.square(pos_diff), axis=-1)

    # Create a Gaussian kernel from distances
    spatial_kernel = tf.exp(-sq_dist_matrix / (2.0 * sigma**2))

    def loss(y_true, y_pred):
        # Normalize y_true and y_pred to unit vectors along the last dimension
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)

        # Compute cosine similarity for each pair in the batch
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)  # Shape becomes [batch_size, n]

        # Expand the spatial kernel and cosine similarity for broadcasting
        expanded_spatial_kernel = tf.expand_dims(spatial_kernel, axis=0)  # Shape becomes [1, n, n]
        expanded_cosine_sim = tf.expand_dims(cosine_sim, axis=1)  # Shape becomes [batch_size, 1, n]

        # Apply spatial kernel
        print(expanded_cosine_sim.shape, expanded_spatial_kernel.shape)
        weighted_cosine_sim = expanded_cosine_sim * expanded_spatial_kernel
        weighted_sum_cosine_sim = tf.reduce_sum(weighted_cosine_sim, axis=-1)  # Sum over last dim (n)
        normalization = tf.reduce_sum(expanded_spatial_kernel, axis=-1)  # Sum spatial weights over n

        # Calculate final loss by averaging over the batch and inverting the cosine similarity
        weighted_cosine_loss = 1 - tf.reduce_mean(weighted_sum_cosine_sim / normalization)

        return weighted_cosine_loss

    return loss



def custom_loss(distances, scaler=1):
    distances = tf.constant(distances, dtype=tf.float32)  # Ensure distances is a tensor

    def loss(y_true, y_pred):
        # Normalize each sample in the batch
        y_true_norm = y_true / tf.reduce_max(tf.abs(y_true), axis=1, keepdims=True)
        y_pred_norm = y_pred / tf.reduce_max(tf.abs(y_pred), axis=1, keepdims=True)
        # Calculate the absolute differences
        # diff = tf.abs(y_true_norm - y_pred_norm)
        diff = tf.square(y_true_norm - y_pred_norm)
        
        
        # Perform element-wise multiplication with distances
        weighted_diff = tf.reduce_mean( tf.matmul(tf.matmul(diff, distances),  tf.transpose(diff)))
        # Compute the mean across the batch
        error = tf.reduce_mean(weighted_diff)# + tf.reduce_mean(diff)
        return error * scaler

    return loss

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the neural network architecture
input_shape = (n_chans, n_chans, 1)  # Specify the input shape based on your data
model = keras.Sequential([
    layers.Conv2D(n_chans, (1, n_chans), 
          activation="tanh", padding="valid",
          input_shape=input_shape,
          name='CNN1'),
    layers.Flatten(),
    layers.Dense(100, activation='tanh'),
    layers.Dense(n_dipoles, activation='sigmoid')
])



# Compile the model
model.compile(optimizer='adam', loss='cosine_similarity', metrics=['accuracy'])  # Specify the loss function and optimizer
model.build()
model.summary()
model.load_weights('.rap-weights.keras')

from scipy.optimize import linear_sum_assignment
from copy import deepcopy

batch_size = 1024
n_sources = np.arange(5)+1

epochs = 1000
# Training loop within the RAP-MUSIC framework
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch}")
    X_train = []
    Y_train = []
    for n_source in n_sources:
        sim_params["batch_size"] = batch_size // n_source
        sim_params["n_sources"] = (n_source, n_source)
        gen = generator(fwd, **sim_params)
        X, true_dipoles, Y = generate_initial_data(gen) 
        current_data = deepcopy(X)
        n_samples = len(true_dipoles)
        estimated_dipole_idc = [list() for _ in range(n_samples)]

        for i_iter in range(n_source):
            current_covs = np.stack([x @ x.T for x in current_data], axis=0)
            current_covs = np.stack([cov / abs(cov).max() for cov in current_covs], axis=0)
            X_train.append(current_covs)
            estimated_sources = model.predict(current_covs, verbose=0)
            estimated_sources_temp = estimated_sources.copy()
            for i_sample in range(n_samples):
                estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

            new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)
            
            for i_idx, new_idx in enumerate(new_dipole_idc):
                estimated_dipole_idc[i_idx].append(new_idx)

            true_data_matched = np.zeros((n_samples, n_dipoles))
            avg_dists = []
            for i_sample in range(n_samples):
                true_data_matched[i_sample, true_dipoles[i_sample]] = 1

            Y_train.append(true_data_matched)
            current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc)

    for _ in range(5):
        try:
            loss = model.train_on_batch(np.concatenate(X_train, axis=0), np.concatenate(Y_train, axis=0))
            print(f"Loss: {np.mean(loss[0]):.3f}, {np.mean(loss[1]):.3f}")
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    model.save('.rap-weights.keras')
    print(f"Saved model at epoch {epoch}")

last_epoch = epoch
epochs = 1000
# Training loop within the RAP-MUSIC framework
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch+last_epoch}")
    X_train = []
    Y_train = []
    for n_source in n_sources:
        sim_params["batch_size"] = batch_size // n_source
        sim_params["n_sources"] = (n_source, n_source)
        gen = generator(fwd, **sim_params)
        X, true_dipoles, Y = generate_initial_data(gen) 
        current_data = deepcopy(X)
        n_samples = len(true_dipoles)
        estimated_dipole_idc = [list() for _ in range(n_samples)]

        for i_iter in range(n_source):
            current_covs = np.stack([x @ x.T for x in current_data], axis=0)
            current_covs = np.stack([cov / abs(cov).max() for cov in current_covs], axis=0)
            X_train.append(current_covs)
            estimated_sources = model.predict(current_covs, verbose=0)
            estimated_sources_temp = estimated_sources.copy()
            for i_sample in range(n_samples):
                estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

            new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)
            
            for i_idx, new_idx in enumerate(new_dipole_idc):
                estimated_dipole_idc[i_idx].append(new_idx)

            true_data_matched = np.zeros((n_samples, n_dipoles))
            avg_dists = []
            for i_sample in range(n_samples):
                true_data_matched[i_sample, true_dipoles[i_sample]] = 1

            Y_train.append(true_data_matched)
            current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc)

    for _ in range(5):
        try:
            loss = model.train_on_batch(np.concatenate(X_train, axis=0), np.concatenate(Y_train, axis=0))
            print(f"Loss: {np.mean(loss[0]):.3f}, {np.mean(loss[1]):.3f}")
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    model.save('.rap-weights.keras')
    logging.info(f"Saved model at epoch {epoch}")


last_epoch = epoch
epochs = 1000
# Training loop within the RAP-MUSIC framework
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch+last_epoch}")
    X_train = []
    Y_train = []
    for n_source in n_sources:
        sim_params["batch_size"] = batch_size // n_source
        sim_params["n_sources"] = (n_source, n_source)
        gen = generator(fwd, **sim_params)
        X, true_dipoles, Y = generate_initial_data(gen) 
        current_data = deepcopy(X)
        n_samples = len(true_dipoles)
        estimated_dipole_idc = [list() for _ in range(n_samples)]

        for i_iter in range(n_source):
            current_covs = np.stack([x @ x.T for x in current_data], axis=0)
            current_covs = np.stack([cov / abs(cov).max() for cov in current_covs], axis=0)
            X_train.append(current_covs)
            estimated_sources = model.predict(current_covs, verbose=0)
            estimated_sources_temp = estimated_sources.copy()
            for i_sample in range(n_samples):
                estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

            new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)
            
            for i_idx, new_idx in enumerate(new_dipole_idc):
                estimated_dipole_idc[i_idx].append(new_idx)

            true_data_matched = np.zeros((n_samples, n_dipoles))
            avg_dists = []
            for i_sample in range(n_samples):
                true_data_matched[i_sample, true_dipoles[i_sample]] = 1

            Y_train.append(true_data_matched)
            current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc)

    for _ in range(5):
        try:
            loss = model.train_on_batch(np.concatenate(X_train, axis=0), np.concatenate(Y_train, axis=0))
            print(f"Loss: {np.mean(loss[0]):.3f}, {np.mean(loss[1]):.3f}")
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    model.save('.rap-weights.keras')
    logging.info(f"Saved model at epoch {epoch}")

last_epoch = epoch
epochs = 1000
# Training loop within the RAP-MUSIC framework
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch+last_epoch}")
    X_train = []
    Y_train = []
    for n_source in n_sources:
        sim_params["batch_size"] = batch_size // n_source
        sim_params["n_sources"] = (n_source, n_source)
        gen = generator(fwd, **sim_params)
        X, true_dipoles, Y = generate_initial_data(gen) 
        current_data = deepcopy(X)
        n_samples = len(true_dipoles)
        estimated_dipole_idc = [list() for _ in range(n_samples)]

        for i_iter in range(n_source):
            current_covs = np.stack([x @ x.T for x in current_data], axis=0)
            current_covs = np.stack([cov / abs(cov).max() for cov in current_covs], axis=0)
            X_train.append(current_covs)
            estimated_sources = model.predict(current_covs, verbose=0)
            estimated_sources_temp = estimated_sources.copy()
            for i_sample in range(n_samples):
                estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

            new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)
            
            for i_idx, new_idx in enumerate(new_dipole_idc):
                estimated_dipole_idc[i_idx].append(new_idx)

            true_data_matched = np.zeros((n_samples, n_dipoles))
            avg_dists = []
            for i_sample in range(n_samples):
                true_data_matched[i_sample, true_dipoles[i_sample]] = 1

            Y_train.append(true_data_matched)
            current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc)

    for _ in range(5):
        try:
            loss = model.train_on_batch(np.concatenate(X_train, axis=0), np.concatenate(Y_train, axis=0))
            print(f"Loss: {np.mean(loss[0]):.3f}, {np.mean(loss[1]):.3f}")
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    model.save('.rap-weights.keras')
    logging.info(f"Saved model at epoch {epoch}")

last_epoch = epoch
epochs = 1000
# Training loop within the RAP-MUSIC framework
for epoch in range(epochs):  # Number of epochs
    print(f"Epoch {epoch+last_epoch}")
    X_train = []
    Y_train = []
    for n_source in n_sources:
        sim_params["batch_size"] = batch_size // n_source
        sim_params["n_sources"] = (n_source, n_source)
        gen = generator(fwd, **sim_params)
        X, true_dipoles, Y = generate_initial_data(gen) 
        current_data = deepcopy(X)
        n_samples = len(true_dipoles)
        estimated_dipole_idc = [list() for _ in range(n_samples)]

        for i_iter in range(n_source):
            current_covs = np.stack([x @ x.T for x in current_data], axis=0)
            current_covs = np.stack([cov / abs(cov).max() for cov in current_covs], axis=0)
            X_train.append(current_covs)
            estimated_sources = model.predict(current_covs, verbose=0)
            estimated_sources_temp = estimated_sources.copy()
            for i_sample in range(n_samples):
                estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

            new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)
            
            for i_idx, new_idx in enumerate(new_dipole_idc):
                estimated_dipole_idc[i_idx].append(new_idx)

            true_data_matched = np.zeros((n_samples, n_dipoles))
            avg_dists = []
            for i_sample in range(n_samples):
                true_data_matched[i_sample, true_dipoles[i_sample]] = 1

            Y_train.append(true_data_matched)
            current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc)

    for _ in range(5):
        try:
            loss = model.train_on_batch(np.concatenate(X_train, axis=0), np.concatenate(Y_train, axis=0))
            print(f"Loss: {np.mean(loss[0]):.3f}, {np.mean(loss[1]):.3f}")
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    # model.save('.rap-weights.keras')
    print(f"Saved model at epoch {epoch}")