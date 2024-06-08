
import sys; sys.path.insert(0, '../')
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import mne

from invert.forward import get_info, create_forward_model
from invert.util import pos_from_forward
from invert.evaluate import eval_mean_localization_error
from invert.simulate import generator

pp = dict(surface='inflated', hemi='both', verbose=0, cortex='low_contrast')

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
    snr_range=(-5, 5),
    amplitude_range=(0.1, 1),
    n_timecourses=200,
    n_timepoints=50,
    scale_data=False,
    add_forward_error=False,
    forward_error=0.1,
    inter_source_correlation=(0, 1),
    return_info=True,
    diffusion_parameter=0.1,
    # correlation_mode="cholesky",
    # noise_color_coeff=(0, 0.99),
    normalize_leadfield=True,
    
    random_seed=None)



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
    return torch.norm(current_data - new_data)


# def spatially_weighted_cosine_loss(pos, sigma=10.0):
#     """
#     Returns a loss function that combines cosine similarity with a spatial weighting
#     based on the positions of dipoles in the brain.
    
#     Parameters:
#     - pos: numpy array of shape (n, 3) containing the positions of each dipole.
#     - sigma: controls the spread of the spatial influence (lower value -> steeper).

#     Returns:
#     - A loss function compatible with Keras.
#     """
#     # Convert positions to a tensor and compute pairwise squared Euclidean distances
#     pos_tensor = torch.tensor(pos, dtype=tf.float32)
#     pos_diff = tf.expand_dims(pos_tensor, 0) - tf.expand_dims(pos_tensor, 1)
#     sq_dist_matrix = tf.reduce_sum(tf.square(pos_diff), axis=-1)

#     # Create a Gaussian kernel from distances
#     spatial_kernel = tf.exp(-sq_dist_matrix / (2.0 * sigma**2))

#     def loss(y_true, y_pred):
#         # Normalize y_true and y_pred to unit vectors along the last dimension
#         y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
#         y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)

#         # Compute cosine similarity for each pair in the batch
#         cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)  # Shape becomes [batch_size, n]

#         # Expand the spatial kernel and cosine similarity for broadcasting
#         expanded_spatial_kernel = tf.expand_dims(spatial_kernel, axis=0)  # Shape becomes [1, n, n]
#         expanded_cosine_sim = tf.expand_dims(cosine_sim, axis=1)  # Shape becomes [batch_size, 1, n]

#         # Apply spatial kernel
#         print(expanded_cosine_sim.shape, expanded_spatial_kernel.shape)
#         weighted_cosine_sim = expanded_cosine_sim * expanded_spatial_kernel
#         weighted_sum_cosine_sim = tf.reduce_sum(weighted_cosine_sim, axis=-1)  # Sum over last dim (n)
#         normalization = tf.reduce_sum(expanded_spatial_kernel, axis=-1)  # Sum spatial weights over n

#         # Calculate final loss by averaging over the batch and inverting the cosine similarity
#         weighted_cosine_loss = 1 - tf.reduce_mean(weighted_sum_cosine_sim / normalization)

#         return weighted_cosine_loss

#     return loss

def get_lower_triangular(C):
    ''' Get the lower triangular part of a matrix C, excluding the diagonal

    Parameters:
    -----------
    C: np.array
        The matrix to extract the lower triangular part from
    
    Returns:
    --------
    np.array
        The lower triangular part of the matrix C, excluding the diagonal
    '''
    C = np.tril(C, -1)
    C = C[np.nonzero(C)]
    return C

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


def custom_loss(distances):
    """Closure to encapsulate the distances matrix."""
    distances = torch.tensor(distances, dtype=torch.float32).cuda()

    def loss(y_true, y_pred):
        """
        Args:
        y_true: Tensor of true values with shape (batch_size, n).
        y_pred: Tensor of predicted values with shape (batch_size, n).

        Returns:
        A scalar tensor representing the loss.
        """
        # Normalize y_true and y_pred so that the maximum of each sample is 1
        # max_y_true = torch.max(y_true, dim=1, keepdim=True)
        # max_y_pred = torch.max(y_pred, dim=1, keepdim=True)

        norm_y_true = torch.norm(y_true, dim=1, keepdim=True)
        norm_y_pred = torch.norm(y_pred, dim=1, keepdim=True)

        y_true_scaled = y_true / norm_y_true
        y_pred_scaled = y_pred / norm_y_pred

        # Compute element-wise absolute differences
        E = torch.square(y_true_scaled - y_pred_scaled)  # shape (batch_size, n)
        
        # Apply the distances weighting in a quadratic form
        # Diag(E) @ distances @ Diag(E)
        # First, compute diag(E) @ distances for each example in the batch
        weighted = torch.matmul(E, distances)  # shape (batch_size, n)
        
        # Then multiply element-wise with E and sum over all elements
        error = torch.mean(weighted * E, dim=1)  # sum across each sample, shape (batch_size,)
        
        # Finally, compute the mean over the batch to get a single scalar loss
        return torch.mean(error) #+ torch.mean(E)
    
    return loss


import torch
import torch.nn as nn
import torch.optim as optim


# Assuming n_chans and other necessary variables are defined elsewhere
input_size = int((n_chans**2 - n_chans) / 2 + n_chans)
input_shape = (input_size,)  # Specify the input shape based on your data

class MyModel(nn.Module):
    def __init__(self, input_size, n_dipoles, custom_loss, distance_matrix):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 128, (1, input_size))
        self.fc1 = nn.Linear(128, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, n_dipoles)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.custom_loss = custom_loss(distance_matrix)  # Ensure custom_loss is defined in PyTorch

    def forward(self, x):
        if x.dim() != 4 or x.shape[1:] != (1, 1, input_size):
            x = x.view(-1, 1, 1, input_size)  # Ensure input is correct for Conv2d
        x = self.activation(self.conv(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the model
model1 = MyModel(input_size, n_dipoles, custom_loss, distance_matrix)

model2 = deepcopy(model1)

model1.cuda()
model2.cuda()

optimizer1 = optim.AdamW(model1.parameters(), lr=0.001)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.001)
sim_params["batch_size"] = 1024
# Training loop
gen = generator(fwd, **sim_params)  # assuming generator and fwd are defined elsewhere
for i in range(2000):
    X, y, _ = next(gen)
    covs = [get_diag_and_lower(xx.T @ xx) for xx in X]
    covs = np.stack([cov / np.abs(cov).max() for cov in covs], axis=0)
    y_true = np.stack([(yy != 0)[0, :].astype(float) for yy in y], axis=0).astype(np.float32)
    
    covs = torch.tensor(covs, dtype=torch.float32).cuda()
    y_true = torch.tensor(y_true, dtype=torch.float32).cuda()
    for j in range(10):
        output = model1(covs).cuda()
        loss = model1.custom_loss(output, y_true)
        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()  # Reset gradients for next step
    if i % 10 == 0:
        print(f"epoch {i+1} loss: {loss.item():.2f}")
        torch.save(model1.state_dict(), 'model1-weights.pt')


# # Train Model 2
# batch_size = 1024 // 5
# n_sources = np.arange(5)+1

# sim_params_temp = deepcopy(sim_params)
# sim_params_temp["batch_size"] = 1024
# sim_params_temp["n_sources"] = (1,5)
# gen_pre = generator(fwd, **sim_params_temp)
# epochs = 300
# epoch_distances = np.zeros(epochs)
# # Training loop within the RAP-MUSIC framework
# for epoch in np.arange(0, 100).astype(int):  # Number of epochs
#     print(f"epoch {epoch}")

#     # Start with a basic training
    
#     X, y, _ = gen_pre.__next__()
#     covs = [get_diag_and_lower(xx.T@xx) for xx in X]
#     covs = np.stack([cov/abs(cov).max() for cov in covs], axis=0)
#     y_true = np.stack([(yy!=0)[0,:].astype(float) for yy in y], axis=0).astype(int)
#     try:
#         for j in range(10):
#             loss = model2.train_on_batch(covs, y_true)
#             print(f"\tPretraining: {loss[0]:.3f}")
#     except:
#         pass
#     # print(f"\tPretraining: {loss[0]:.3f}")
    
#     X_train = []
#     Y_train = []
#     for n_source in n_sources:
#         # print(f"\ttraining for {n_source} sources")
#         sim_params["batch_size"] = batch_size #// n_source
#         sim_params["n_sources"] = (n_source, n_source)
#         gen_post = generator(fwd, **sim_params)
#         X, true_dipoles, Y = generate_initial_data(gen_post) 
#         current_data = deepcopy(X)
#         n_samples = len(true_dipoles)
#         estimated_dipole_idc = [list() for _ in range(n_samples)]

#         for i_iter in range(n_source):
#             # Compute Covariances
#             current_covs = [get_diag_and_lower(xx@xx.T) for xx in current_data]
#             current_covs = np.stack([cov/abs(cov).max() for cov in current_covs], axis=0)
#             X_train.append(current_covs)
#             # Predict the sources using the model
#             estimated_sources = model2.predict(current_covs, verbose=0)

#             estimated_sources_temp = estimated_sources.copy()
#             for i_sample in range(n_samples):
#                 estimated_sources_temp[i_sample, estimated_dipole_idc[i_sample]] = 0

#             new_dipole_idc = np.argmax(estimated_sources_temp, axis=1)  # Convert to dipole indices
            
#             for i_idx, new_idx in enumerate(new_dipole_idc):
#                 estimated_dipole_idc[i_idx].append(new_idx)
            
#             Y_train.append((Y!=0).astype(int)[:, :, 0])
#             # Outproject the dipoles from the respective data
#             current_data = wrap_outproject_from_data(X, leadfield, estimated_dipole_idc, alpha=0)
            
#     # Adjust parameters
#     X_train = np.concatenate(X_train, axis=0)
#     Y_train = np.concatenate(Y_train, axis=0)
#     # print(model2.test_on_batch(X_train, Y_train))
#     try:
#         for _ in range(10):
#             loss = model2.train_on_batch(X_train, Y_train)
#             print(f"\tRAP-Training: {np.mean(loss[0]):.3f}")
#     except:
#         pass
#     # print(f"\tRAP-Training: {np.mean(loss[0]):.3f}")
#     # Save the model
#     if epoch % 10 == 0:
#         model2.save('.rap-weights.keras')
    