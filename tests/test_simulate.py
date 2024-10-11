import sys; sys.path.insert(0, '../')

import numpy as np
from scipy.sparse import csr_matrix
from invert.simulate.simulate import generator

# Test case 1: Default parameters
from invert.forward import get_info, create_forward_model
pp = dict(surface='white', hemi='both', verbose=0)


# Parameters
sampling = 'ico1'
montage = "biosemi16"
alpha = 0.1
epochs = 1

# Forward Model
info = get_info(kind=montage)
fwd = create_forward_model(info=info, sampling=sampling)
n_chans, n_dipoles = fwd["sol"]["data"].shape
print(n_chans, n_dipoles)

x, y = generator(fwd).__next__()
assert isinstance(x, np.ndarray)
assert isinstance(y, np.ndarray)
assert x.shape == (1284, n_chans, n_chans, 1)
assert y.shape == (1284, n_dipoles)
print(x.shape, y.shape)

# Test case 2: Custom parameters
params = {
    "use_cov": False,
    "batch_size": 100,
    "batch_repetitions": 1,
    "n_sources": 5,
    "n_orders": 3,
    "amplitude_range": (0.1, 0.5),
    "n_timepoints": 50,
    "snr_range": (10, 20),
    "n_timecourses": 1000,
    "beta_range": (1, 2),
    "return_mask": False,
    "scale_data": False,
    "return_info": False,
    "add_forward_error": True,
    "forward_error": 0.2,
    "remove_channel_dim": True,
    "inter_source_correlation": 0.8,
    "diffusion_smoothing": False,
    "diffusion_parameter": 0.5,
    "correlation_mode": "banded",
    "noise_color_coeff": 0.8,
    "random_seed": 123,
    "verbose": 1
}
x, y = generator(fwd, **params).__next__()
assert isinstance(x, np.ndarray)
assert isinstance(y, np.ndarray)
assert x.shape == (params["batch_size"], params["n_timepoints"], n_chans)
assert y.shape == (params["batch_size"], params["n_timepoints"], n_dipoles)


print("All test cases passed!")