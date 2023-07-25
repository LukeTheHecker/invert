from copy import deepcopy
import numpy as np
import colorednoise as cn
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix, vstack
import mne
import pandas as pd


def generator(fwd, use_cov=True, batch_size=1284, batch_repetitions=30, n_sources=10, 
              n_orders=2, amplitude_range=(0.001,1), n_timepoints=20, 
              snr_range=(1, 100), n_timecourses=5000, beta_range=(0, 3),
              return_mask=True, scale_data=True, return_info=False,
              add_forward_error=False, forward_error=0.1, remove_channel_dim=False, 
              inter_source_correlation=0.5, diffusion_smoothing=True, 
              diffusion_parameter=0.1, fixed_covariance=False, iid_noise=False, verbose=0):
    """
    Parameters
    ----------
    fwd : object
        Forward solution object containing the source space and orientation information.
    use_cov : bool
        If True, a covariance matrix is used in the simulation. Default is True.
    batch_size : int
        Size of each batch of simulations. Default is 1284.
    batch_repetitions : int
        Number of repetitions of each batch. Default is 30.
    n_sources : int
        Number of sources in the brain from which activity is simulated. Default is 10.
    n_orders : int
        The order of the model used to generate time courses. Default is 2.
    amplitude_range : tuple
        Range of possible amplitudes for the simulated sources. Default is (0.001,1).
    n_timepoints : int
        Number of timepoints in each simulated time course. Default is 20.
    snr_range : tuple
        Range of signal to noise ratios to be used in the simulations. Default is (1, 100).
    n_timecourses : int
        Number of unique time courses to simulate. Default is 5000.
    beta_range : tuple
        Range of possible power-law exponents for the power spectral density of the simulated sources. Default is (0, 3).
    return_mask : bool
        If True, the function will also return a mask of the sources used. Default is True.
    scale_data : bool
        If True, the EEG data will be scaled. Default is True.
    return_info : bool
        If True, the function will return a dictionary with information about the generated data. Default is False.
    add_forward_error : bool
        If True, the function will add an error to the forward model. Default is False.
    forward_error : float
        Amount of error to add to the forward model if 'add_forward_error' is True. Default is 0.1.
    remove_channel_dim : bool
        If True, the channel dimension will be removed from the output data. Default is False.
    inter_source_correlation : float|Tuple
        The level of correlation between different sources. Default is 0.5.
    diffusion_smoothing : bool
        Whether to use diffusion smoothing. Default is True.
    diffusion_parameter : float
        The diffusion parameter (alpha). Default is 0.1.
    iid_noise : bool
        If True: use independently distributed noise
        if False: use correlated noise
    verbose : int
        Level of verbosity for the function. Default is 0.
    
    Return
    ------
    x : numpy.ndarray
        The EEG data matrix.
    y : numpy.ndarray
        The source data matrix.
    """
    leadfield = deepcopy(fwd["sol"]["data"])
    leadfield_original = deepcopy(fwd["sol"]["data"])
    n_chans, n_dipoles = leadfield.shape

    if isinstance(n_sources, (int, float)):
        n_sources = [n_sources, n_sources]
    min_sources, max_sources = n_sources

    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose)
    # Convert to sparse matrix for speedup
    adjacency = csr_matrix(adjacency)
    if diffusion_smoothing:
        print("Using Diffusion Smoothing on Graph")
        gradient = np.identity(n_dipoles) - diffusion_parameter*laplacian(adjacency)
    else:
        gradient = abs(laplacian(adjacency))
    
    # Convert to sparse matrix for speedup
    gradient = csr_matrix(gradient)

    del adjacency
    
    
    leadfield -= leadfield.mean()
    # Normalize columns of the leadfield
    leadfield /= np.linalg.norm(leadfield, axis=0)

    


    sources = csr_matrix(np.identity(n_dipoles))
    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = n_orders
    else:
        min_order = 0
        max_order = n_orders

    if isinstance(inter_source_correlation, (tuple, list)):
        get_inter_source_correlation = lambda n=1: np.random.uniform(inter_source_correlation[0], inter_source_correlation[1], n)
    else:
        get_inter_source_correlation = lambda n=1: np.random.uniform(inter_source_correlation, inter_source_correlation, n)

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
        n_sources_batch = np.random.randint(min_sources, max_sources+1, batch_size)
        selection = [np.random.randint(0, n_candidates, n) for n in n_sources_batch]

        # Assign each source (or source patch) a time course
        # amplitude_values = [np.random.uniform(*amplitude_range, n) for n in n_sources_batch]
        # amplitudes = [time_courses[np.random.choice(n_timecourses, n)].T * amplitude_values[i] for i, n in enumerate(n_sources_batch)]

        amplitude_values = [np.random.uniform(*amplitude_range, n) for n in n_sources_batch]
        amplitudes = [time_courses[np.random.choice(n_timecourses, n)].T for i, n in enumerate(n_sources_batch)]

        inter_source_correlations = get_inter_source_correlation(n=batch_size)
        source_covariances = [get_cov(n, isc) for n, isc in zip(n_sources_batch, inter_source_correlations)]
        amplitudes = [amp @ np.diag(amplitude_values[i]) @ cov for i, (amp, cov) in enumerate(zip(amplitudes, source_covariances))]



        # print(np.stack(amplitudes, axis=0).shape)

        # y = np.stack([(amplitudes[i] @ sources.toarray()[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)
        y = np.stack([(amplitudes[i] @ sources[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)

        # Project simulated sources through leadfield
        x = np.stack([leadfield @ yy.T for yy in y], axis=0)

        # Add white noise to clean EEG
        snr_levels = np.random.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        # print("iid in generator is: ", iid_noise)
        x = np.stack([add_white_noise(xx, snr_level, iid=iid_noise) for (xx, snr_level) in zip(x, snr_levels)], axis=0)


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
            if not remove_channel_dim:
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
            # (1) binary
            # Calculate mean source activity
            y = abs(y).mean(axis=1)
            # Masking the source vector (1-> active, 0-> inactive)
            y = (y>0).astype(float)
            
            # (2.1) Source Covariance
            # The diagonal of the source covariance matrix:
            # y = np.stack([np.diagonal(yy.T@yy) for yy in y], axis=0)

            # (2.2) Source Covariance (efficient)
            # y = np.mean(y**2, axis=1)
            # y = np.stack([yy / np.max(abs(yy)) for yy in y], axis=0)
        
        else:
            if scale_data:
                y = np.stack([ (yy.T / np.max(abs(yy), axis=1)).T for yy in y], axis=0)
            

        
        # Return same batch multiple times:
        if return_info:
            info = pd.DataFrame(dict(n_sources=n_sources_batch, amplitudes=amplitude_values, snr=snr_levels, inter_source_correlations=inter_source_correlations))
            output = (x, y, info)
        else:
            output = (x, y)

        for _ in range(batch_repetitions):
            yield output

def get_cov(n, corr_coef):
    '''Generate a random covariance matrix  that is symmetric along the
    diagonal.'''
    cov = np.ones((n,n)) * corr_coef + np.eye(n)*(1-corr_coef)
    return np.linalg.cholesky(cov)

def add_white_noise(X_clean, snr, iid=False):
    ''' '''
    X_noise = np.random.randn(*X_clean.shape)
    if not iid:
        # print("iid in function is: ", iid)
        # Inter-channel correlations
        coeff_mat = np.random.rand(X_clean.shape[0], X_clean.shape[0])
        np.fill_diagonal(coeff_mat, 1)
        
        # Make positive semi-definite
        coeff_mat = np.dot(coeff_mat, coeff_mat.T)

        # Make matrix symmetric
        coeff_mat = (coeff_mat + coeff_mat.T)/2
        # print(coeff_mat)
        # Random partially correlated noise
        X_noise = np.linalg.cholesky( coeff_mat ) @ X_noise
    
    rms_noise = rms(X_noise)
    rms_signal = rms(X_clean)

    scaler = rms_signal / (snr * rms_noise)

    X_full = X_clean + X_noise*scaler
    X_full -= X_full.mean(axis=0)
    return X_full

def add_error(leadfield, forward_error, gradient):
    n_chans, n_dipoles = leadfield.shape
    noise = np.random.uniform(-1, 1, (n_chans, n_dipoles)) @ gradient
    leadfield_mix = leadfield / np.linalg.norm(leadfield) + forward_error * noise / np.linalg.norm(noise)
    return leadfield_mix

def rms(x):
    return np.sqrt(np.mean(x**2))
    

