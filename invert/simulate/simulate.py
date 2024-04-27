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
              diffusion_parameter=0.1, correlation_mode=None, 
              noise_color_coeff=0.5, random_seed=None, verbose=0):
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
    correlation_mode : None/str
        correlation_mode : None/str
        None implies no correlation between the noise in different channels.
        'bounded' : Colored bounded noise, where channels closer to each other will be more correlated.
        'diagonal' : Some channels have varying degrees of noise.
    noise_color_coeff : float
        The magnitude of spatial coloring of the noise.
    random_seed : None / int
        The random seed for replicable simulations
    verbose : int
        Level of verbosity for the function. Default is 0.
    
    Return
    ------
    x : numpy.ndarray
        The EEG data matrix.
    y : numpy.ndarray
        The source data matrix.
    """
    rng = np.random.default_rng(random_seed)
    leadfield = deepcopy(fwd["sol"]["data"])
    leadfield_original = deepcopy(fwd["sol"]["data"])
    n_chans, n_dipoles = leadfield.shape

    if isinstance(n_sources, (int, float)):
        n_sources = [np.clip(n_sources, a_min=1, a_max=np.inf), n_sources]
    min_sources, max_sources = n_sources

    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose)
    # Convert to sparse matrix for speedup
    adjacency = csr_matrix(adjacency)
    if diffusion_smoothing:
        # print("Using Diffusion Smoothing on Graph")
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
        if min_order == max_order:
            max_order += 1
    else:
        min_order = 0
        max_order = n_orders

    if isinstance(inter_source_correlation, (tuple, list)):
        get_inter_source_correlation = lambda n=1: rng.uniform(inter_source_correlation[0], inter_source_correlation[1], n)
    else:
        get_inter_source_correlation = lambda n=1: rng.uniform(inter_source_correlation, inter_source_correlation, n)

    if isinstance(noise_color_coeff, (tuple, list)):
        get_noise_color_coeff = lambda n=1: rng.uniform(noise_color_coeff[0], noise_color_coeff[1], n)
    else:
        get_noise_color_coeff = lambda n=1: rng.uniform(noise_color_coeff, noise_color_coeff, n)


    for i in range(1, max_order):
        # new_sources = sources[-n_dipoles:, -n_dipoles:] @ gradient
        # new_sources /= new_sources.max(axis=0)
        # sources = np.concatenate( [sources, new_sources], axis=0 )
        
        # New Smoothness order
        new_sources = csr_matrix(sources.toarray()[-n_dipoles:, -n_dipoles:]) @ gradient
        
        # Scaling
        row_maxes = new_sources.max(axis=0).toarray().flatten()
        new_sources = new_sources / row_maxes[np.newaxis]

        sources = vstack([sources, new_sources])
    # print(sources.shape)
    if min_order > 0:
        start = int(min_order*n_dipoles)
        sources = sources.toarray()[start:, :]
    
    sources = csr_matrix(sources)
    # Pre-compute random time courses
    betas = rng.uniform(*beta_range,n_timecourses)
    time_courses = np.stack([cn.powerlaw_psd_gaussian(beta, n_timepoints) for beta in betas], axis=0)
    
    # Normalize time course to max(abs()) == 1
    time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T

    n_candidates = sources.shape[0]
    # print(sources.shape, n_orders)
    while True:
        if add_forward_error:
            leadfield = add_error(leadfield_original, forward_error, gradient, rng)
        # print("yeet")
        # select sources or source patches
        n_sources_batch = rng.integers(min_sources, max_sources+1, batch_size)
        selection = [rng.integers(0, n_candidates, n) for n in n_sources_batch]

        # Assign each source (or source patch) a time course
        # amplitude_values = [rng.uniform(*amplitude_range, n) for n in n_sources_batch]
        # amplitudes = [time_courses[rng.choice(n_timecourses, n)].T * amplitude_values[i] for i, n in enumerate(n_sources_batch)]

        amplitude_values = [rng.uniform(*amplitude_range, n) for n in n_sources_batch]
        choices = [rng.choice(n_timecourses, n) for n in n_sources_batch]
        # print(choices)
        amplitudes = [time_courses[choice].T for choice in choices]
        from scipy.stats import pearsonr
        # print(len(amplitudes), amplitudes[0].shape)
        # print("Initial corr between two timecourses: ", pearsonr(amplitudes[0][0], amplitudes[0][1])[0])
        inter_source_correlations = get_inter_source_correlation(n=batch_size)
        noise_color_coeffs = get_noise_color_coeff(n=batch_size)
        source_covariances = [get_cov(n, isc) for n, isc in zip(n_sources_batch, inter_source_correlations)]
        amplitudes = [amp @ np.diag(amplitude_values[i]) @ cov for i, (amp, cov) in enumerate(zip(amplitudes, source_covariances))]

        # print("All the data: \n")
        # print("n_sources_batch: ", n_sources_batch, np.min(n_sources_batch), np.max(n_sources_batch))
        # print("selection: ", selection)
        # print("amplitude_values: ", amplitude_values)
        # print("inter_source_correlations: ", inter_source_correlations)
        

        # print(np.stack(amplitudes, axis=0).shape)

        # y = np.stack([(amplitudes[i] @ sources.toarray()[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)
        y = np.stack([(amplitudes[i] @ sources[selection[i]]) / len(amplitudes[i]) for i in range(batch_size)], axis=0)

        # Project simulated sources through leadfield
        x = np.stack([leadfield @ yy.T for yy in y], axis=0)
        # Add white noise to clean EEG
        snr_levels = rng.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        x = np.stack([add_white_noise(xx, snr_level, rng, correlation_mode=correlation_mode, noise_color_coeff=noise_color_level) for (xx, snr_level, noise_color_level) in zip(x, snr_levels, noise_color_coeffs)], axis=0)

        # Apply common average reference
        # x = np.stack([xx - xx.mean(axis=0) for xx in x], axis=0)
        
        if use_cov:
            # Calculate Covariance
            x = np.stack([xx@xx.T for xx in x], axis=0)
            
            # Normalize Covariance to abs. max. of 1
            x = np.stack([C / np.max(abs(C)) for C in x], axis=0)
            # x = np.stack([C / (np.trace(C) / C.shape[0]) for C in x], axis=0)

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
        
        if return_info:
            info = pd.DataFrame(dict(
                n_sources=n_sources_batch, 
                amplitudes=amplitude_values, 
                snr=snr_levels, 
                inter_source_correlations=inter_source_correlations, 
                n_orders=[[min_order, max_order],]*batch_size,
                diffusion_parameter=[diffusion_parameter,] * batch_size,
                n_timepoints=[n_timepoints,] * batch_size,
                n_timecourses=[n_timecourses,] * batch_size,
                correlation_mode=[correlation_mode,] * batch_size,
                noise_color_coeff=[noise_color_coeff,] * batch_size,
                ))
            output = (x, y, info)
        else:
            output = (x, y)

        for _ in range(batch_repetitions):
            yield output

def generator_simple(fwd, batch_size, corrs, T, n_sources, SNR_range, 
                     random_seed=42, return_info=True):
        rng = np.random.default_rng(random_seed)
        leadfield = deepcopy(fwd["sol"]["data"])
        leadfield /= leadfield.std(axis=0)
        n_channels, n_dipoles = leadfield.shape

        while True:
            sim_info = list()
            X = np.zeros((batch_size, n_channels, T))
            y = np.zeros((batch_size, n_dipoles, T))
            corrs_batch = rng.uniform(corrs[0], corrs[1], batch_size)
            SNR_batch = rng.uniform(SNR_range[0], SNR_range[1], batch_size)
            indices = [rng.choice(fwd["sol"]["data"].shape[1], n_sources) for _ in range(batch_size)]
            
            for i in range(batch_size):
                # print(leadfield.shape, corrs_batch[i], T, n_sources, indices[i], SNR_batch[i], random_seed)
                X[i], y[i] = generator_single_simple(leadfield, corrs_batch[i], T, n_sources, indices[i], SNR_batch[i], random_seed=random_seed)
                d = dict(
                    n_sources=n_sources, 
                    amplitudes=1,
                    snr=SNR_batch[i],
                    inter_source_correlations=corrs_batch[i],
                    n_orders = [0,0],
                    diffusion_parameter=0,
                    n_timepoints=T,
                    n_timecourses=np.inf,
                    iid_noise=True)
                sim_info.append(d)
            if return_info:
                sim_info = pd.DataFrame(sim_info)
                yield X, y, sim_info
            else:
                yield X, y


def generator_single_simple(leadfield, corr, T, n_sources, indices, SNR, random_seed=42):
    """
    Parameters
    ----------
    leadfield : numpy.ndarray
        The leadfield matrix.
    corr : float
        The correlation coefficient between the sources.
    T : int
        The number of time points in the sources.
    n_sources : int
        The number of sources to generate.
    indices : list
        The indices of the sources to generate.
    SNR : float
        The signal to noise ratio.
    random_seed : int
        The random seed for replicable simulations.
    
    Return
    ------
    X : numpy.ndarray
        The simulated EEG data. 
    y: numpy.ndarray
        The simulated source data.
    """
    rng = np.random.default_rng(random_seed)
    
    S = gen_correlated_sources(corr, T, n_sources)
    M = leadfield[:, indices] @ S # use Ground Truth Gain matrix
    n_channels, n_dipoles = leadfield.shape

    scale = np.max(abs(M))
    Ms = M * scale
    MEG_energy = np.trace(Ms @ Ms.T) / (n_channels*T)
    noise_var = MEG_energy/(10**(SNR/10))
    Noise = rng.standard_normal((n_channels, T)) * np.sqrt(noise_var)
    X = Ms + Noise
    y = np.zeros((n_dipoles, T))
    y[indices, :] = S
    
    
    return X, y

def gen_correlated_sources(corr_coeff, T, Q):
    ''' Generate Q correlated sources with a specified correlation coefficient.
    The sources are generated as sinusoids with random frequencies and phases.

    Parameters
    ----------
    corr_coeff : float
        The correlation coefficient between the sources.
    T : int
        The number of time points in the sources.
    Q : int
        The number of sources to generate.

    Returns
    -------
    Y : numpy.ndarray
        The generated sources.
    '''
    Cov = np.ones((Q, Q)) * corr_coeff + np.diag(np.ones(Q) * (1 - corr_coeff))  # required covariance matrix
    freq = np.random.randint(10, 31, Q)  # random frequencies between 10Hz to 30Hz

    phases = 2 * np.pi * np.random.rand(Q)  # random phases
    t = np.linspace(10 * np.pi / T, 10 * np.pi, T)
    Signals = np.sqrt(2) * np.cos(2 * np.pi * freq[:, None] * t + phases[:, None])  # the basic signals

    if corr_coeff < 1:
        A = np.linalg.cholesky(Cov).T  # Cholesky Decomposition
        Y = A @ Signals
    else:  # Coherent Sources
        Y = np.tile(Signals[0, :], (Q, 1))

    return Y


def get_cov(n, corr_coef):
    '''Generate a covariance matrix that is symmetric along the
    diagonal that correlates sources to a specified degree.'''
    if corr_coef < 1:
        cov = np.ones((n,n)) * corr_coef + np.eye(n)*(1-corr_coef)
        cov = np.linalg.cholesky(cov)
    else:
        # Make all signals be exactly the first one (perfectly coherent)
        cov = np.zeros((n, n))
        cov[:, 0] = 1
    return cov.T

def add_white_noise(X_clean, snr, rng, noise_color_coeff=0.5, correlation_mode=None):
    ''' 
    Parameters
    ----------
    correlation_mode : None/str
        None implies no correlation between the noise in different channels.
        'bounded' : Colored bounded noise, where channels closer to each other will be more correlated.
        'diagonal' : Some channels have varying degrees of noise.
        'cholesky' : A set correlation coefficient between each pair of channels
    noise_color_coeff : float
        The magnitude of spatial coloring of the noise (not the magnitude of noise overall!).
    '''
    # print(X_clean.shape)
    n_chans, n_time = X_clean.shape
    X_noise = rng.standard_normal((n_chans, n_time))
    # X_noise = rng.uniform(0, 1, (n_chans, n_time))
    # Ensure equal noise variance
    # X_noise = (X_noise.T / np.var(X_noise, axis=1)).T
    
    if correlation_mode == "cholesky":

        covariance_matrix = np.full((n_chans, n_chans), noise_color_coeff)
        np.fill_diagonal(covariance_matrix, 1)  # Set diagonal to 1 for variance

        # Generate correlated noise
        mean = np.zeros(n_chans)  # Mean of the noise
        X_noise = np.random.multivariate_normal(mean, covariance_matrix, n_time).T
        
        # if noise_color_coeff < 1:
        #     Cov = np.ones((n_chans, n_chans)) * noise_color_coeff + np.diag(np.ones(n_chans) * (1 - noise_color_coeff))
        #     # Correlate the noise channels
        #     X_noise = np.linalg.cholesky(Cov).T @ X_noise
        # else:
        #     X_noise = np.tile(X_noise[0, :], (n_chans, 1))
        
        # plt.figure()
        # plt.bar(np.arange(n_chans), X_noise.std(axis=1))
        # plt.title("Channel Noise Power Histogram After")

        # Some old code
        # # Inter-channel correlations
        # coeff_mat = rng.random((X_clean.shape[0], X_clean.shape[0]))
        # np.fill_diagonal(coeff_mat, 1)
        
        # # Make positive semi-definite
        # coeff_mat = np.dot(coeff_mat, coeff_mat.T)

        # # Make matrix symmetric
        # coeff_mat = (coeff_mat + coeff_mat.T)/2
        # # print(coeff_mat)
        # # Random partially correlated noise
        # X_noise = np.linalg.cholesky( coeff_mat ) @ X_noise
    elif correlation_mode == "bounded":
        num_sensors = X_noise.shape[0]
        Y = np.zeros_like(X_noise)
        
        # Apply coloring to the noise
        for i in range(num_sensors):
            Y[i, :] = X_noise[i, :]
            for j in range(num_sensors):
                if abs(i - j) % num_sensors == 1:
                    Y[i, :] += (noise_color_coeff / np.sqrt(2)) * X_noise[j, :]
        X_noise = Y
    elif correlation_mode == "diagonal":
        # Apply coloring to the noise
        X_noise[1::3, :] *= (1 - noise_color_coeff)
        X_noise[2::3, :] *= (1 + noise_color_coeff)
    elif correlation_mode is None:
        pass
    else:
        msg = f"correlation_mode can bei either None, cholesky, bounded or diagonal, but was {correlation_mode}"
        raise AttributeError(msg)
    
    # rms_noise = rms(X_noise)
    # rms_signal = rms(X_clean)

    # scaler = rms_signal / (snr * rms_noise)
    
    # # According to Adler et al.:
    X_clean_energy = np.trace(X_clean@X_clean.T)/(X_clean.shape[0]*X_clean.shape[1])
    noise_var = X_clean_energy/snr
    scaler = np.sqrt(noise_var)

    X_full = X_clean + X_noise*scaler
    # print(rms(X_clean), rms(X_noise*scaler))
    return X_full

def add_error(leadfield, forward_error, gradient, rng):
    n_chans, n_dipoles = leadfield.shape
    noise = rng.uniform(-1, 1, (n_chans, n_dipoles)) @ gradient
    leadfield_mix = leadfield / np.linalg.norm(leadfield) + forward_error * noise / np.linalg.norm(noise)
    return leadfield_mix

def rms(x):
    return np.sqrt(np.mean(x**2))
    

