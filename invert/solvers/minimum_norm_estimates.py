import numpy as np

def make_mne_inverse_operator(leadfield, alpha=0.001, noise_cov=None):
    """ Calculate the inverse operator using Minimum Norm Estimates.

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    alpha : float
        The regularization parameter.
    noise_cov : numpy.ndarray
        The noise covariance matrix (channels x channels).

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, _ = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)
    inverse_operator = leadfield.T @ np.linalg.inv(leadfield @ leadfield.T + alpha * noise_cov)
    return inverse_operator

def make_wmne_inverse_operator(leadfield, alpha=0.001, noise_cov=None):
    """ Calculate the inverse operator using depth weighted Minimum Norm Estimates.

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    alpha : float
        The regularization parameter.
    noise_cov : numpy.ndarray
        The noise covariance matrix (channels x channels).

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, _ = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)

    omega = np.diag(np.linalg.norm(leadfield, axis=0))
    # I_3 = np.identity(3)
    W = omega # np.kron(omega, I_3)

    inverse_operator = np.linalg.inv(W.T @ W) @ leadfield.T  @ np.linalg.inv(leadfield @ np.linalg.inv(W.T @ W) @ leadfield.T + alpha * np.identity(n_chans))

    return inverse_operator



def make_dspm_inverse_operator(leadfield, alpha=0.001, noise_cov=None, source_cov=None):
    """ Calculate the inverse operator using dSPM.

    Parameters
    ----------
    leadfield : numpy.ndarray
        Leadfield (or gain matrix) G which constitutes the forward model of M =
        J @ G, where sources J are projected through the leadfield producing the
        observed EEG matrix M.
    alpha : float
        The regularization parameter.
    noise_cov : numpy.ndarray
        The noise covariance matrix (channels x channels).
    source_cov : numpy.ndarray
        The source covariance matrix (dipoles x dipoles). This can be used if
        prior information, e.g., from fMRI images, is available.

    Return
    ------
    inverse_operator : numpy.ndarray
        The inverse operator that is used to calculate source.

    """
    n_chans, n_dipoles = leadfield.shape
    if noise_cov is None:
        noise_cov = np.identity(n_chans)
    if source_cov is None:
        source_cov = np.identity(n_dipoles)
    
    K = source_cov @ leadfield.T @ np.linalg.inv(leadfield @ source_cov @ leadfield.T + alpha**2 * noise_cov)

    W_dSPM = np.diag(np.sqrt(1/np.diagonal(K @ noise_cov @ K.T)))
    inverse_operator = W_dSPM @ K
    
    return inverse_operator






