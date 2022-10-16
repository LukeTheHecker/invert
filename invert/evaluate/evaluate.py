import numpy as np
from scipy.stats import pearsonr

def nmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = np.mean(((y_true_slice/abs(y_true_slice).max()) - (y_pred_slice/abs(y_pred_slice).max()))**2)
    return error


def corr(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    error = np.zeros(y_true.shape[1])
    for i, (y_true_slice, y_pred_slice) in enumerate(zip(y_true.T, y_pred.T)):
        error[i] = pearsonr(y_true_slice, y_pred_slice)[0]
    return error

def true_variance_explained(y_true, y_pred, leadfield):
    ''' Calculate the true variance explained using the predicted source and the noiseless EEG.
    '''
    M = leadfield @ y_true
    M_hat = leadfield @ y_pred
    # Common average reference
    # M -= M.mean(axis=-1)
    # M_hat -= M_hat.mean(axis=-1)

    return calc_residual_variance(M_hat, M)

def calc_residual_variance(M_hat, M):
    return 100 *  np.sum( (M-M_hat)**2 ) / np.sum(M**2)