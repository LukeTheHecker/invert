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