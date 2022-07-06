import mne
import numpy as np

def pos_from_forward(forward, verbose=0):
    ''' Get vertex/dipole positions from mne.Forward model

    Parameters
    ----------
    forward : instance of mne.Forward
        The forward model. 
    
    Return
    ------
    pos : numpy.ndarray
        A 2D matrix containing the MNI coordinates of the vertices/ dipoles

    Note
    ----
    forward must contain some subject id in forward["src"][0]["subject_his_id"]
    in order to work.
    '''
    # Get Subjects ID
    subject_his_id = forward["src"][0]["subject_his_id"]
    src = forward["src"]

    # Extract vertex positions from left and right source space
    pos_left = mne.vertex_to_mni(src[0]["vertno"], 0, subject_his_id, 
                                verbose=verbose)
    pos_right = mne.vertex_to_mni(src[1]["vertno"], 1, subject_his_id, 
                                verbose=verbose)

    # concatenate coordinates from both hemispheres
    pos = np.concatenate([pos_left, pos_right], axis=0)

    return pos