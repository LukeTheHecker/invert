import sys; sys.path.insert(0, '../')
from invert.config import all_solvers
from invert import Solver
import numpy as np
import mne
from esinet.forward import get_info, create_forward_model
pp = dict(surface='white', hemi='both', verbose=0)


# Parameters
sampling = 'ico3'
montage = "biosemi16"
alpha = 0.1
epochs = 1

# Forward Model
info = get_info(kind=montage)
fwd = create_forward_model(info=info, sampling=sampling)
vertices = [fwd["src"][0]['vertno'], fwd["src"][1]['vertno']]
leadfield = fwd["sol"]["data"]

# Simulation Data for testing
n_time = 20
# source_mat = np.zeros((leadfield.shape[1], n_time))
source_mat = np.random.randn(leadfield.shape[1], n_time)
stc = mne.SourceEstimate(source_mat, vertices, tmin=0, tstep=0.001)
evoked_mat = leadfield @ source_mat
evoked_mat -= evoked_mat.mean(axis=0)
evoked_mat += np.random.randn(*evoked_mat.shape)*evoked_mat.std()*1
evoked = mne.EvokedArray(evoked_mat, info, verbose=0).set_eeg_reference("average", projection=True, verbose=0).apply_proj()

# Tidy up
del evoked_mat, source_mat, leadfield, vertices

def test_solvers():
    for solver_name in all_solvers:

        print("########################\n", solver_name, "\n########################")
        solver = Solver(solver_name)
        solver.make_inverse_operator(fwd, evoked, alpha=alpha, epochs=epochs)
        stc_hat = solver.apply_inverse_operator(evoked)
        print(type(stc_hat))
