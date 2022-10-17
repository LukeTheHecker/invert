# Let Us Combine All Source estimates (LUCAS)
import numpy as np
from copy import deepcopy
import mne
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from esinet import Simulation
from .base import BaseSolver

class SolverLUCAS(BaseSolver):
    ''' Class for the combined LUCAS inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="LUCAS", **kwargs):
        self.name = name
        self.solvers = None
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, alpha='auto', solvers="all",
                                verbose=0):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        evoked : mne.EvokedArray
            Evoked EEG data object.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        from invert import Solver
        
        self.forward = forward
        leadfield = self.forward['sol']['data']
        n_chans, _ = leadfield.shape

        if solvers == "all":
            solver_names = [ "MNE", "wMNE", "dSPM", 
                    "LORETA", "sLORETA", "eLORETA", 
                    "LAURA",  
                    "S-MAP",
                    "Champagne", "Multiple Sparse Priors", "Bayesian LORETA", "Bayesian MNE", "Bayesian Beamformer", "Bayesian Beamformer LORETA",
                    "Fully-Connected", 
                    "LUCAS",
                    "FISTA",
                    "OMP", "COSAMP", "SOMP", "REMBO",
                ]
        else:
            solver_names = deepcopy(solvers)

        solvers = []
        self.solver_names = solver_names
        for solver_name in self.solver_names:
            print(f"Preparing {solver_name}")
            solver = Solver(solver=solver_name).make_inverse_operator(forward, evoked, alpha=alpha, verbose=verbose)
            solvers.append(solver)

        
        self.weights = np.ones(len(solvers))
        self.solvers = solvers

        
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        all_stcs = []
        for solver in self.solvers:
            print("Applying ", solver.name)
            stc = solver.apply_inverse_operator(evoked)
            all_stcs.append( stc )
        
        stc_lucas = stc.copy()
        stc_lucas.data = np.average(np.stack([stc.data for stc in all_stcs], axis=0), axis=0, weights=self.weights)
        return stc_lucas
    
    def optimize_weights(self, forward, info, n_samples=1024):
        if self.solvers is None:
            msg = f'No solvers found. Please call --make_inverse_operator(forward)-- first!'
            raise AttributeError(msg)

        # duration_of_trial = info["sfreq"]
        settings = dict(duration_of_trial=0)
        sim = Simulation(forward, info, settings=settings).simulate(n_samples=n_samples)
        coefficients = []
        for i in range(n_samples):
            # print("sample ", i)
            evoked = sim.eeg_data[i].average()
            stc = sim.source_data[i]
            J_true = stc.data[:, 0]
            J = np.stack([solver.apply_inverse_operator(evoked).data[:, 0] for solver in self.solvers], axis=1)
            # Normalize Source Vectors
            J_true /= np.linalg.norm(J_true)
            J /= np.linalg.norm(J, axis=0)
            

            coef = LinearRegression().fit(J, J_true).coef_
            # coef = Lasso().fit(J, J_true).coef_
            
            coefficients.append( coef )
        
        weights = np.array(coefficients).mean(axis=0)
        weights = np.clip(weights, a_min=0, a_max=None)

        weights /= weights.mean()

        self.weights = weights
    
    def plot_weights(self):
        weight_dict = {solver.name: weight for solver, weight in zip(self.solvers, self.weights)}
        # weight_dict = {s.name.replace(' ', '_'): weight for s, weight in zip(solver.solvers, solver.weights)}
        
        plt.figure()
        plt.bar(range(len(weight_dict)), list(weight_dict.values()), align='center')
        plt.xticks(range(len(weight_dict)), list(weight_dict.keys()))
        plt.title("Weights of the LUCAS procedure")
    
    @staticmethod
    def filter_solvers(all_solvers):
        keepers = []
        for s in all_solvers:
            if not (s == "LUCAS" or "bayes" in s.lower() or s.lower() == "multiple sparse priors" or "backus" in s.lower()):
                keepers.append(s)
        return keepers