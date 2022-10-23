from esinet import Simulation, Net
from .base import BaseSolver, InverseOperator
import mne

class SolverFullyConnected(BaseSolver):
    ''' Class for the Fully-Connected (FC) neural network's inverse solution.
    
    Attributes
    ----------
    forward : mne.Forward
        The mne-python Forward model instance.
    '''
    def __init__(self, name="Fully-Connected", **kwargs):
        self.name = name
        return super().__init__(**kwargs)

    def make_inverse_operator(self, forward, evoked, *args, alpha='auto', 
                            n_simulations=5000, activation_function="tanh", 
                            verbose=0, **kwargs):
        ''' Calculate inverse operator.

        Parameters
        ----------
        forward : mne.Forward
            The mne-python Forward model instance.
        alpha : float
            The regularization parameter.
        
        Return
        ------
        self : object returns itself for convenience
        '''
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        info = evoked.info
        settings = dict(duration_of_trial=0.)
        sim = Simulation(forward, info, settings=settings, verbose=verbose).simulate(n_simulations)

        model_args = dict(model_type="FC", activation_function=activation_function, )
        inverse_operator = InverseOperator(Net(forward, **model_args, verbose=verbose).fit(sim), self.name)
        self.inverse_operators = [inverse_operator,]
        
        return self

    def apply_inverse_operator(self, evoked) -> mne.SourceEstimate:
        return super().apply_inverse_operator(evoked)

def make_fullyconnected_inverse_operator(fwd, info, verbose=0):
    """ Calculate the inverse operator using the Fully-Connected artificial neural network model.

    Parameters
    ----------
    leadfield : mne.Foward
        The forward model object.
    info : mne.Info
        The mne info object.

    Return
    ------
    inverse_operator : esinet.Net
        The neural network model object from the esinet package.

    """
    settings = dict(duration_of_trial=0.)
    sim = Simulation(fwd, info, settings=settings, verbose=verbose).simulate(5000)

    model_args = dict(model_type="FC", activation_function="tanh")
    inverse_operator = Net(fwd, **model_args, verbose=verbose).fit(sim)

    return inverse_operator


def make_lstm_inverse_operator(fwd, info, verbose=0):
    """ Calculate the inverse operator using the Long-Short Term Memory
    artificial neural network model.

    Parameters
    ----------
    leadfield : mne.Foward
        The forward model object.
    info : mne.Info
        The mne info object.

    Return
    ------
    inverse_operator : esinet.Net
        The neural network model object from the esinet package.

    """
    settings = dict(duration_of_trial=0.01)
    sim = Simulation(fwd, info, settings=settings, verbose=verbose).simulate(5000)

    model_args = dict(model_type="LSTM")
    inverse_operator = Net(fwd, **model_args, verbose=verbose).fit(sim)

    return inverse_operator