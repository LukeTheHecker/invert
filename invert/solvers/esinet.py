from esinet import Simulation, Net

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