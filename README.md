# **invertmeeg** - A high-level M/EEG Python library for EEG inverse solutions

This package contains various approaches to solve the M/EEG inverse problems. It
integrates with the [mne-python](https://mne.tools) framework.

Read the [Documentation here!](https://lukethehecker.github.io/invert/)

Install the package from [pypi](https://pypi.org/project/invertmeeg/):

```
pip install invertmeeg
```

To check if the installation works run:

```
python -c 'import invert'
```

To test the package simply run:

```
pytest tests
```

To calculate an inverse solution using *minimum norm estimates* simply type:

```
from invert import Solver

# fwd = ...
# evoked = ...

# Create a Solver instance
solver_name = "MNE"
solver = Solver(solver_name)

# Calculate the inverse operator
solver.make_inverse_operator(fwd)

# Apply the inverse operator to your data
stc = solver.apply_inverse_operator(evoked)

# Plot the resulting source estimate
stc.plot()
```

There are many solvers implemented in the package, and you can find them
[here!](https://lukethehecker.github.io/invert/content/solvers.html)
