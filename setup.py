import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="invertmeeg",
    version="0.0.6",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="A high-level M/EEG Python library for EEG inverse solutions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/inverse",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow','mne', 'scipy', 'colorednoise', 'matplotlib', 'pyvista', 'pyvistaqt', 'PyQt5', 'vtk', 'tqdm', 'pytest', 'dill', 'scikit-learn', 'pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.3',
)
