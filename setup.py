from setuptools import setup, find_packages

setup(name='meslf',  # name of package on import
      version='1.0',  # package version
      description='Load flow solver for multi-carrier energy systems',  # brief description
      author='Anne Markensteijn',  # author(s)
      author_email='a.s.markensteijn@tudelft.nl',  # email
      packages=find_packages(),
      install_requires=[  # dependencies
        'matplotlib',  # for plotting
        'numpy',  # for numerical calculations
        'pandas', # for writing network to, and reading a network from a pandas dataframe
        'pandapower', # for some examples
        'numpydoc',  # numpy-style docstrings for sphinx
        'pytest',  # for testing
      ],
    zip_safe=False) # package can be installed from zip file
