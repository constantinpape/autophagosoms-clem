from setuptools import setup, find_packages
__version__ = '0.0.1'

setup(name='phago_network_utils',
      version=__version__,
      packages=find_packages(exclude=['training',
                                      'prediction']))
