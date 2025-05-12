from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='UM_TKE_Budget',
    url='https://github.com/SheriffOfRobinhood/UM_TKE_Budget',
    author='Yuqi Bai',
    author_email='yuqi.bai@reading.ac.uk',
    contributors='Peter Clark',
    # Needed to actually package something
    packages=['um_tke_budget', 
              ],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'dask', 'xarray', 'loguru', 'datetime', 'monc_utils', 'Subfilter'],
    # *strongly* suggested for sharing
    version='0.0.1',
    # The license can be anything you like
    license='MIT',
    description='A python module for computing sub-filter TKE budget components from UM outputs.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)