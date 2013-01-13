from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='hetnet',
    version='0.0.1',
    description='Heterosis Random Network Study',
    long_description=readme,
    author='Tal Levy',
    author_email='tlevy@ucdavis.edu',
    url='http://tal.cs.ucdavis.edu/heterosis',
    license=license,
    packages = ['hetnet'],
    install_requires=['igraph', 'numpy', 'scipy']
)
