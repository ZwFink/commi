import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "commi",
    version = "0.0.1",
    author = "Zane Fink",
    author_email = "zanef2@illinois.edu",
    description = ("Communication interface bridging between Charm4Py and mpi4py"),
    license = "BSD",
    keywords = "parallel hpc communication network performance",
    url = "https://github.com/zwfink/commi",
    packages=['commi', 'tests'],
    long_description=read('README.md'),
    #install_requires=['mpi4py', 'pytest', 'pytest-mpi'],
    install_requires=['mpi4py', 'numba'],
    classifiers=[
        "Development Status :: 1 - Planning",
    ],
)
