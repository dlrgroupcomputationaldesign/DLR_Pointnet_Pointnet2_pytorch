# create setup.py
from setuptools import setup, find_packages

setup(
    name="inference",
    version="0.0.0",    #PEP 440
    description="Point cloud inference module.",
    packages=find_packages(),
    install_requires=[]
)   