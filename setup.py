# setup.py
from setuptools import setup, find_packages

setup(
    name="my_anymal_project",
    version="0.0.1",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    install_requires=[
        "isaaclab",  # Isaac Lab dependency
    ],
)