from setuptools import setup, find_packages

setup(
    name="CBR-FoX",
    version="1.0.0",
    long_description=open("README.md").read(),    # Root level
    install_requires=open("requirements.txt"),    # Root level
    packages=find_packages(),                     # Scans from root
)