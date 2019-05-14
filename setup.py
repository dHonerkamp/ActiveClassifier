from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(
    name='activeClassifier',
    version='0.1.0',
    author='Daniel Honerkamp',
    author_email='',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=requirements
)