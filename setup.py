from setuptools import setup, find_packages

setup(
    name='grokking_transformer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'scipy',
        'tqdm',
    ]
)
