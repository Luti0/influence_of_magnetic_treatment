from setuptools import setup, find_packages

setup(
    name="influence_of_magnetic_treatment",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'torch',
        'xgboost',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'typing',
        'optuna',
        'joblib'
    ],
)