from setuptools import setup, find_packages

setup(
    name="greek_energy_flow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "torch",
        "optuna"
    ],
)