from setuptools import setup, find_packages

setup(
    name="titanic",
    version="1.0.0",
    description="Titanic survival prediction",
    packages=find_packages(),
    python_requires= ">=3.10",
    install_requires=[
        "mlflow",
        "numpy",
        "matplotlib",
        "seaborn",
        "optuna",
        "pandas",
        "scikit-learn",
        "xgboost",
    ],
    extras_require={
        "dev": ['jupyter']
    }
)