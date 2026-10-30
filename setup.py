from setuptools import setup, find_packages

setup(
    name="lob_alpha",
    version="0.1.0",
    description="High-Frequency Limit Order Book Prediction: Microstructure Signals, Deep Learning & Execution Systems",
    author="Aarav Singhal",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "torch>=2.0.0",
        "pyarrow>=12.0.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "kaggle>=1.6.0",
        "tqdm>=4.65.0"
    ],
)
