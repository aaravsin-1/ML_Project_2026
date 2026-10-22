from setuptools import setup, find_packages

setup(
    name="lob_alpha",
    version="0.1.0",
    description="High-Frequency Limit Order Book Prediction: Microstructure Signals, Deep Learning & Execution Systems",
    author="Quantitative Research",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "shap>=0.42.0",
        "pytest>=7.4.0",
        "tqdm>=4.65.0"
    ],
)
