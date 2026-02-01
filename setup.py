from setuptools import setup, find_packages

setup(
    name="daf_moe",
    version="0.1.0",
    description="Official implementation of DAF-MoE (Distribution-Aware Feature-level Mixture of Experts)",
    author="Anonymous",  # Anonymous for Double-Blind Review
    author_email="anonymous@example.com",
    packages=find_packages(), # Automatically finds 'src' and subpackages
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "pyyaml",
        "scikit-learn",
        "tqdm",
        "optuna",
        "xgboost",
        "catboost"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)