from setuptools import setup, find_packages

setup(
    name="daf_moe",
    version="0.1.0",
    packages=find_packages(), # src 폴더를 자동으로 패키지로 인식
    install_requires=[
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "pyyaml",
        "scikit-learn",
        "tqdm"
    ],
)