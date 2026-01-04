from setuptools import setup, find_packages

setup(
    name="learn_noise_vst",
    version="0.1.0",
    description="Learnable Variance-Stabilizing Transforms for Noise Characterization",
    author="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.7.0",
        ]
    },
)
