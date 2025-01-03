from setuptools import setup, find_packages

setup(
    name="topo_explorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "gymnasium>=0.26.0",
        "pyyaml>=5.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
        ]
    },
    author="Pedro Cortes",
    description="A framework for exploring topological spaces with reinforcement learning",
    long_description="A framework for exploring and learning on geometric manifolds using reinforcement learning.",
    python_requires=">=3.7",
)