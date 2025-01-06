# Topological Space Explorer

A framework for exploring and learning on geometric manifolds using reinforcement learning.

![hippo](https://giphy.com/gifs/PvYyvxkRsgsANGcPK3)

## Installation

Install in development mode:
```bash
pip install -e .
```

## Usage

Train an agent:
```bash
python scripts/train.py --config configs/sphere.yaml
```

Evaluate a trained agent:
```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --config configs/sphere.yaml
```

Quick start example:
```bash
python scripts/quickstart.py
```

## Project Structure

- `topo_explorer/`: Main package directory
  - `environments/`: Environment implementations
  - `agents/`: RL agent implementations
  - `visualization/`: Visualization tools
- `scripts/`: Training and evaluation scripts
- `configs/`: Configuration files

## Features

- Support for various manifolds (sphere, torus, hyperbolic space)
- Geometric deep learning on manifolds
- Parallel transport and curvature-aware learning
- Visualization tools for geometric properties

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Gymnasium
- Matplotlib
