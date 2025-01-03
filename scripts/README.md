# Training and Evaluation Scripts

This directory contains scripts for training and evaluating agents on different manifolds.

## Training

To train an agent on a specific manifold:
```bash
python train.py --config configs/sphere.yaml --log-dir logs/
python train.py --config configs/torus.yaml --log-dir logs/
python train.py --config configs/hyperbolic.yaml --log-dir logs/
```

## Evaluation

To evaluate a trained agent:
```bash
python evaluate.py --checkpoint logs/sphere_xxx/best_model.pt --config configs/sphere.yaml --render
```

## Available Scripts

- `train.py`: Main training script
- `evaluate.py`: Evaluation script for trained agents