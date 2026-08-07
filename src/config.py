"""
Global configuration for the project.
"""

# Dataset
TRAIN_RATIO = 0.90

# Training
BLOCK_SIZE = 8
BATCH_SIZE = 4

# Reproducibility
RANDOM_SEED = 1337

# Device
DEVICE = "cuda"