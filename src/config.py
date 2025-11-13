import torch
import torch.nn as nn

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimizer choice
OPTIMIZER = "SGD"  

# Loss function
LOSS_FN = nn.CrossEntropyLoss()

# Scheduler (optional)
LR_STEP_SIZE = 30
LR_GAMMA = 0.1
