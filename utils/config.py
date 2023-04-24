#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config for running experiments

"""

from pathlib import Path
import torch 
import gpytorch 

TORCH_VERSION = torch.__version__
GPYTORCH_VERSION = gpytorch.__version__

AVAILABLE_GPU = torch.cuda.device_count()
GPU_ACTIVE = bool(AVAILABLE_GPU)
EPSILON = 1e-5
BASE_SEED = 42

BASE_PATH = Path(__file__).parent.parent
RESULTS_DIR = BASE_PATH / "results"
DATASET_DIR = BASE_PATH / "data"
LOG_DIR = BASE_PATH / "logs"
TRAINED_MODELS = BASE_PATH / "trained_models"


