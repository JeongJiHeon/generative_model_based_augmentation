#@title Autoload all modules
import sys
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import glob
import torch.nn as nn
import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint


import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets



# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  #from configs.ve import cifar10_ncsnpp_continuous as configs
  #ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  from configs.ve import gbm_ncsnpp as configs
  ckpt_filename = "" # Checkpoint file path
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5




random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
# scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())
os.makedirs('result', exist_ok = True)


from PIL import Image
with torch.no_grad():
    for i in range(1000):
        predictor_size = 300
        img_size = config.data.image_size
        channels = config.data.num_channels
        sampling_shape = (predictor_size, channels, img_size, img_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        sample, n = sampling_fn(score_model)

        sample = sample.cpu().numpy()
        sample[sample<0] = 0
        sample[sample>1] = 1
        sample = (sample*255).astype(np.uint8)

        for j in range(len(sample)):
            np.save(f'result/{i * predictor_size + j:06}.npy', sample[j].transpose(1,2,0))


        
