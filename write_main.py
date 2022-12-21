
def write_main(n_atoms, v_max, v_min, noisy_sigma, path):
    model_file_string = r"""
import os
import sys
import cv2
import collections
import numpy as np
from model import DistributionalDuelingDQN
from model import to_factorized_noisy
from gfootball.env import observation_preprocessing

# PFRL
import torch
import pfrl

import base64
from model_weights import model_string

def make_model():
    # Q_function
    model = DistributionalDuelingDQN(n_actions=19, n_atoms={}, v_min={}, v_max={}, n_input_channels=4)

    # Noisy nets
    to_factorized_noisy(model, sigma_scale={})

    # load weights
    model_path = os.path.join("{}", "model.dat")
    with open(model_path, "wb") as f:
        f.write(base64.b64decode(model_string))
    weights = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    return model


model = make_model()

def agent(obs):
    global model

    # Get observations for the first (and only one) player we control.
    obs = obs['players_raw'][0]
    # Agent we trained uses Super Mini Map (SMM) representation.
    # See https://github.com/google-research/seed_rl/blob/master/football/env.py for details.
    obs = observation_preprocessing.generate_smm([obs])[0]
    # preprocess for obs
    obs = cv2.resize(obs, (84,84))    # resize
    obs = np.transpose(obs, [2,0,1])  # transpose to chw
    obs = torch.tensor(obs).float()   # to tensor
    obs = torch.unsqueeze(obs,0)      # add batch

    actions = model(obs)
    action = int(actions.greedy_actions.numpy()[0])  # modified
    return [action]
    """.format(n_atoms, v_min, v_max, noisy_sigma, path)
    return model_file_string