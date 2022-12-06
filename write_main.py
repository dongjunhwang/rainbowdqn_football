
_MODEL_MAIN = r"""
import os
import sys
import cv2
import collections
import numpy as np
from gfootball.env import observation_preprocessing

# PFRL
import torch
import pfrl

import base64
from model_weights import model_string

def make_model():
    # Q_function
    model = pfrl.q_functions.DistributionalDuelingDQN(n_actions=19, n_atoms=51, v_min=-10, v_max=10, n_input_channels=4)

    # Noisy nets
    pfrl.nn.to_factorized_noisy(model, sigma_scale=0.5)

    # load weights
    with open("model.dat", "wb") as f:
        f.write(base64.b64decode(model_string))
    weights = torch.load("model.dat", map_location=torch.device('cpu'))
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
"""