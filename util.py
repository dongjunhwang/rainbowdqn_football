import random
import numpy as np
import torch
import csv
import os
import argparse

from kaggle_environments import make
from logging import getLogger, StreamHandler, FileHandler

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def logger_config(filepath):
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel("DEBUG")
    logger.setLevel("DEBUG")
    logger.addHandler(handler)
    logger.propagate = False

    file_handler = FileHandler(filepath)
    logger.addHandler(file_handler)
    return logger

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def text_csv_converter(datas):
    file_csv = datas.replace("txt", "csv")
    with open(datas) as rf:
        with open(file_csv, "w") as wf:
            readfile = rf.readlines()
            for read_text in readfile:
                read_text = read_text.split()
                writer = csv.writer(wf, delimiter=',')
                writer.writerow(read_text)

def exec_command(command):
    stream = os.popen(command)
    output = stream.readlines()
    return output


def make_video(log_path, process_file_path_list=None):
    env = make("football",
               configuration={"save_video": True,
                              "scenario_name": "11_vs_11_kaggle",
                              "running_in_notebook": False,
                              "logdir": log_path,
                              },
               debug=True)
    if process_file_path_list is None:
        agent = os.path.join(log_path, "sub", "main.py")
        output = env.run([agent, agent])[-1]
    else:
        agent = process_file_path_list
        output = env.run([agent[0], agent[1]])[-1]
    print('Left player: action = %s, reward = %s, status = %s, info = %s' % (
    output[0]["action"], output[0]['reward'], output[0]['status'], output[0]['info']))
    print('Right player: action = %s, reward = %s, status = %s, info = %s' % (
    output[1]["action"], output[1]['reward'], output[1]['status'], output[1]['info']))
    env.render(mode="human", width=800, height=600)

    webm_path = os.path.join(log_path, env.configuration.id)
    dir_list = os.listdir(webm_path)
    for dir_name in dir_list:
        if '.webm' in dir_name:
            src = os.path.join(webm_path, dir_name)
            dst = os.path.join(webm_path, 'result_video.mp4')
            exec_command("ffmpeg -i {} {}".format(src, dst))
