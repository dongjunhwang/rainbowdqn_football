import os
from os.path import join as ospj
import numpy as np
import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse
import base64

from pfrl import experiments
from model import DistributionalDuelingDQN
from model import to_factorized_noisy
from agents import CategoricalDoubleDQN
from explorer import Greedy
from replay_buffer import PrioritizedReplayBuffer

from util import logger_config
from util import seed_everything
from util import text_csv_converter
from util import str2bool
from util import make_video
from write_main import write_main
from wrapper import make_env

class Trainer(object):
    def __init__(self, args):
        self.args = args

        print(torch.cuda.is_available())
        self.gpu = 0 if torch.cuda.is_available() else -1
        self.logger = logger_config(os.path.join(args.log_path, "result.log"))

        seed_everything(args.seed)
        self.train_seed = args.seed
        self.test_seed = 2 ** 31 - 1 - args.seed

        self.env, self.eval_env = self.set_env()
        self.q_func = self.set_q_func()

        self.explorer = Greedy()
        self.optimizer = self.set_optimizer()
        self.rbuf = self.set_replay()

        self.agent = self.set_model()

    @staticmethod
    def phi(x):
        return np.asarray(x, dtype=np.float32) / 255

    def set_env(self):
        env = make_env(self.test_seed, self.train_seed, test=False)
        eval_env = make_env(self.test_seed, self.train_seed, test=True)

        print('observation space:', env.observation_space.low.shape)
        print('action space:', env.action_space)
        env.reset()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        print('next observation:', obs.shape)
        print('reward:', r)
        print('done:', done)
        print('info:', info)

        return env, eval_env

    def set_q_func(self):
        obs_n_channels = self.env.observation_space.low.shape[0]
        n_actions = self.env.action_space.n
        print("obs_n_channels: ", obs_n_channels)
        print("n_actions: ", n_actions)

        q_func = DistributionalDuelingDQN(n_actions, self.args.n_atoms, self.args.v_min,
                                          self.args.v_max, obs_n_channels)
        print(q_func)

        to_factorized_noisy(q_func, sigma_scale=self.args.noisy_sigma)
        return q_func

    def set_optimizer(self):
        # Use the same hyper parameters as https://arxiv.org/abs/1710.02298
        return torch.optim.Adam(self.q_func.parameters(), self.args.lr, eps=1.5 * 10 ** -4)

    def set_replay(self):
        betasteps = 5 * 10 ** 7 / self.args.update_interval
        rbuf = PrioritizedReplayBuffer(
            self.args.buffer_capacity,
            alpha=self.args.alpha,
            beta0=self.args.beta0,
            betasteps=betasteps,
            num_steps=self.args.multi_step,
            normalize_by_max="memory",
        )
        return rbuf

    def set_model(self):
        agent = CategoricalDoubleDQN(
            self.q_func,
            self.optimizer,
            self.rbuf,
            gpu=self.gpu,
            gamma=self.args.gamma,
            explorer=self.explorer,
            minibatch_size=self.args.batch_size,
            replay_start_size=self.args.replay_start_size,
            target_update_interval=self.args.target_update_interval,
            update_interval=self.args.update_interval,
            batch_accumulator="mean",
            phi=self.phi,
        )

        if self.args.pretrained_path is not None:
            agent.load(self.args.pretrained_path)
        return agent

    def export_graph(self, scores):
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_title("median reward")
        ax2.set_title("average loss")
        sns.lineplot(x="episodes", y="median", data=scores, ax=ax1)
        sns.lineplot(x="episodes", y="average_loss", data=scores, ax=ax2)
        plt.savefig(os.path.join(self.args.log_path, 'loss_graph.png'))

    def train(self):
        experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=self.env,
            steps=self.args.num_steps,
            eval_n_steps=None,
            eval_n_episodes=1,
            eval_interval=3000,
            outdir=self.args.log_path,
            checkpoint_freq=100000,
            save_best_so_far_agent=True,
            eval_env=self.eval_env,
            logger=self.logger
        )

        file_path_txt = os.path.join(self.args.log_path, "scores.txt")
        file_path_csv = os.path.join(self.args.log_path, "scores.csv")
        file_path_png = os.path.join(self.args.log_path, "scores.png")
        text_csv_converter(file_path_txt)
        scores = pd.read_csv(file_path_csv)
        self.export_graph(scores)
        df_styled = scores.style.background_gradient()
        dfi.export(df_styled, file_path_png)

    def save_model_to_submission_file(self):
        pfrl_path = os.path.join(self.args.log_path, "sub")
        os.makedirs(pfrl_path, exist_ok=True)
        os.chdir(os.getcwd())
        path_list = []
        model_path = os.path.join(self.args.log_path, f'{self.args.num_steps}_finish/model.pt')
        model_weight_path = os.path.join(self.args.log_path, 'sub', 'model_weights.py')
        process_file_path = os.path.join(self.args.log_path, 'sub', 'main.py')
        path_list.append([model_path, model_weight_path, process_file_path])

        model_path = self.args.player2_path
        model_path, model_weight_path, process_file_path = ospj(model_path, 'model.pt'), \
                                                           ospj(model_path, 'model_weights.py'), \
                                                           ospj(model_path, 'main.py')
        path_list.append([model_path, model_weight_path, process_file_path])

        process_file_path_list = []
        for i, path in enumerate(path_list):
            model_path, model_weight_path, process_file_path = map(str, path)
            if i == 1:
                main_code = write_main(51, 10, -10, 0.5, self.args.player2_path)
            else:
                main_code = write_main(self.args.n_atoms, self.args.v_max,
                                       self.args.v_min, self.args.noisy_sigma, self.args.log_path)
            with open(model_path, 'rb') as f:
                encoded_string = base64.b64encode(f.read())
            with open(model_weight_path, 'w') as f:
                f.write(f'model_string={encoded_string}')
            with open(process_file_path, 'w') as f:
                f.write(main_code)
            process_file_path_list.append(process_file_path)
        make_video(self.args.log_path, process_file_path_list)

def evaluate(args):
    model_path_list = [args.player1_path, args.player2_path]
    process_file_path_list = []
    for i, model_path in enumerate(model_path_list):
        model_path, model_weight_path, process_file_path = ospj(model_path, 'model.pt'), \
                                                           ospj(model_path, 'model_weights.py'), \
                                                           ospj(model_path, 'main.py')
        if i == 1:
            main_code = write_main(51, 10, -10, 0.5, args.player2_path)
        else:
            main_code = write_main(args.n_atoms, args.v_max,
                                   args.v_min, args.noisy_sigma, args.log_path)

        with open(model_path, 'rb') as f:
            encoded_string = base64.b64encode(f.read())
        with open(model_weight_path, 'w') as f:
            f.write(f'model_string={encoded_string}')
        with open(process_file_path, 'w') as f:
            f.write(main_code)

        process_file_path_list.append(process_file_path)
    make_video(args.log_path, process_file_path_list)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log_path', type=str, default='./train_log')
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--pretrained_path', type=str, default=None)
    args.add_argument('--test_only', type=str2bool, default=False)

    # for evaluation
    args.add_argument('--player1_path', type=str, default='./train_log/best')
    args.add_argument('--player2_path', type=str, default='./train_log/best')

    # HP Search
    # Replay Start Size
    args.add_argument('--replay_start_size', type=int, default=2 * 10 ** 4)
    # Noisy Net
    args.add_argument('--noisy_sigma', type=float, default=0.5)
    # learning rate and optimizer
    args.add_argument('--lr', type=float, default=6.25e-5)
    # Prioritized replay buffer
    args.add_argument('--alpha', type=float, default=0.5)
    args.add_argument('--beta0', type=float, default=0.4)
    args.add_argument('--multi_step', type=int, default=3)
    # Distributional Dueling DQN
    args.add_argument('--n_atoms', type=int, default=51)
    args.add_argument('--v_max', type=int, default=10)
    args.add_argument('--v_min', type=int, default=-10)
    # Categorical Double DQN
    args.add_argument('--target_update_interval', type=int, default=32000)
    args.add_argument('--update_interval', type=int, default=4)
    args.add_argument('--gamma', type=float, default=0.99)
    args.add_argument('--batch_size', type=int, default=32)
    # Etc.
    args.add_argument('--buffer_capacity', type=int, default=10 ** 5)
    args.add_argument('--num_steps', type=int, default=100000)

    args = args.parse_args()
    os.makedirs(args.log_path, exist_ok=True)

    trainer = Trainer(args)
    if args.test_only:
        evaluate(args)
    else:
        trainer.train()
        trainer.save_model_to_submission_file()
