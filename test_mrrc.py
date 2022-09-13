import os

from pprint import pprint
from IPython import embed

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from envs.mrrc import MRRC
from models.gnn import Model
from agent.QLearning import Agent
from utils.simulator import Simulator

import torch
import torch.multiprocessing as mp
from torch.optim import Adam

torch.set_num_threads(1)

import time
from datetime import datetime

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_mtsp')
    parser.add_argument('--conf', type=str,
                        help='conf to be used for test')
    args = parser.parse_args()
    config = __import__(f'conf.{args.conf}', fromlist=[None]).config
    config_env = config['env_test'] if 'env_test' in config else config['env']

    env = MRRC(config_env)
    s = env.reset()
    env.render()

    # Testing random action
    for _ in range(5):
        a = env.sample_action()
        print(a)
        s_next, r, done = env.step(a)
        print(f'reward: {r}')
        print(s_next['x_a'][:, :, :, 3])
        env.render()


    # Testing specific action
    # a = [0, None]
    # s_next, r, done = env.step(a)
    # print(f'reward: {r}')
    # env.render()





