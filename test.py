import os

from pprint import pprint
from IPython import embed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from envs.mtsp_simple import MTSP, MTSPSimple
from models.gnn import Model
from agent.QLearning import Agent
from utils.simulator import Simulator
from utils.logger_tool import LoggerTool

import torch
import torch.multiprocessing as mp
from torch.optim import Adam
torch.set_num_threads(1)

import time
from datetime import datetime

import argparse

def test(config, model):
    now = datetime.now()
    config_env = config['env_test'] if 'env_test' in config else config['env']

    env = MTSP(config_env)
    s = env.reset()
    done = False
    score = 0
    step = 0
    while not done:
        action = model.action(s, softmax=False)
        #if score !=0: print(f"p_max: {model.p.max():.3f}, p_min: {model.p.min():.3f}")
        s_next, reward, done = env.step(action['list'])
        s = s_next
        score += reward
        step += 1
        print(f"step: {step}")
    #env.render()
    location_history = env.robots[0].location_history
    costs = [ robot.cost for robot in env.robots ]
    amplitude = max(costs) - min(costs)

    name_test = config['env']['file'].split('.')[0]
    path_test = f"test/[{now.strftime('%y%m%d')}][{now.strftime('%H-%M-%S')}]/{name_test}"
    os.makedirs(path_test, exist_ok=True)
    env.draw(path=path_test + f"/location_history-{name_test}_{config['env']['num_robots']}.png")

    return sum(costs), max(costs), amplitude

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train_mtsp')
    parser.add_argument('--path_model', type=str, 
                        help='path to model for test')
    parser.add_argument('--conf', type=str, 
                        help='conf to be used for test')
    args = parser.parse_args()
    print(args)

    config = __import__(f'conf.{args.conf}', fromlist=[None]).config
    model = Model(config, 'cpu').to('cpu')

    path_model = args.path_model
    state_dict = torch.load(path_model)
    model.load_state_dict(state_dict)

    total_cost, max_cost, amplitude = test(config, model)
    
    logger.info(
        f"total_cost: {total_cost}\n"
        + f"max_cost: {max_cost}\n"
        + f"amplitude: {amplitude}"
        )

    
