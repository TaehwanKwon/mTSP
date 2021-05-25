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

def test(config, model, step_train=0, path_log=None):
    config_env = config['env_test'] if 'env_test' in config else config['env']
    model.config['env'] = config_env # switch the configuration of environment

    env = MTSP(config_env)
    s = env.reset()
    done = False
    score = 0
    while not done:
        action = model.action(s)
        s_next, reward, done = env.step(action['list'])
        s = s_next
        score += reward
    #env.render()
    location_history = env.robots[0].location_history
    costs = [ robot.cost for robot in env.robots ]
    amplitude = max(costs) - min(costs)
    
    if not path_log is None:
        env.draw(path=path_log + f'/location_history_{step_train}.png')

    model.config['env'] = config['env'] # Rollback the configuration of environment

    return sum(costs), max(costs), amplitude

def train(args, config, model, agent, simulator):
    now = datetime.now()
    path_log = f"logs/[{now.strftime('%y%m%d')}][{now.strftime('%H-%M-%S')}]"
    os.makedirs(path_log, exist_ok=True)

    path_prev = args.path_prev
    if path_prev:
        state_dict = torch.load(path_prev)
        state_dict = {key: state_dict[key].to(device) for key in state_dict}
        model.load_state_dict(state_dict)
        f = open(f"{path_log}/comment.txt", 'w')
        f.write(path_prev)
        f.close()

    step_prev = args.step_prev
    model.step_train = step_prev
    
    optimizer = Adam(model.parameters(), lr=config['learning']['lr'])
    logger_tool = LoggerTool(path_log)

    _time_10_step = time.time()
    simulator.save_to_replay_buffer(config['learning']['size_replay_buffer'])
    logger.info("###### Start training #####")
    for step_train in range(step_prev, config['learning']['step'] + 1):
        _time_train = time.time()
        
        model.step_train = step_train
        optimizer.zero_grad()
        processed_batch = model.get_processed_batch()
        loss = agent.get_loss(processed_batch)
        loss.backward()
        optimizer.step()

        time_train = time.time() - _time_train

        if step_train % 10 == 0:
            # showing and writing loss
            time_10_step = time.time() - _time_10_step
            print(
                f"[{step_train}] loss: {loss:.3f}, "
                + f"time_10_step: {time_10_step:.2f} "
                )
            _time_10_step = time.time()

        if step_train % 200 == 0:
            # adding new data to replay buffer
            simulator.save_to_replay_buffer(config['learning']['size_batch'])
            
        if step_train % 100 == 0:
            # Test the performance of the training agent
            total_cost, max_cost, amplitude = test(config, model, step_train=step_train, path_log=path_log)
            print(
                f"toptal_cost: {total_cost} "
                + f"max_cost: {max_cost} "
                + f"amplitude: {amplitude} "
                    )
            logger_tool.write(
                step_train, 
                {
                    'loss': loss,
                    'training_time': time_train,
                    'total_cost': total_cost,
                    'max_cost': max_cost,
                    'amplitude': amplitude,
                    }
                )

        # saving model
        if step_train > 1 and step_train % 1000 == 0:
            state_dict = model.state_dict()
            state_dict_cpu = {key: state_dict[key].cpu() for key in state_dict}
            torch.save(state_dict_cpu, f"{path_log}/model_{step_train}.pt")

    # kill spawned processes
    simulator.terminate()

if __name__=='__main__':
    mp.set_start_method('spawn')
    device = 'cuda:0'

    parser = argparse.ArgumentParser(description='train_mtsp')
    parser.add_argument('--path_prev', type=str, 
                        help='path to previous model')
    parser.add_argument('--step_prev', type=int, default=0,
                        help='previous train step')
    parser.add_argument('--conf', type=str, 
                        help='conf to be used for training')
    args = parser.parse_args()
    print(args)
    

    config = __import__(f'conf.{args.conf}', fromlist=[None]).config
    model = Model(config, device).to(device)
    model.initialize_batch()
    model.set_extra_gpus()    
    agent = Agent(config)
    simulator = Simulator(config, model)
    model.simulator = simulator
    
    try:
        train(args, config, model, agent, simulator)
    except KeyboardInterrupt:
        # terminate processes generated for collecting data
        simulator.terminate()




    
