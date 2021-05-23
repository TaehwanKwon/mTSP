import os

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['MKL_SERIAL'] = 'YES'

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

    return sum(costs), amplitude

def train(config, model, agent, simulator):
    now = datetime.now()
    path_log = f"logs/[{now.strftime('%y%m%d')}][{now.strftime('%H-%M-%S')}]"
    os.makedirs(path_log, exist_ok=True)
    
    optimizer = Adam(model.parameters(), lr=config['learning']['lr'])
    logger_tool = LoggerTool(path_log)

    _time_10_step = time.time()
    simulator.save_to_replay_buffer(config['learning']['size_replay_buffer'])
    logger.info("###### Start training #####")
    for step_train in range(config['learning']['step'] + 1):
        _time_train = time.time()
        
        model.step_train = step_train
        optimizer.zero_grad()
        processed_batch = model.get_processed_batch()
        loss = agent.get_loss(processed_batch)
        loss.backward()
        optimizer.step()

        time_train = time.time() - _time_train

        # adding new data to replay buffer
        simulator.save_to_replay_buffer(config['learning']['size_batch'])

        if step_train % 10 == 0:
            # showing and writing loss
            time_10_step = time.time() - _time_10_step
            print(
                f"[{step_train}] loss: {loss:.3f}, "
                + f"time_10_step: {time_10_step:.2f} "
                )
            _time_10_step = time.time()
            
        if step_train % 100 == 0:
            # Test the performance of the training agent
            score, amplitude = test(config, model, step_train=step_train, path_log=path_log)
            print(
                f"cost: {score} "
                + f"amplitude: {amplitude} "
                    )
            logger_tool.write(
                step_train, 
                {
                    'loss': loss,
                    'training_time': time_train,
                    'score': score,
                    'amplitude': amplitude,
                    }
                )

        # saving model
        if step_train > 1 and step_train % 1000 == 0:
            torch.save(simulator.model_cpu.state_dict(), f"{path_log}/model_{step_train}.pt")

    # kill spawned processes
    simulator.terminate()

if __name__=='__main__':
    mp.set_start_method('spawn')
    device = 'cuda:0'
    #device = 'cpu'

    config = __import__('conf.conf-kth', fromlist=[None]).config
    model = Model(config, device).to(device)
    model.initialize_batch()
    agent = Agent(config)
    simulator = Simulator(config, model)

    try:
        train(config, model, agent, simulator)
    except KeyboardInterrupt:
        # terminate processes generated for collecting data
        simulator.terminate()




    
