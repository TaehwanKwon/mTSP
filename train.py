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
from torch.optim import Adam, AdamW
torch.set_num_threads(1)

import time
from datetime import datetime

import argparse

def test(config, model, step_train=0, path_log=None, prev_cost_min=None):
    config_env = config['env_test'] if 'env_test' in config else config['env']
    model.config['env'] = config_env # switch the configuration of environment

    env = MTSP(config_env)
    s = env.reset()
    done = False
    score = 0
    step = 0
    Q_pred = 0
    while not done:
        model.show_presence = False
        if step==1:
            model.show_presence = True

        action = model.action(s, softmax=False)
        #if score !=0: print(f"p_max: {model.p.max():.3f}, p_min: {model.p.min():.3f}")
        s_next, reward, done = env.step(action['list'])
        if step == 0:
            Q_pred = action['Q']
            pred = None
            if hasattr(model, "get_pred_from_numpy_action"):
                pred = model.get_pred_from_numpy_action(s, action['numpy'])

        s = s_next
        score += reward
        step += 1
    #env.render()
    location_history = env.robots[0].location_history
    costs = [ robot.cost for robot in env.robots ]
    amplitude = max(costs) - min(costs)
    
    if (
        not path_log is None
        and max(costs) < prev_cost_min
    ):
        env.draw(path=path_log + f'/location_history_{step_train}.png', pred=pred)

    model.config['env'] = config['env'] # Rollback the configuration of environment
    reward_avg = score / step

    return sum(costs), max(costs), amplitude, score, Q_pred, reward_avg

def train(args, config, model, agent):
    path_prev = args.path_prev
    if path_prev:
        state_dict = torch.load(path_prev)
        state_dict = {key: state_dict[key].to(device) for key in state_dict}
        model.load_state_dict(state_dict)
        # f = open(f"{path_log}/comment.txt", 'w')
        # f.write(path_prev)
        # f.close()
        name_model = path_prev.split('/')[-1]
        path_log = path_prev.rstrip(name_model)
    else:
        now = datetime.now()
        path_log = f"logs/[{str(args.conf)}][{now.strftime('%y%m%d')}][{now.strftime('%H-%M-%S')}]"
        os.makedirs(path_log, exist_ok=True)
        os.makedirs(path_log + '/codes', exist_ok=True)

        f = open(f"{path_log}/conf.txt", 'w')
        f.write(str(args))
        f.close()
    
    os.system(f"rsync -av --progress . {path_log}/codes/ --exclude=logs --exclude=logs_backup")

    step_prev = args.step_prev
    model.step_train = step_prev
    step_min = 0
    cost_min = float("inf")
    
    optimizer = AdamW(model.parameters(), lr=config['learning']['lr_start'], weight_decay=1e-3)
    logger_tool = LoggerTool(path_log)

    _time_10_step = time.time()

    model.simulator.start()
    model.simulator.save_to_replay_buffer(0.5 * config['learning']['size_replay_buffer'])
    
    model.reset_target()

    logger.info("###### Start training #####")
    for step_train in range(step_prev, config['learning']['step'] + 1):
        optimizer.param_groups[0]['lr'] = (
            config['learning']['lr_end'] 
            + config['learning']['lr_decay'] ** (step_train // config['learning']['lr_step']) * (
                config['learning']['lr_start'] - config['learning']['lr_end']
                )
            )
        _time_train = time.time()
        
        model.step_train = step_train
        optimizer.zero_grad()
        processed_batch = model.get_processed_batch()
        loss, info = agent.get_loss(processed_batch)
        loss.backward()
        optimizer.step()

        time_train = time.time() - _time_train
        if step_train % 10 == 0:
            # showing and writing loss
            time_10_step = time.time() - _time_10_step
            print(
                f"[{str(args.conf)}][{step_train}] lr: {optimizer.param_groups[0]['lr']:.5f} "
                + f"sqrt_loss_bellman: {info['loss_bellman'] ** 0.5:.3f}, "
                + f"loss_pred: {info['loss_cross_entropy']:.3f}, "
                + f"time_10_step: {time_10_step:.2f} "
                )
            _time_10_step = time.time()

        if step_train % 5 == 0:
            # adding new data to replay buffer
            model.simulator.save_to_replay_buffer(config['learning']['size_batch'])
            model.reset_target()

        if step_train % 25 == 0:
            # Test the performance of the training agent
            try:
                total_cost, max_cost, amplitude, score, Q_pred, reward_avg = test(config,
                                                                                  model,
                                                                                  step_train=step_train,
                                                                                  path_log=path_log,
                                                                                  prev_cost_min=cost_min)

                if max_cost < cost_min:
                    step_min = step_train
                    cost_min = max_cost

                print(
                    f"toptal_cost: {total_cost:.3f} "
                    + f"max_cost: {max_cost:.3f} "
                    + f"amplitude: {amplitude:.3f} "
                    + f"score: {score:.3f} "
                    + f"Q_pred: {Q_pred:.3f} "
                    + f"r_avg: {reward_avg:.3f} "
                    + f"step_min: {step_min} "
                    )
                logger_tool.write(
                    step_train, 
                    {
                        'sqrt_loss_bellman': info['loss_bellman'] ** 0.5,
                        'loss_pred': info['loss_cross_entropy'],
                        'training_time': time_train,
                        'total_cost': total_cost,
                        'max_cost': max_cost,
                        'amplitude': amplitude,
                        'score': score,
                        'reward_avg': reward_avg,
                        'Q_pred': Q_pred,
                        'step_min': step_min,
                        }
                    )

            except Exception as e:
                print(f"erorr happened in test train.test() as below: \n {e}")

        # saving model
        if step_train > 1 and step_train % 1000 == 0:
            state_dict = model.state_dict()
            state_dict_cpu = {key: state_dict[key].cpu() for key in state_dict}
            torch.save(state_dict_cpu, f"{path_log}/model_{step_train}.pt")

    # kill spawned processes
    model.simulator.terminate()

if __name__=='__main__':
    mp.set_start_method('spawn')


    parser = argparse.ArgumentParser(description='train_mtsp')
    parser.add_argument('--path_prev', type=str, 
                        help='path to previous model')
    parser.add_argument('--step_prev', type=int, default=0,
                        help='previous train step')
    parser.add_argument('--conf', type=str, 
                        help='conf to be used for training')
    parser.add_argument('--gpu', type=int, default=0, help='gpu idx')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.gpu}'

    config = __import__(f'conf.{args.conf}', fromlist=[None]).config
    model = __import__(f"models.{config['learning']['model']}", fromlist=[None]).Model(config, device).to(device)
    model.initialize_batch()
    #model.set_extra_gpus()

    agent = Agent(config)    
    
    try:
        train(args, config, model, agent)
        model.simulator.terminate()
    except KeyboardInterrupt:
        # terminate processes generated for collecting data
        model.simulator.terminate()




    
