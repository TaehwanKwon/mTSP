from pprint import pprint
from IPython import embed

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

from envs.mtsp_simple import MTSP, MTSPSimple
from models.gnn import Model
from agent.QLearning import Agent
from utils.simulator import Simulator

config = __import__('conf.conf-test', fromlist=[None]).config

model = Model(config=config)
model.initialize_batch()
agent = Agent(config=config)
simulator = Simulator(config=config, model=model)

simulator.save_to_replay_buffer(config['learning']['size_replay_buffer'])
for step_train in range(100):
    model.step_train = step_train
    processed_batch = model.get_processed_batch()
    loss = Agent.get_loss(processed_batch)
    print(f"loss: {loss}")

    
